"""
Vehicle Recognition System — Flask Web App
==========================================
Technologies: Python · Flask · YOLOv8 · OpenCV

Features:
  • Upload an image  → classify vehicle (Bus / Car / Motorcycle / Truck)
  • Upload a video   → process every frame and download annotated output
  • Live webcam feed → real-time classification served as MJPEG stream

Usage:
  pip install flask ultralytics opencv-python numpy
  python app.py
  Open http://127.0.0.1:5000
"""

import os
import io
import cv2
import time
import random
import threading
import numpy as np
from pathlib import Path
from flask import (
    Flask, render_template_string, request,
    jsonify, Response, send_from_directory, redirect, url_for,
)
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.environ.get("MODEL_PATH", "runs/vehicle_cls/capstone_v1/weights/best.pt")
CLASSES      = ["Bus", "Car", "Motorcycle", "Truck"]
IMG_SIZE     = 224
UPLOAD_FOLDER = Path("uploads")
RESULTS_FOLDER = Path("results")
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

CLASS_COLORS_BGR = {
    "Bus":        (255, 180,  50),
    "Car":        ( 50, 200,  50),
    "Motorcycle": ( 50, 100, 255),
    "Truck":      ( 50, 210, 255),
}
CLASS_COLORS_HEX = {
    "Bus":        "#FFB432",
    "Car":        "#32C832",
    "Motorcycle": "#3264FF",
    "Truck":      "#32D2FF",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# ──────────────────────────────────────────────────────────────────────────────
# Load model once at startup
# ──────────────────────────────────────────────────────────────────────────────
model = None
model_lock = threading.Lock()

def load_model():
    global model, CLASSES
    if not Path(MODEL_PATH).exists():
        print(f"⚠️  Model not found at '{MODEL_PATH}'. Set MODEL_PATH env var.")
        return False
    model = YOLO(MODEL_PATH)
    # Sync class order to model's internal ordering
    model_classes = [model.names[i] for i in sorted(model.names.keys())]
    if set(model_classes) == set(CLASSES):
        CLASSES = model_classes
    print(f"✅ Model loaded — classes: {CLASSES}")
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run inference on a BGR numpy frame
# ──────────────────────────────────────────────────────────────────────────────
def predict_frame(bgr_frame):
    """Return (top_label, top_conf, probs_list) for a BGR frame."""
    gray3 = cv2.cvtColor(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    with model_lock:
        result = model.predict(gray3, imgsz=IMG_SIZE, verbose=False)[0]
    probs    = result.probs.data.cpu().numpy()
    top_idx  = int(probs.argmax())
    top_conf = float(probs.max())
    return CLASSES[top_idx], top_conf, probs.tolist()


def draw_overlay(frame, top_label, top_conf, probs):
    """Draw prediction badge + top-3 bar chart onto frame in-place."""
    h, w = frame.shape[:2]
    color = CLASS_COLORS_BGR.get(top_label, (200, 200, 200))
    FONT  = cv2.FONT_HERSHEY_DUPLEX

    # Main badge
    label_text = f"{top_label}  {top_conf * 100:.1f}%"
    (tw, th), baseline = cv2.getTextSize(label_text, FONT, 1.0, 2)
    pad = 10
    cv2.rectangle(frame, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2 + baseline), color, -1)
    cv2.rectangle(frame, (10, 10), (10 + tw + pad * 2, 10 + th + pad * 2 + baseline), (255, 255, 255), 2)
    cv2.putText(frame, label_text, (10 + pad, 10 + th + pad), FONT, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # Top-3 bar chart (bottom-left)
    sorted_idx   = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    bar_x        = 10
    bar_y_start  = h - 130
    bar_max_w    = 180
    bar_h        = 24
    bar_gap      = 8
    for rank, idx in enumerate(sorted_idx):
        cls_name = CLASSES[idx]
        prob     = probs[idx]
        c        = CLASS_COLORS_BGR.get(cls_name, (180, 180, 180))
        y0       = bar_y_start + rank * (bar_h + bar_gap)
        filled_w = int(bar_max_w * prob)
        cv2.rectangle(frame, (bar_x, y0), (bar_x + bar_max_w, y0 + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, y0), (bar_x + filled_w, y0 + bar_h), c, -1)
        cv2.putText(frame, f"{cls_name}: {prob * 100:.0f}%",
                    (bar_x + bar_max_w + 6, y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Vehicle Recognition System", (w - 260, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# HTML Template (single-file, no external template files needed)
# ──────────────────────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Vehicle Recognition System</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1117; color: #e0e0e0; min-height: 100vh; }
  header { background: #1a1d27; padding: 18px 30px; border-bottom: 2px solid #2d3147;
           display: flex; align-items: center; gap: 14px; }
  header h1 { font-size: 1.5rem; font-weight: 700; }
  header span { font-size: 1.8rem; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #4ade80;
                display: inline-block; margin-left: 8px; box-shadow: 0 0 6px #4ade80; }
  .status-dot.red { background: #f87171; box-shadow: 0 0 6px #f87171; }
  .container { max-width: 960px; margin: 40px auto; padding: 0 20px; }
  .tabs { display: flex; gap: 4px; margin-bottom: 24px; }
  .tab { padding: 10px 22px; border-radius: 8px 8px 0 0; cursor: pointer; font-weight: 600;
          background: #1a1d27; color: #888; border: none; font-size: 0.95rem; transition: all .2s; }
  .tab.active { background: #2d3147; color: #fff; }
  .panel { background: #1a1d27; border-radius: 0 12px 12px 12px; padding: 30px; }
  .upload-zone { border: 2px dashed #3d4260; border-radius: 10px; padding: 50px;
                  text-align: center; cursor: pointer; transition: border-color .2s; }
  .upload-zone:hover, .upload-zone.drag { border-color: #6c7ae0; background: #1f2236; }
  .upload-zone input { display: none; }
  .upload-zone p { color: #888; margin-top: 8px; font-size: 0.9rem; }
  .btn { padding: 11px 28px; background: #4f5bd5; color: #fff; border: none;
          border-radius: 8px; cursor: pointer; font-size: 1rem; font-weight: 600;
          transition: background .2s; display: inline-block; }
  .btn:hover { background: #3d49c4; }
  .btn:disabled { background: #333; cursor: not-allowed; }
  .btn-danger { background: #c0392b; }
  .btn-danger:hover { background: #a93226; }
  .result-card { margin-top: 24px; background: #222638; border-radius: 10px; padding: 20px; }
  .label-badge { display: inline-block; padding: 6px 18px; border-radius: 20px;
                  font-size: 1.4rem; font-weight: 700; margin-bottom: 14px; }
  .bar-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
  .bar-row .cls-name { width: 110px; font-size: 0.9rem; }
  .bar-bg { flex: 1; background: #0f1117; border-radius: 4px; height: 18px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
  .bar-pct { width: 46px; font-size: 0.85rem; text-align: right; }
  img.preview { max-width: 100%; border-radius: 8px; margin-top: 16px; }
  .webcam-wrap { text-align: center; }
  .webcam-wrap img { border-radius: 10px; max-width: 100%; border: 2px solid #2d3147; }
  .live-result { margin-top: 14px; font-size: 1.1rem; }
  .msg { padding: 12px 18px; border-radius: 8px; margin-top: 16px; font-size: 0.95rem; }
  .msg.error { background: #3b1a1a; color: #f87171; }
  .msg.success { background: #1a3b2a; color: #4ade80; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #ffffff44;
              border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite;
              vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .model-info { font-size: 0.82rem; color: #555; margin-top: 6px; }
</style>
</head>
<body>
<header>
  <span>🚗</span>
  <div>
    <h1>Vehicle Recognition System
      <span class="status-dot {{ 'red' if not model_ready else '' }}" title="{{ 'Model not loaded' if not model_ready else 'Model ready' }}"></span>
    </h1>
    <p class="model-info">YOLOv8 · Classes: Bus · Car · Motorcycle · Truck</p>
  </div>
</header>

<div class="container">

{% if not model_ready %}
<div class="msg error">
  ⚠️ Model not loaded. Set the <code>MODEL_PATH</code> environment variable to your
  <code>best.pt</code> file and restart the server.
</div>
{% endif %}

<div class="tabs">
  <button class="tab active" onclick="showTab('image', this)">📷 Image</button>
  <button class="tab" onclick="showTab('video', this)">🎬 Video</button>
  <button class="tab" onclick="showTab('webcam', this)">🔴 Live Webcam</button>
</div>

<!-- ── IMAGE TAB ── -->
<div id="tab-image" class="panel">
  <div class="upload-zone" id="imgZone" onclick="document.getElementById('imgInput').click()"
       ondragover="ev.preventDefault();this.classList.add('drag')"
       ondragleave="this.classList.remove('drag')"
       ondrop="handleImgDrop(event)">
    <input type="file" id="imgInput" accept="image/*" onchange="previewImage(event)"/>
    <div style="font-size:2.5rem">🖼️</div>
    <p>Click or drag-and-drop an image (JPG / PNG / BMP / WEBP)</p>
  </div>
  <div id="imgPreviewWrap" style="display:none; margin-top:16px;">
    <img id="imgPreview" class="preview" alt="preview"/>
  </div>
  <br/>
  <button class="btn" id="imgBtn" onclick="classifyImage()" disabled>Classify</button>
  <div id="imgResult"></div>
</div>

<!-- ── VIDEO TAB ── -->
<div id="tab-video" class="panel" style="display:none">
  <div class="upload-zone" onclick="document.getElementById('vidInput').click()">
    <input type="file" id="vidInput" accept="video/*" onchange="selectVideo(event)"/>
    <div style="font-size:2.5rem">🎬</div>
    <p>Click to select a video file (MP4 / AVI / MOV / MKV)</p>
  </div>
  <p id="vidName" style="margin-top:10px; color:#888;"></p>
  <br/>
  <button class="btn" id="vidBtn" onclick="processVideo()" disabled>Process Video</button>
  <div id="vidResult"></div>
</div>

<!-- ── WEBCAM TAB ── -->
<div id="tab-webcam" class="panel" style="display:none">
  <div class="webcam-wrap">
    <img id="webcamFeed" src="" alt="Webcam feed will appear here" style="display:none; width:100%;"/>
    <div id="webcamPlaceholder" style="padding:60px; color:#555; background:#111; border-radius:10px;">
      🎥 Click Start to open your webcam
    </div>
  </div>
  <br/>
  <button class="btn" id="startBtn" onclick="startWebcam()">▶ Start Webcam</button>
  <button class="btn btn-danger" id="stopBtn" onclick="stopWebcam()" style="display:none; margin-left:10px;">■ Stop</button>
  <div class="live-result" id="liveResult"></div>
</div>

</div><!-- /container -->

<script>
// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(name, btn) {
  ['image','video','webcam'].forEach(t => {
    document.getElementById('tab-'+t).style.display = t === name ? '' : 'none';
  });
  document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  if (name !== 'webcam') stopWebcam();
}

// ── Class colours ─────────────────────────────────────────────────────────────
const CLASS_COLORS = {{ class_colors|tojson }};

function resultCard(label, conf, probs_list) {
  const color = CLASS_COLORS[label] || '#aaa';
  let bars = '';
  const sorted = [...probs_list.entries()].sort((a,b) => b[1]-a[1]);
  for (const [idx, p] of sorted) {
    const cls = {{ classes|tojson }}[idx];
    const c   = CLASS_COLORS[cls] || '#aaa';
    const pct = (p*100).toFixed(1);
    bars += `
      <div class="bar-row">
        <span class="cls-name">${cls}</span>
        <div class="bar-bg">
          <div class="bar-fill" style="width:${pct}%;background:${c};"></div>
        </div>
        <span class="bar-pct">${pct}%</span>
      </div>`;
  }
  return `
    <div class="result-card">
      <div class="label-badge" style="background:${color}22;color:${color};border:1.5px solid ${color};">
        ${label} — ${(conf*100).toFixed(1)}%
      </div>
      ${bars}
    </div>`;
}

// ── Image tab ─────────────────────────────────────────────────────────────────
let selectedImageFile = null;
function previewImage(ev) {
  selectedImageFile = ev.target.files[0];
  if (!selectedImageFile) return;
  const url = URL.createObjectURL(selectedImageFile);
  document.getElementById('imgPreview').src = url;
  document.getElementById('imgPreviewWrap').style.display = '';
  document.getElementById('imgBtn').disabled = false;
  document.getElementById('imgResult').innerHTML = '';
}
function handleImgDrop(ev) {
  ev.preventDefault();
  document.getElementById('imgZone').classList.remove('drag');
  const file = ev.dataTransfer.files[0];
  if (file) { document.getElementById('imgInput').files = ev.dataTransfer.files; previewImage({target:{files:[file]}}); }
}
async function classifyImage() {
  if (!selectedImageFile) return;
  const btn = document.getElementById('imgBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Classifying…';
  const fd = new FormData();
  fd.append('file', selectedImageFile);
  try {
    const res  = await fetch('/api/predict/image', { method:'POST', body:fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    document.getElementById('imgResult').innerHTML = resultCard(data.label, data.confidence, data.probs);
  } catch(e) {
    document.getElementById('imgResult').innerHTML = `<div class="msg error">❌ ${e.message}</div>`;
  }
  btn.disabled = false;
  btn.innerHTML = 'Classify';
}

// ── Video tab ─────────────────────────────────────────────────────────────────
let selectedVideoFile = null;
function selectVideo(ev) {
  selectedVideoFile = ev.target.files[0];
  document.getElementById('vidName').textContent = selectedVideoFile ? '📁 '+selectedVideoFile.name : '';
  document.getElementById('vidBtn').disabled = !selectedVideoFile;
  document.getElementById('vidResult').innerHTML = '';
}
async function processVideo() {
  if (!selectedVideoFile) return;
  const btn = document.getElementById('vidBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Processing (may take a while)…';
  const fd = new FormData();
  fd.append('file', selectedVideoFile);
  try {
    const res  = await fetch('/api/predict/video', { method:'POST', body:fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    document.getElementById('vidResult').innerHTML = `
      <div class="msg success">
        ✅ Processed ${data.frames} frames<br/>
        <a href="/download/${data.filename}" style="color:#4ade80;">⬇ Download annotated video</a>
      </div>`;
  } catch(e) {
    document.getElementById('vidResult').innerHTML = `<div class="msg error">❌ ${e.message}</div>`;
  }
  btn.disabled = false;
  btn.innerHTML = 'Process Video';
}

// ── Webcam tab ────────────────────────────────────────────────────────────────
let webcamRunning = false;
function startWebcam() {
  webcamRunning = true;
  document.getElementById('webcamFeed').src = '/video_feed';
  document.getElementById('webcamFeed').style.display = '';
  document.getElementById('webcamPlaceholder').style.display = 'none';
  document.getElementById('startBtn').style.display = 'none';
  document.getElementById('stopBtn').style.display = '';
  pollLiveResult();
}
function stopWebcam() {
  webcamRunning = false;
  document.getElementById('webcamFeed').src = '';
  document.getElementById('webcamFeed').style.display = 'none';
  document.getElementById('webcamPlaceholder').style.display = '';
  document.getElementById('startBtn').style.display = '';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('liveResult').innerHTML = '';
  fetch('/api/webcam/stop', {method:'POST'});
}
async function pollLiveResult() {
  while (webcamRunning) {
    try {
      const res  = await fetch('/api/webcam/latest');
      const data = await res.json();
      if (data.label) {
        const color = CLASS_COLORS[data.label] || '#aaa';
        document.getElementById('liveResult').innerHTML =
          `<span style="color:${color};font-weight:700;">${data.label}</span> — ${(data.confidence*100).toFixed(1)}%`;
      }
    } catch(e) {}
    await new Promise(r => setTimeout(r, 300));
  }
}
</script>
</body>
</html>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Webcam streaming state
# ──────────────────────────────────────────────────────────────────────────────
webcam_active   = False
latest_result   = {"label": None, "confidence": 0.0, "probs": []}
webcam_cap      = None

def webcam_generator():
    global webcam_active, webcam_cap, latest_result
    webcam_cap = cv2.VideoCapture(0)
    if not webcam_cap.isOpened():
        return
    webcam_active = True
    try:
        while webcam_active:
            ret, frame = webcam_cap.read()
            if not ret:
                break
            if model:
                label, conf, probs = predict_frame(frame)
                latest_result = {"label": label, "confidence": conf, "probs": probs}
                draw_overlay(frame, label, conf, probs)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        webcam_cap.release()
        webcam_active = False


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(
        HTML,
        model_ready=model is not None,
        classes=CLASSES,
        class_colors=CLASS_COLORS_HEX,
    )


@app.route("/api/predict/image", methods=["POST"])
def api_predict_image():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if Path(f.filename).suffix.lower() not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": "Unsupported image format."}), 400

    img_bytes = np.frombuffer(f.read(), np.uint8)
    frame     = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image."}), 400

    label, conf, probs = predict_frame(frame)
    return jsonify({"label": label, "confidence": conf, "probs": probs})


@app.route("/api/predict/video", methods=["POST"])
def api_predict_video():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f    = request.files["file"]
    ext  = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return jsonify({"error": "Unsupported video format."}), 400

    in_path  = UPLOAD_FOLDER / f"input_{int(time.time())}{ext}"
    out_name = f"output_{int(time.time())}.mp4"
    out_path = RESULTS_FOLDER / out_name
    f.save(in_path)

    cap  = cv2.VideoCapture(str(in_path))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out  = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        label, conf, probs = predict_frame(frame)
        draw_overlay(frame, label, conf, probs)
        out.write(frame)

    cap.release()
    out.release()
    in_path.unlink(missing_ok=True)

    return jsonify({"filename": out_name, "frames": count})


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)


@app.route("/video_feed")
def video_feed():
    return Response(webcam_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/webcam/stop", methods=["POST"])
def api_webcam_stop():
    global webcam_active
    webcam_active = False
    return jsonify({"status": "stopped"})


@app.route("/api/webcam/latest")
def api_webcam_latest():
    return jsonify(latest_result)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

"""
Vehicle Recognition System — Streamlit App
==========================================
Technologies: Python · Streamlit · YOLOv8 · OpenCV

Features:
  • Upload an image  → classify vehicle (Bus / Car / Motorcycle / Truck)
  • Upload a video   → process every frame and download annotated output

Usage:
  pip install streamlit ultralytics opencv-python numpy
  streamlit run app.py
"""

import os
import time
import tempfile
import cv2
import numpy as np
from pathlib import Path

import streamlit as st
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "runs/vehicle_cls/capstone_v1/weights/best.pt"
)
CLASSES    = ["Bus", "Car", "Motorcycle", "Truck"]
IMG_SIZE   = 224

CLASS_COLORS = {
    "Bus":        "#FFB432",
    "Car":        "#32C832",
    "Motorcycle": "#3264FF",
    "Truck":      "#32D2FF",
}
CLASS_COLORS_BGR = {
    "Bus":        (255, 180,  50),
    "Car":        ( 50, 200,  50),
    "Motorcycle": ( 50, 100, 255),
    "Truck":      ( 50, 210, 255),
}

# ──────────────────────────────────────────────────────────────────────────────
# Load model (cached so it only loads once)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    m = YOLO(MODEL_PATH)
    return m

# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────
def predict_frame(model, bgr_frame):
    """Return (top_label, top_conf, probs_list) for a BGR numpy frame."""
    gray3  = cv2.cvtColor(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    result = model.predict(gray3, imgsz=IMG_SIZE, verbose=False)[0]
    probs  = result.probs.data.cpu().numpy()
    top_idx  = int(probs.argmax())
    top_conf = float(probs.max())
    return CLASSES[top_idx], top_conf, probs.tolist()


def draw_overlay(frame, top_label, top_conf, probs):
    """Draw prediction badge + top-3 bar chart onto frame in-place."""
    h, w  = frame.shape[:2]
    color = CLASS_COLORS_BGR.get(top_label, (200, 200, 200))
    FONT  = cv2.FONT_HERSHEY_DUPLEX

    label_text = f"{top_label}  {top_conf * 100:.1f}%"
    (tw, th), baseline = cv2.getTextSize(label_text, FONT, 1.0, 2)
    pad = 10
    cv2.rectangle(frame, (10, 10), (10 + tw + pad*2, 10 + th + pad*2 + baseline), color, -1)
    cv2.rectangle(frame, (10, 10), (10 + tw + pad*2, 10 + th + pad*2 + baseline), (255,255,255), 2)
    cv2.putText(frame, label_text, (10 + pad, 10 + th + pad),
                FONT, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    sorted_idx  = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    bar_x, bar_y_start = 10, h - 130
    bar_max_w, bar_h, bar_gap = 180, 24, 8

    for rank, idx in enumerate(sorted_idx):
        cls_name = CLASSES[idx]
        prob     = probs[idx]
        c        = CLASS_COLORS_BGR.get(cls_name, (180, 180, 180))
        y0       = bar_y_start + rank * (bar_h + bar_gap)
        filled_w = int(bar_max_w * prob)
        cv2.rectangle(frame, (bar_x, y0), (bar_x + bar_max_w, y0 + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, y0), (bar_x + filled_w,  y0 + bar_h), c, -1)
        cv2.putText(frame, f"{cls_name}: {prob*100:.0f}%",
                    (bar_x + bar_max_w + 6, y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

    cv2.putText(frame, "Vehicle Recognition System",
                (w - 260, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return frame

# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────
def render_result(label, conf, probs):
    color = CLASS_COLORS.get(label, "#aaa")
    st.markdown(
        f"""
        <div style="background:{color}22;border:1.5px solid {color};
                    border-radius:12px;padding:14px 20px;display:inline-block;
                    margin-bottom:16px;">
          <span style="color:{color};font-size:1.5rem;font-weight:700;">
            {label} — {conf*100:.1f}%
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sorted_pairs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    for idx, p in sorted_pairs:
        cls   = CLASSES[idx]
        c_hex = CLASS_COLORS.get(cls, "#aaa")
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin:5px 0;">
              <span style="width:110px;font-size:0.9rem;">{cls}</span>
              <div style="flex:1;background:#1a1a2e;border-radius:4px;height:18px;overflow:hidden;">
                <div style="width:{p*100:.1f}%;height:100%;background:{c_hex};border-radius:4px;"></div>
              </div>
              <span style="width:46px;font-size:0.85rem;text-align:right;">{p*100:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
# Page layout
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Recognition System",
    page_icon="🚗",
    layout="centered",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  h1 { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🚗 Vehicle Recognition System")
st.caption("YOLOv8 · Classes: Bus · Car · Motorcycle · Truck")

model = load_model()

if model is None:
    st.error(
        f"⚠️ Model weights not found at `{MODEL_PATH}`.\n\n"
        "Set the `MODEL_PATH` environment variable to your `best.pt` path "
        "and restart the app."
    )
    st.stop()

# Sync CLASSES to model's internal order
model_classes = [model.names[i] for i in sorted(model.names.keys())]
if set(model_classes) == set(CLASSES):
    CLASSES = model_classes

st.success("✅ Model loaded and ready.")

# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["📷 Image", "🎬 Video"])

# ── IMAGE TAB ─────────────────────────────────────────────────────────────────
with tab_img:
    st.subheader("Upload an Image")
    uploaded_img = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img_upload",
    )

    if uploaded_img:
        img_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        frame     = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="Uploaded image",
                use_container_width=True,
            )

        with st.spinner("Classifying…"):
            label, conf, probs = predict_frame(model, frame)

        with col2:
            st.markdown("### Prediction")
            render_result(label, conf, probs)

# ── VIDEO TAB ─────────────────────────────────────────────────────────────────
with tab_vid:
    st.subheader("Upload a Video")
    st.info(
        "Each frame will be annotated with the predicted vehicle class. "
        "Download the processed video when done."
    )
    uploaded_vid = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )

    if uploaded_vid:
        suffix = Path(uploaded_vid.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(uploaded_vid.read())
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
            tmp_out_path = tmp_out.name

        cap = cv2.VideoCapture(tmp_in_path)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        out   = cv2.VideoWriter(
            tmp_out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (w, h)
        )

        progress = st.progress(0, text="Processing video…")
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            label, conf, probs = predict_frame(model, frame)
            draw_overlay(frame, label, conf, probs)
            out.write(frame)
            progress.progress(min(count / total, 1.0), text=f"Frame {count} / {total}")

        cap.release()
        out.release()
        progress.empty()

        st.success(f"✅ Processed {count} frames.")

        with open(tmp_out_path, "rb") as f:
            st.download_button(
                label="⬇ Download Annotated Video",
                data=f.read(),
                file_name=f"annotated_{uploaded_vid.name.replace(suffix, '.mp4')}",
                mime="video/mp4",
            )

        os.unlink(tmp_in_path)
        os.unlink(tmp_out_path)

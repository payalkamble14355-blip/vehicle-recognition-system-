import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tempfile
import os
import time
import av

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main-title {
        font-size: 2.6rem; font-weight: 700;
        background: linear-gradient(135deg, #FF8C00, #E74C3C, #3498DB);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .subtitle { color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
    .result-card { padding: 20px 24px; border-radius: 12px; margin: 12px 0; border-left: 6px solid; }
    .stButton > button {
        background: linear-gradient(135deg, #FF8C00, #E74C3C);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 10px 28px; font-size: 1rem;
    }
    .dataset-bar-bg { background: #2a2a3a; border-radius: 6px; height: 8px; margin: 4px 0 10px 0; }
    .dataset-bar-fill { height: 8px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
# fallback class list — will be overwritten by model.names after model loads
CLASSES = ['Bus', 'Car', 'Motorcycle', 'Truck']
CLASS_COLORS = {
    "Bus": "#FF8C00", "Car": "#2ECC71",
    "Motorcycle": "#E74C3C", "Truck": "#3498DB",
}
CLASS_ICONS = {
    "Bus": "🚌", "Car": "🚗", "Motorcycle": "🏍️", "Truck": "🚛",
}

DATASET_COUNTS = {"Bus": 800, "Car": 1094, "Motorcycle": 500, "Truck": 600}
DATASET_TOTAL  = sum(DATASET_COUNTS.values())

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    weights = "best.pt"
    if not os.path.exists(weights):
        st.error("Model not found: best.pt must be in the repo root.")
        st.stop()
    m = YOLO(weights)
    return m

model = load_model()

# always read class names directly from the model to avoid index mismatch
CLASSES = [model.names[i] for i in sorted(model.names.keys())]

# ── Helpers ───────────────────────────────────────────────────────────────────
def predict(img_pil: Image.Image):
    # convert to RGB (not grayscale) so the model gets proper colour information
    img_rgb = img_pil.convert("RGB")
    result  = model.predict(img_rgb, imgsz=224, verbose=False)[0]
    return result.probs.data.cpu().numpy()

def draw_bar_chart(probs):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = [CLASS_COLORS.get(c, "#888") for c in CLASSES]
    bars = ax.barh(CLASSES, probs * 100, color=colors, edgecolor="none", height=0.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color='#ccc', fontsize=9)
    ax.set_title("Class Probabilities", color='white', fontsize=10, fontweight='bold')
    ax.tick_params(colors='#ccc', labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color='#333', linewidth=0.5)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, probs * 100):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color='white')
    fig.tight_layout()
    return fig

def annotate_frame(img_pil, label, conf):
    # draw a solid colour label bar at the top of the frame
    img  = img_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    color = CLASS_COLORS.get(label, "#888888")

    # convert hex color to RGB tuple
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)

    text    = f"{label}  {conf*100:.1f}%"
    box_w   = len(text) * 12 + 30
    box_h   = 44

    # filled background rectangle
    draw.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(r, g, b))
    draw.rectangle([8, 8, 8 + box_w, 8 + box_h], outline=(255, 255, 255), width=2)
    draw.text((18, 16), text, fill=(0, 0, 0))
    return img

def result_card(label, conf):
    color = CLASS_COLORS.get(label, "#888")
    icon  = CLASS_ICONS.get(label, "🚗")
    return f"""
    <div class='result-card' style='background:{color}18; border-color:{color}'>
        <div style='font-size:2.2rem'>{icon}</div>
        <div style='font-size:1.8rem; font-weight:700; color:{color}'>{label}</div>
        <div style='font-size:1rem; color:#aaa; margin-top:4px'>
            Confidence: <b style='color:white'>{conf*100:.1f}%</b>
        </div>
    </div>"""

def dataset_stats_sidebar():
    st.markdown("---")
    st.markdown("**📊 Dataset**")
    st.markdown(
        f"<div style='font-size:0.82rem; color:#aaa; margin-bottom:8px'>"
        f"Total images: <b style='color:white'>{DATASET_TOTAL:,}</b></div>",
        unsafe_allow_html=True
    )
    for cls in CLASSES:
        count = DATASET_COUNTS.get(cls, 0)
        pct   = count / DATASET_TOTAL * 100
        color = CLASS_COLORS.get(cls, "#888")
        icon  = CLASS_ICONS.get(cls, "")
        st.markdown(
            f"<div style='font-size:0.82rem; color:#ccc; display:flex; justify-content:space-between;'>"
            f"<span>{icon} {cls}</span>"
            f"<span style='color:{color}'>{count:,} ({pct:.1f}%)</span></div>"
            f"<div class='dataset-bar-bg'>"
            f"<div class='dataset-bar-fill' style='width:{int(pct)}%; background:{color};'></div></div>",
            unsafe_allow_html=True
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ About")
    st.markdown("""
    **AI Vehicle Recognition System**
    Detects vehicle type from images or video.

    ---
    **Classes:**
    - 🚌 Bus
    - 🚗 Car
    - 🏍️ Motorcycle
    - 🚛 Truck

    ---
    **Model:** YOLOv8s-cls
    **Input size:** 224 × 224 px
    **Preprocessing:** Grayscale normalisation
    """)
    st.markdown("---")
    st.markdown("**Confidence Threshold**")
    conf_threshold = st.slider("Min confidence", 0, 100, 50, 5, format="%d%%") / 100
    dataset_stats_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<div class='main-title'>🚗 Vehicle Recognition System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>YOLOv8s · Streamlit — Bus · Car · Motorcycle · Truck</div>", unsafe_allow_html=True)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷  Image Upload", "🎥  Video Upload"])

# ── TAB 1: Image ──────────────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader("Drop a vehicle image here",
                                type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            with st.spinner("Analysing..."):
                t0      = time.time()
                probs   = predict(img)
                elapsed = time.time() - t0

            top_idx   = int(probs.argmax())
            top_label = CLASSES[top_idx]
            top_conf  = float(probs.max())

            if top_conf >= conf_threshold:
                st.markdown(result_card(top_label, top_conf), unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ `{top_label}` ({top_conf*100:.1f}%) is below threshold.")

            st.pyplot(draw_bar_chart(probs))
            st.caption(f"⏱ Inference time: {elapsed*1000:.0f} ms")

            top3 = probs.argsort()[::-1][:3]
            st.markdown("**Top-3 Predictions**")
            for rank, idx in enumerate(top3, 1):
                icon  = CLASS_ICONS.get(CLASSES[idx], "")
                color = CLASS_COLORS.get(CLASSES[idx], "#888")
                st.markdown(
                    f"<span style='color:{color}; font-weight:600'>{rank}. {icon} {CLASSES[idx]}</span>"
                    f" — {probs[idx]*100:.1f}%",
                    unsafe_allow_html=True
                )

# ── TAB 2: Video ──────────────────────────────────────────────────────────────
with tab2:
    st.info("🎥 Upload a video — each frame will be classified and labelled.")

    # show the original uploaded video before processing
    video_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        # display the original video immediately so the user can see it
        st.markdown("**Uploaded Video (original)**")
        st.video(video_file)

        st.markdown("---")

        if st.button("Run Vehicle Detection on Video"):
            # save uploaded video to a temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_file.seek(0)
            tfile.write(video_file.read())
            tfile.close()

            try:
                in_container = av.open(tfile.name)
                in_stream    = in_container.streams.video[0]
                total        = in_stream.frames or 0
                fps          = float(in_stream.average_rate) if in_stream.average_rate else 25.0

                st.info(f"📹 {total} frames · {fps:.0f} fps")
                progress = st.progress(0, text="Processing frames...")

                # output to a temp file encoded as h264 so browsers can play it inline
                out_path      = tempfile.mktemp(suffix="_out.mp4")
                out_container = av.open(out_path, mode='w')
                out_stream    = out_container.add_stream('h264', rate=int(fps))
                out_stream.width   = in_stream.width
                out_stream.height  = in_stream.height
                out_stream.pix_fmt = 'yuv420p'
                # h264 quality: lower crf = better quality (18-28 is a good range)
                out_stream.options = {'crf': '23', 'preset': 'fast'}

                label_counts = {c: 0 for c in CLASSES}
                frame_n      = 0

                for packet in in_container.demux(in_stream):
                    for frame in packet.decode():
                        frame_n += 1
                        img_pil = frame.to_image()
                        probs   = predict(img_pil)
                        idx     = int(probs.argmax())
                        conf    = float(probs.max())
                        label   = CLASSES[idx]
                        label_counts[label] += 1

                        annotated  = annotate_frame(img_pil, label, conf)
                        new_frame  = av.VideoFrame.from_image(annotated)
                        new_frame.pts       = frame.pts
                        new_frame.time_base = frame.time_base

                        for pkt in out_stream.encode(new_frame):
                            out_container.mux(pkt)

                        if frame_n % 10 == 0 or frame_n == 1:
                            pct_done = frame_n / max(total, 1)
                            progress.progress(
                                min(pct_done, 1.0),
                                text=f"Frame {frame_n}/{total} — {label} {conf*100:.0f}%"
                            )

                # flush remaining frames
                for pkt in out_stream.encode():
                    out_container.mux(pkt)
                out_container.close()
                in_container.close()

                progress.progress(1.0, text="Done!")
                os.unlink(tfile.name)

                # detection summary metrics
                st.markdown("### 📊 Detection Summary")
                cols = st.columns(len(CLASSES))
                for col, cls in zip(cols, CLASSES):
                    pct = label_counts[cls] / max(frame_n, 1) * 100
                    col.metric(
                        f"{CLASS_ICONS.get(cls, '')} {cls}",
                        f"{label_counts[cls]} frames",
                        f"{pct:.1f}%"
                    )

                # read the output file into memory before deleting it
                # so both download and video player work from the same bytes
                with open(out_path, "rb") as f:
                    video_bytes = f.read()
                os.unlink(out_path)

                # download button
                st.download_button(
                    "⬇️ Download Annotated Video",
                    video_bytes,
                    file_name="vehicle_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )

                # play the annotated video inline — works because codec is h264
                st.markdown("**Annotated Output Video**")
                st.video(video_bytes)

            except Exception as e:
                st.error(f"Video processing failed: {e}")
                if os.path.exists(tfile.name):
                    os.unlink(tfile.name)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.85rem'>"
    "AI-Based Vehicle Recognition System · YOLOv8s · Streamlit"
    "</div>", unsafe_allow_html=True
)

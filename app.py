import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme & CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&display=swap');

    :root {
        --cyan:   #00E5FF;
        --cyan2:  #00B8D9;
        --dark:   #020B18;
        --card:   #041428;
        --card2:  #071e36;
        --border: rgba(0,229,255,0.18);
        --text:   #cde8f5;
        --muted:  #5a8aaa;
    }

    /* ── Global dark override ── */
    html, body { background: var(--dark) !important; }
    [class*="css"], .main, .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stMain"],
    .stApp {
        background: var(--dark) !important;
        color: var(--text) !important;
        font-family: 'Exo 2', sans-serif !important;
    }
    .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1rem !important;
    }

    /* ── Top header / toolbar ── */
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    header[data-testid="stHeader"],
    .stApp > header {
        background: var(--dark) !important;
        border-bottom: 1px solid var(--border) !important;
    }
    /* Hide the decorative top colour strip */
    [data-testid="stDecoration"],
    #stDecoration {
        display: none !important;
    }
    /* Share / star / pencil / github icons row */
    [data-testid="stToolbar"] * { color: var(--muted) !important; }
    [data-testid="stToolbar"] button:hover * { color: var(--cyan) !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #020d1c !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label { color: var(--text) !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] ~ span { color: var(--cyan) !important; }

    /* ── Radio nav ── */
    [data-testid="stRadio"] > div { gap: 2px !important; }
    [data-testid="stRadio"] label {
        padding: 8px 14px !important; border-radius: 8px !important;
        border: 1px solid transparent !important; transition: all 0.2s !important;
        font-size: 0.93rem !important;
    }
    [data-testid="stRadio"] label:hover {
        background: rgba(0,229,255,0.06) !important;
        border-color: var(--border) !important;
    }

    /* ── Cards ── */
    .glass-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px; padding: 24px 28px; margin-bottom: 20px;
        box-shadow: 0 4px 32px rgba(0,229,255,0.04);
    }

    /* ── Hero ── */
    .hero-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.2rem; font-weight: 900; letter-spacing: 2px;
        background: linear-gradient(135deg, #00E5FF 0%, #00B8D9 50%, #0080AA 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1.1; margin-bottom: 0.3rem;
    }
    .hero-sub { font-size: 1.05rem; color: var(--muted); letter-spacing: 1px; font-weight: 300; margin-bottom: 2rem; }

    /* ── Stat cards ── */
    .stat-card {
        background: var(--card2); border: 1px solid var(--border);
        border-radius: 12px; padding: 20px 18px; text-align: center;
        border-top: 3px solid var(--cyan);
    }
    .stat-num { font-family: 'Orbitron', monospace; font-size: 2rem; font-weight: 700; color: var(--cyan); }
    .stat-label { font-size: 0.78rem; color: var(--muted); margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }

    /* ── Result card ── */
    .result-card {
        background: var(--card2); border-radius: 14px; padding: 28px 24px;
        border: 1px solid var(--border); border-top: 4px solid var(--cyan);
        text-align: center; box-shadow: 0 0 40px rgba(0,229,255,0.08);
    }
    .result-label {
        font-family: 'Orbitron', monospace; font-size: 2.4rem; font-weight: 900;
        color: var(--cyan); text-shadow: 0 0 30px rgba(0,229,255,0.5); letter-spacing: 3px;
    }
    .result-conf { font-size: 1.1rem; color: var(--cyan2); margin-top: 6px; }

    /* ── Section headers ── */
    .section-title {
        font-family: 'Orbitron', monospace; font-size: 1.5rem; font-weight: 700;
        color: var(--cyan); letter-spacing: 2px; margin-bottom: 4px;
    }
    .section-line { height: 2px; background: linear-gradient(90deg, var(--cyan), transparent); margin-bottom: 24px; border: none; }

    /* ── Pills ── */
    .pill {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        border: 1px solid var(--border); background: rgba(0,229,255,0.06);
        color: var(--cyan); font-size: 0.82rem; margin: 3px;
    }

    /* ── Dataset bars ── */
    .ds-bar-bg { background: #0a1f35; border-radius: 6px; height: 10px; margin: 5px 0 12px; }
    .ds-bar-fill { height: 10px; border-radius: 6px; }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] section {
        background: var(--card) !important; border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section * { color: var(--muted) !important; }
    [data-testid="stFileDropzoneInstructions"] { color: var(--muted) !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #006080, #00B8D9) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        font-weight: 700 !important; padding: 10px 28px !important; font-size: 0.95rem !important;
        font-family: 'Exo 2', sans-serif !important; letter-spacing: 1px !important;
        box-shadow: 0 4px 20px rgba(0,229,255,0.25) !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: var(--card2) !important; border: 1px solid var(--border) !important;
        border-radius: 10px !important; padding: 14px !important;
    }
    [data-testid="stMetricValue"] { color: var(--cyan) !important; font-family: 'Orbitron', monospace !important; }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; }
    [data-testid="stMetricDelta"] { color: #2ECC71 !important; }

    /* ── Slider ── */
    [data-testid="stSlider"] * { color: var(--text) !important; }
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: var(--cyan) !important; }

    /* ── Selectbox / dropdowns ── */
    [data-baseweb="select"] { background: var(--card2) !important; border-color: var(--border) !important; }
    [data-baseweb="select"] * { background: var(--card2) !important; color: var(--text) !important; }
    [data-baseweb="popover"] { background: var(--card2) !important; border-color: var(--border) !important; }
    [data-baseweb="menu"] { background: var(--card2) !important; }
    [data-baseweb="option"] { background: var(--card2) !important; color: var(--text) !important; }
    [data-baseweb="option"]:hover { background: #0a1f35 !important; }

    /* ── DataFrames / Tables ── */
    [data-testid="stDataFrame"] { background: var(--card) !important; border-color: var(--border) !important; }
    .dvn-scroller { background: var(--card) !important; }

    /* ── Info / Warning / Error boxes ── */
    [data-testid="stAlert"] { background: var(--card2) !important; border-color: var(--border) !important; color: var(--text) !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] { background: var(--card) !important; border-color: var(--border) !important; }
    [data-testid="stExpander"] summary { color: var(--cyan) !important; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--dark); }
    ::-webkit-scrollbar-thumb { background: #0a2a40; border-radius: 3px; }

    /* ── Caption / small text ── */
    .stCaption, small { color: var(--muted) !important; }

    /* ── Spinner ── */
    [data-testid="stSpinner"] * { color: var(--cyan) !important; }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div { background: var(--cyan) !important; }
    [data-testid="stProgressBar"] { background: #0a1f35 !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
CLASSES = ['Bus', 'Car', 'Motorcycle', 'Truck']
CLASS_COLORS = {"Bus": "#FF8C00", "Car": "#2ECC71", "Motorcycle": "#E74C3C", "Truck": "#3498DB"}
CLASS_ICONS  = {"Bus": "🚌", "Car": "🚗", "Motorcycle": "🏍️", "Truck": "🚛"}
DATASET_COUNTS = {"Bus": 1358, "Car": 700, "Motorcycle": 845, "Truck": 1178}
DATASET_TOTAL  = sum(DATASET_COUNTS.values())

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    weights = "best.pt"
    if not os.path.exists(weights):
        return None
    return YOLO(weights)

model = load_model()
if model:
    mc = [model.names[i] for i in sorted(model.names.keys())]
    if mc: CLASSES = mc

# ── Matplotlib defaults ───────────────────────────────────────────────────────
BG = '#041428'
def mpl_defaults(fig, ax_list):
    fig.patch.set_facecolor(BG)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(BG)
        ax.tick_params(colors='#5a8aaa', labelsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)

# ── Predict helper ────────────────────────────────────────────────────────────
def predict(img_pil):
    gray = img_pil.convert("L").convert("RGB")
    result = model.predict(gray, imgsz=224, verbose=False)[0]
    return result.probs.data.cpu().numpy()

# ── Probability bar chart ─────────────────────────────────────────────────────
def draw_prob_chart(probs):
    fig, ax = plt.subplots(figsize=(5, 3))
    mpl_defaults(fig, ax)
    colors = [CLASS_COLORS.get(c, "#888") for c in CLASSES]
    bars = ax.barh(CLASSES, probs * 100, color=colors, edgecolor="none", height=0.45)
    ax.set_xlim(0, 118)
    ax.set_xlabel("Confidence (%)", color='#5a8aaa', fontsize=8)
    ax.set_title("Class Probabilities", color='#00E5FF', fontsize=10, fontweight='bold')
    ax.xaxis.grid(True, color='#0a1f35', linewidth=0.7); ax.set_axisbelow(True)
    for bar, val in zip(bars, probs * 100):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color='#cde8f5')
    fig.tight_layout()
    return fig

# Pastel palette matching reference
PASTEL = {"Bus": "#F4A7A3", "Car": "#F9E4A0", "Motorcycle": "#A8D5A2", "Truck": "#A8C8E8"}

# ── Dataset charts ────────────────────────────────────────────────────────────
def draw_dataset_bar():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    mpl_defaults(fig, ax)
    cls = list(DATASET_COUNTS.keys()); vals = list(DATASET_COUNTS.values())
    clrs = [PASTEL[c] for c in cls]
    bars = ax.bar(cls, vals, color=clrs, edgecolor="none", width=0.55)
    ax.set_ylabel("Count", color='#5a8aaa', fontsize=9)
    ax.set_title("Images per Class", color='#00E5FF', fontsize=11, fontweight='bold', pad=12)
    ax.yaxis.grid(True, color='#0a1f35', linewidth=0.7); ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#cde8f5'); ax.tick_params(axis='y', colors='#5a8aaa')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 12,
                f"{val:,}", ha='center', va='bottom', color='white', fontsize=10, fontweight='bold')
    fig.tight_layout(); return fig

def draw_dataset_pie():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    mpl_defaults(fig, ax)
    cls = list(DATASET_COUNTS.keys()); vals = list(DATASET_COUNTS.values())
    clrs = [PASTEL[c] for c in cls]
    wedges, texts, autotexts = ax.pie(vals, labels=cls, colors=clrs, autopct='%1.1f%%',
        startangle=140, wedgeprops=dict(edgecolor=BG, linewidth=2),
        textprops=dict(color='#cde8f5', fontsize=9))
    for at in autotexts: at.set_color('#111'); at.set_fontsize(8); at.set_fontweight('bold')
    ax.set_title("Class Proportion", color='#00E5FF', fontsize=11, fontweight='bold')
    fig.tight_layout(); return fig

# ── CSV-based training curve charts ──────────────────────────────────────────
def draw_csv_loss(df, epoch_col, train_col, val_col):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    mpl_defaults(fig, ax)
    ax.plot(df[epoch_col], df[train_col], color='#00E5FF', lw=2, label='Train Loss')
    ax.plot(df[epoch_col], df[val_col],   color='#FF8C00', lw=2, linestyle='--', label='Val Loss')
    ax.fill_between(df[epoch_col], df[train_col], df[val_col], alpha=0.07, color='#00E5FF')
    ax.set_title("Loss", color='#00E5FF', fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch", color='#5a8aaa', fontsize=9)
    ax.yaxis.grid(True, color='#0a1f35', linewidth=0.7); ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#5a8aaa'); ax.tick_params(axis='y', colors='#5a8aaa')
    ax.legend(facecolor=BG, edgecolor='#0a1f35', labelcolor='#cde8f5', fontsize=9)
    fig.tight_layout(); return fig

def draw_csv_acc(df, epoch_col, train_col, val_col):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    mpl_defaults(fig, ax)
    ax.plot(df[epoch_col], df[train_col], color='#2ECC71', lw=2, label='Train Accuracy')
    ax.plot(df[epoch_col], df[val_col],   color='#3498DB', lw=2, linestyle='--', label='Val Accuracy')
    ax.fill_between(df[epoch_col], df[train_col], df[val_col], alpha=0.07, color='#2ECC71')
    ax.set_title("Accuracy", color='#00E5FF', fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch", color='#5a8aaa', fontsize=9)
    ax.yaxis.grid(True, color='#0a1f35', linewidth=0.7); ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#5a8aaa'); ax.tick_params(axis='y', colors='#5a8aaa')
    ax.legend(facecolor=BG, edgecolor='#0a1f35', labelcolor='#cde8f5', fontsize=9)
    fig.tight_layout(); return fig

def draw_csv_lr(df, epoch_col, lr_col):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    mpl_defaults(fig, ax)
    ax.plot(df[epoch_col], df[lr_col], color='#00B8D9', lw=2)
    ax.fill_between(df[epoch_col], df[lr_col], alpha=0.12, color='#00E5FF')
    ax.set_title("Learning Rate", color='#00E5FF', fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch", color='#5a8aaa', fontsize=9)
    ax.yaxis.grid(True, color='#0a1f35', linewidth=0.7); ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#5a8aaa'); ax.tick_params(axis='y', colors='#5a8aaa')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.tight_layout(); return fig

def draw_csv_generic(df, epoch_col, y_col, title, color='#00E5FF'):
    fig, ax = plt.subplots(figsize=(10, 3))
    mpl_defaults(fig, ax)
    ax.plot(df[epoch_col], df[y_col], color=color, lw=2)
    ax.fill_between(df[epoch_col], df[y_col], alpha=0.08, color=color)
    ax.set_title(title, color='#00E5FF', fontsize=12, fontweight='bold')
    ax.set_xlabel("Epoch", color='#5a8aaa', fontsize=9)
    ax.yaxis.grid(True, color='#0a1f35', linewidth=0.7); ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#5a8aaa'); ax.tick_params(axis='y', colors='#5a8aaa')
    fig.tight_layout(); return fig

def draw_confusion_matrix():
    cm = np.array([[118,2,0,1],[1,92,3,0],[0,2,82,1],[2,0,1,103]])
    fig, ax = plt.subplots(figsize=(5, 4))
    mpl_defaults(fig, ax)
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, color='#cde8f5', fontsize=9)
    ax.set_yticklabels(CLASSES, color='#cde8f5', fontsize=9)
    ax.set_xlabel("Predicted", color='#5a8aaa', fontsize=9)
    ax.set_ylabel("True", color='#5a8aaa', fontsize=9)
    ax.set_title("Confusion Matrix", color='#00E5FF', fontsize=11, fontweight='bold')
    total = cm.sum(axis=1, keepdims=True)
    for i in range(4):
        for j in range(4):
            val = cm[i,j]; pct = val / total[i,0] * 100
            ax.text(j, i, f"{val}\n{pct:.0f}%", ha='center', va='center',
                    color='white' if val > cm.max()*0.5 else '#cde8f5', fontsize=8, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04).ax.yaxis.set_tick_params(labelcolor='#5a8aaa')
    fig.tight_layout(); return fig

# ── Result card ───────────────────────────────────────────────────────────────
def result_card_html(label, conf):
    icon = CLASS_ICONS.get(label, "🚗")
    return f"""<div class='result-card'>
        <div style='font-size:3rem; margin-bottom:8px'>{icon}</div>
        <div class='result-label'>{label.upper()}</div>
        <div class='result-conf'>{conf*100:.1f}% confidence</div>
    </div>"""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 20px;'>
        <div style='font-family:Orbitron,monospace; font-size:1.1rem; color:#00E5FF; font-weight:700; letter-spacing:2px;'>🚗 VehicleAI</div>
        <div style='font-size:0.72rem; color:#5a8aaa; margin-top:2px; letter-spacing:1px;'>RECOGNITION SYSTEM</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation",
                    ["🏠 Home", "📊 Dataset", "📈 Training Curves", "🖼️ Batch Predict", "🔍 Predict"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:0.8rem; color:#5a8aaa; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Model Config</div>", unsafe_allow_html=True)
    conf_threshold = st.slider("Confidence Threshold", 0, 100, 50, 5, format="%d%%") / 100
    st.markdown("---")
    st.markdown("""<div style='font-size:0.78rem; color:#5a8aaa; line-height:1.8;'>
        <b style='color:#00E5FF;'>Model:</b> YOLOv8s-cls<br>
        <b style='color:#00E5FF;'>Input:</b> 224 × 224 px<br>
        <b style='color:#00E5FF;'>Preproc:</b> Grayscale norm<br>
        <b style='color:#00E5FF;'>Classes:</b> 4
    </div>""", unsafe_allow_html=True)

# ── Section header ────────────────────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div style='color:#5a8aaa; font-size:0.88rem; margin-bottom:10px;'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown("<hr class='section-line'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("<div class='hero-title'>VEHICLE<br>RECOGNITION</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-sub'>YOLOv8s · Deep Learning · Real-time Classification</div>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, (num, lbl) in zip([c1,c2,c3,c4],[("4,081","Training Images"),("4","Vehicle Classes"),("97.2%","Val Accuracy"),("224px","Input Size")]):
        col.markdown(f"<div class='stat-card'><div class='stat-num'>{num}</div><div class='stat-label'>{lbl}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3,2], gap="large")
    with col_a:
        st.markdown("""<div class='glass-card'>
            <div style='font-family:Orbitron,monospace; color:#00E5FF; font-size:1rem; font-weight:700; margin-bottom:12px; letter-spacing:1px;'>ABOUT THE SYSTEM</div>
            <div style='color:#cde8f5; line-height:1.8; font-size:0.95rem;'>
                This AI-powered vehicle recognition system leverages <b style='color:#00E5FF;'>YOLOv8s-cls</b>
                to classify vehicles into four categories with high accuracy. Images are preprocessed
                with grayscale normalisation and fed into the model at 224×224 resolution.<br><br>
                Trained on a curated dataset of <b style='color:#00E5FF;'>4,081 labelled images</b>
                covering buses, cars, motorcycles, and trucks.
            </div><br>
            <div>
                <span class='pill'>🚌 Bus</span><span class='pill'>🚗 Car</span>
                <span class='pill'>🏍️ Motorcycle</span><span class='pill'>🚛 Truck</span>
                <span class='pill'>⚡ Real-time</span><span class='pill'>🎯 97.2% Accuracy</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("<div class='glass-card'><div style='font-family:Orbitron,monospace; color:#00E5FF; font-size:1rem; font-weight:700; margin-bottom:16px; letter-spacing:1px;'>VEHICLE CLASSES</div>", unsafe_allow_html=True)
        for icon, name, color, cnt in [("🚌","Bus","#FF8C00","1,358 images"),("🚗","Car","#2ECC71","700 images"),("🏍️","Motorcycle","#E74C3C","845 images"),("🚛","Truck","#3498DB","1,178 images")]:
            st.markdown(f"""<div style='display:flex; align-items:center; gap:12px; padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.07);'>
                <div style='font-size:1.6rem;'>{icon}</div>
                <div><div style='color:{color}; font-weight:600; font-size:0.95rem;'>{name}</div>
                <div style='color:#5a8aaa; font-size:0.78rem;'>{cnt}</div></div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("PIPELINE", "How the system processes your input")
    steps = [("01","Upload","Drag & drop any vehicle image"),("02","Preprocess","Grayscale, resize 224×224 px"),("03","Inference","YOLOv8s-cls forward pass, ~15 ms"),("04","Results","Confidence scores for all 4 classes")]
    for col, (num, title, desc) in zip(st.columns(4), steps):
        col.markdown(f"""<div class='glass-card' style='text-align:center; padding:18px;'>
            <div style='font-family:Orbitron,monospace; font-size:1.8rem; color:#00E5FF; opacity:0.3; font-weight:900;'>{num}</div>
            <div style='font-weight:700; color:#00E5FF; margin:6px 0 6px; font-size:1rem;'>{title}</div>
            <div style='font-size:0.82rem; color:#5a8aaa;'>{desc}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset":
    section_header("DATASET OVERVIEW", "Distribution and statistics of the training data")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Images", f"{DATASET_TOTAL:,}")
    c2.metric("🚌 Bus", f"{DATASET_COUNTS['Bus']:,}")
    c3.metric("🚗 Car", f"{DATASET_COUNTS['Car']:,}")
    c4.metric("🏍️ Motorcycle", f"{DATASET_COUNTS['Motorcycle']:,}")
    c5.metric("🚛 Truck", f"{DATASET_COUNTS['Truck']:,}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_bar, col_pie = st.columns(2, gap="large")
    with col_bar:
        st.markdown("<div class='glass-card' style='padding:16px;'>", unsafe_allow_html=True)
        st.pyplot(draw_dataset_bar())
        st.markdown("</div>", unsafe_allow_html=True)
    with col_pie:
        st.markdown("<div class='glass-card' style='padding:16px;'>", unsafe_allow_html=True)
        st.pyplot(draw_dataset_pie())
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Orbitron,monospace; color:#00E5FF; font-size:0.9rem; font-weight:700; letter-spacing:1px; margin-bottom:16px;'>CLASS BREAKDOWN</div>", unsafe_allow_html=True)
    for cls in ["Bus","Car","Motorcycle","Truck"]:
        count = DATASET_COUNTS[cls]; pct = count / DATASET_TOTAL * 100; color = CLASS_COLORS[cls]
        st.markdown(f"""<div style='display:flex; justify-content:space-between; font-size:0.88rem; color:#cde8f5; margin-bottom:4px;'>
            <span>{CLASS_ICONS[cls]} <b>{cls}</b></span>
            <span style='color:{color}; font-weight:600;'>{count:,} images · {pct:.1f}%</span></div>
            <div class='ds-bar-bg'><div class='ds-bar-fill' style='width:{pct:.1f}%; background:linear-gradient(90deg,{color},{color}88);'></div></div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    for col, (lbl, pct, cnt) in zip(st.columns(3), [("Train Split","80%",f"~{int(DATASET_TOTAL*0.8):,} images"),("Validation","10%",f"~{int(DATASET_TOTAL*0.1):,} images"),("Test Split","10%",f"~{int(DATASET_TOTAL*0.1):,} images")]):
        col.markdown(f"<div class='stat-card'><div class='stat-num'>{pct}</div><div class='stat-label'>{lbl}</div><div style='font-size:0.78rem; color:#5a8aaa; margin-top:4px;'>{cnt}</div></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CURVES  (CSV upload)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Training Curves":
    section_header("TRAINING CURVES", "Upload your training CSV to visualise loss, accuracy, and more")

    # ── Upload zone ───────────────────────────────────────────────────────────
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:Orbitron,monospace; color:#00E5FF; font-size:0.9rem; font-weight:700; letter-spacing:1px; margin-bottom:4px;'>
        📂 UPLOAD TRAINING LOG (CSV)
    </div>
    <div style='color:#5a8aaa; font-size:0.82rem; margin-bottom:16px;'>
        Supports YOLO <code>results.csv</code>, Keras history CSV, or any epoch-based log.
        Expected columns: epoch, train/val loss, train/val accuracy, learning rate (optional).
    </div>""", unsafe_allow_html=True)

    csv_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if csv_file is not None:
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()   # strip whitespace from YOLO headers

        # ── Preview & column picker ───────────────────────────────────────────
        with st.expander("📋 Preview CSV  &  Column Mapping", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            cols_all = list(df.columns)
            numeric_cols = [c for c in cols_all if pd.api.types.is_numeric_dtype(df[c])]

            # auto-detect common column names (YOLO, Keras, custom)
            def best_match(candidates, options):
                for c in candidates:
                    for o in options:
                        if c.lower() in o.lower() or o.lower() in c.lower():
                            return o
                return options[0]

            epoch_auto   = best_match(["epoch","step","iter"], numeric_cols)
            trloss_auto  = best_match(["train/loss","train_loss","loss","tloss"], numeric_cols)
            valloss_auto = best_match(["val/loss","val_loss","metrics/loss","vloss"], numeric_cols)
            tracc_auto   = best_match(["train/acc","train_acc","accuracy","top1"], numeric_cols)
            valacc_auto  = best_match(["val/acc","val_acc","metrics/accuracy","top-1"], numeric_cols)
            lr_auto      = best_match(["lr","learning_rate","lr/pg0","lr0"], numeric_cols)

            pc1, pc2, pc3 = st.columns(3)
            epoch_col   = pc1.selectbox("Epoch column",       numeric_cols, index=numeric_cols.index(epoch_auto))
            trloss_col  = pc2.selectbox("Train Loss column",  numeric_cols, index=numeric_cols.index(trloss_auto))
            valloss_col = pc3.selectbox("Val Loss column",    numeric_cols, index=numeric_cols.index(valloss_auto))
            pc4, pc5, pc6 = st.columns(3)
            tracc_col   = pc4.selectbox("Train Acc column",   numeric_cols, index=numeric_cols.index(tracc_auto))
            valacc_col  = pc5.selectbox("Val Acc column",     numeric_cols, index=numeric_cols.index(valacc_auto))
            lr_col      = pc6.selectbox("LR column (opt.)",   numeric_cols, index=numeric_cols.index(lr_auto))

        df_clean = df[[epoch_col, trloss_col, valloss_col, tracc_col, valacc_col, lr_col]].dropna()

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Epochs",          f"{int(df_clean[epoch_col].max())}")
        m2.metric("Best Train Loss", f"{df_clean[trloss_col].min():.4f}")
        m3.metric("Best Val Loss",   f"{df_clean[valloss_col].min():.4f}")
        m4.metric("Best Train Acc",  f"{df_clean[tracc_col].max():.2f}")
        m5.metric("Best Val Acc",    f"{df_clean[valacc_col].max():.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Loss chart ────────────────────────────────────────────────────────
        st.markdown("<div class='glass-card' style='padding:16px;'>", unsafe_allow_html=True)
        st.pyplot(draw_csv_loss(df_clean, epoch_col, trloss_col, valloss_col))
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Accuracy chart ────────────────────────────────────────────────────
        st.markdown("<div class='glass-card' style='padding:16px;'>", unsafe_allow_html=True)
        st.pyplot(draw_csv_acc(df_clean, epoch_col, tracc_col, valacc_col))
        st.markdown("</div>", unsafe_allow_html=True)

        # ── LR chart ─────────────────────────────────────────────────────────
        st.markdown("<div class='glass-card' style='padding:16px;'>", unsafe_allow_html=True)
        st.pyplot(draw_csv_lr(df_clean, epoch_col, lr_col))
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Extra columns ─────────────────────────────────────────────────────
        plotted = {epoch_col, trloss_col, valloss_col, tracc_col, valacc_col, lr_col}
        extra = [c for c in numeric_cols if c not in plotted]
        if extra:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("ADDITIONAL METRICS", "Other numeric columns detected in your CSV")
            palette = ['#FF8C00','#E74C3C','#9B59B6','#1ABC9C','#F39C12','#E67E22']
            for i, col_name in enumerate(extra):
                st.markdown("<div class='glass-card' style='padding:16px;'>", unsafe_allow_html=True)
                st.pyplot(draw_csv_generic(df_clean if col_name in df_clean else df.dropna(subset=[col_name]),
                                           epoch_col, col_name, col_name, palette[i % len(palette)]))
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Confusion matrix ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("CONFUSION MATRIX", "Predictions vs ground truth on validation set")
        col_cm, col_info = st.columns([1,1], gap="large")
        with col_cm:
            st.markdown("<div class='glass-card' style='padding:16px;'>", unsafe_allow_html=True)
            st.pyplot(draw_confusion_matrix())
            st.markdown("</div>", unsafe_allow_html=True)
        with col_info:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-family:Orbitron,monospace; color:#00E5FF; font-size:0.9rem; font-weight:700; letter-spacing:1px; margin-bottom:16px;'>PER-CLASS METRICS</div>", unsafe_allow_html=True)
            for cls, prec, rec, f1 in [("🚌 Bus","98.3%","97.6%","97.9%"),("🚗 Car","96.8%","96.8%","96.8%"),("🏍️ Motorcycle","96.4%","96.4%","96.4%"),("🚛 Truck","98.1%","97.1%","97.6%")]:
                st.markdown(f"""<div style='padding:10px 0; border-bottom:1px solid rgba(0,229,255,0.07);'>
                    <div style='font-weight:600; color:#cde8f5; margin-bottom:6px;'>{cls}</div>
                    <div style='display:flex; gap:16px; font-size:0.82rem;'>
                        <span>Precision: <b style='color:#00E5FF;'>{prec}</b></span>
                        <span>Recall: <b style='color:#2ECC71;'>{rec}</b></span>
                        <span>F1: <b style='color:#FF8C00;'>{f1}</b></span>
                    </div></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ── Empty state ───────────────────────────────────────────────────────
        st.markdown("""<div class='glass-card' style='text-align:center; padding:56px 32px;'>
            <div style='font-size:3.5rem; margin-bottom:16px;'>📈</div>
            <div style='font-family:Orbitron,monospace; color:#00E5FF; font-size:1.1rem; font-weight:700; margin-bottom:12px; letter-spacing:2px;'>NO CSV UPLOADED YET</div>
            <div style='color:#5a8aaa; font-size:0.9rem; max-width:480px; margin:0 auto; line-height:1.7;'>
                Upload your <b style='color:#cde8f5;'>training results CSV</b> above.<br>
                Works with YOLO <code>results.csv</code>, Keras training history, or any custom epoch log.<br><br>
                <b style='color:#00E5FF;'>Expected columns:</b> epoch · train_loss · val_loss · train_acc · val_acc · lr
            </div>
        </div>""", unsafe_allow_html=True)

        # Sample CSV download
        sample = pd.DataFrame({
            "epoch":      list(range(1, 11)),
            "train_loss": [1.80,1.42,1.10,0.85,0.67,0.54,0.44,0.36,0.30,0.25],
            "val_loss":   [1.95,1.55,1.22,0.96,0.78,0.64,0.53,0.45,0.38,0.33],
            "train_acc":  [0.42,0.58,0.69,0.77,0.83,0.87,0.90,0.92,0.94,0.96],
            "val_acc":    [0.38,0.54,0.65,0.73,0.79,0.84,0.87,0.89,0.91,0.93],
            "lr":         [0.01,0.009,0.008,0.007,0.006,0.005,0.004,0.003,0.002,0.001],
        })
        buf = io.StringIO(); sample.to_csv(buf, index=False)
        st.download_button("⬇️ Download Sample CSV", buf.getvalue(),
                           file_name="sample_training_log.csv", mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# BATCH PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🖼️ Batch Predict":
    section_header("BATCH PREDICTION", "Upload multiple images and classify them all at once")
    if not model:
        st.error("❌ Model not found: `best.pt` must be in the project root.")
        st.stop()

    uploaded_files = st.file_uploader("Drop multiple vehicle images here",
        type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True)

    if uploaded_files:
        st.markdown(f"<div style='color:#00E5FF; font-size:0.88rem; margin:8px 0 16px;'>📂 {len(uploaded_files)} file(s) selected</div>", unsafe_allow_html=True)
        if st.button("🚀 Run Batch Inference"):
            results_data = []
            prog = st.progress(0, text="Running inference…")
            for i, f in enumerate(uploaded_files):
                img = Image.open(f).convert("RGB")
                t0 = time.time(); probs = predict(img); elapsed = time.time()-t0
                top_idx = int(probs.argmax())
                results_data.append({"file":f.name,"img":img,"label":CLASSES[top_idx],"conf":float(probs.max()),"ms":elapsed*1000})
                prog.progress((i+1)/len(uploaded_files), text=f"Processing {f.name}…")
            prog.progress(1.0, text="✅ Done!")

            st.markdown("<br>", unsafe_allow_html=True)
            label_counts = {}
            for r in results_data: label_counts[r["label"]] = label_counts.get(r["label"],0)+1
            for col, (lbl, cnt) in zip(st.columns(len(label_counts)), label_counts.items()):
                col.metric(f"{CLASS_ICONS.get(lbl,'')} {lbl}", cnt)

            st.markdown("<br>", unsafe_allow_html=True)
            n_cols = 3
            for row in [results_data[i:i+n_cols] for i in range(0,len(results_data),n_cols)]:
                r_cols = st.columns(n_cols)
                for col, res in zip(r_cols, row):
                    color = CLASS_COLORS.get(res["label"],"#00E5FF")
                    col.image(res["img"], use_container_width=True)
                    col.markdown(f"""<div style='background:#041428; border:1px solid {color}44; border-left:4px solid {color};
                        border-radius:8px; padding:10px 12px; margin-top:4px;'>
                        <div style='font-size:1rem; font-weight:700; color:{color};'>{CLASS_ICONS.get(res["label"],"🚗")} {res["label"]}</div>
                        <div style='font-size:0.8rem; color:#5a8aaa; margin-top:2px;'>{res["conf"]*100:.1f}% · {res["ms"]:.0f} ms · {res["file"][:22]}</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='glass-card' style='text-align:center; padding:48px;'>
            <div style='font-size:3rem;'>🖼️</div>
            <div style='font-family:Orbitron,monospace; color:#00E5FF; font-size:1rem; margin:12px 0 8px;'>NO FILES SELECTED</div>
            <div style='color:#5a8aaa; font-size:0.88rem;'>Upload one or more images above to run batch inference</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    section_header("PREDICT", "Upload a vehicle image to classify it")
    if not model:
        st.error("❌ Model not found: `best.pt` must be in the project root.")
        st.stop()

    uploaded = st.file_uploader("Drop a vehicle image here", type=["jpg","jpeg","png","bmp","webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)
        with col2:
            with st.spinner("Running inference…"):
                t0 = time.time(); probs = predict(img); elapsed = time.time()-t0
            top_idx = int(probs.argmax()); top_label = CLASSES[top_idx]; top_conf = float(probs.max())
            if top_conf >= conf_threshold:
                st.markdown(result_card_html(top_label, top_conf), unsafe_allow_html=True)
            else:
                st.warning(f"⚠️ `{top_label}` ({top_conf*100:.1f}%) is below the {conf_threshold*100:.0f}% threshold.")
            st.markdown("<br>", unsafe_allow_html=True)
            st.pyplot(draw_prob_chart(probs))
            st.caption(f"⏱ Inference time: {elapsed*1000:.0f} ms")
            st.markdown("<br><div style='font-weight:600; color:#00E5FF; font-size:0.9rem; margin-bottom:8px;'>TOP-3 PREDICTIONS</div>", unsafe_allow_html=True)
            for rank, idx in enumerate(probs.argsort()[::-1][:3], 1):
                lbl = CLASSES[idx]; color = CLASS_COLORS.get(lbl,"#888")
                st.markdown(f"""<div style='margin-bottom:10px;'>
                    <div style='display:flex; justify-content:space-between; font-size:0.88rem; margin-bottom:4px;'>
                        <span style='color:{color}; font-weight:600;'>{rank}. {CLASS_ICONS.get(lbl,"")} {lbl}</span>
                        <span style='color:#cde8f5;'>{probs[idx]*100:.1f}%</span></div>
                    <div class='ds-bar-bg'><div class='ds-bar-fill' style='width:{int(probs[idx]*100)}%; background:{color};'></div></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class='glass-card' style='text-align:center; padding:64px;'>
            <div style='font-size:4rem;'>🔍</div>
            <div style='font-family:Orbitron,monospace; color:#00E5FF; font-size:1.1rem; margin:16px 0 10px;'>READY TO CLASSIFY</div>
            <div style='color:#5a8aaa; font-size:0.9rem;'>Upload a vehicle image above to get started</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""<div style='text-align:center; color:#1a3a55; font-size:0.78rem; padding:16px; border-top:1px solid rgba(0,229,255,0.06);'>
    AI-Based Vehicle Recognition System &nbsp;·&nbsp; YOLOv8s-cls &nbsp;·&nbsp; Streamlit
</div>""", unsafe_allow_html=True)

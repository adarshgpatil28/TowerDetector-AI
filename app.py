"""
TowerVision - AI Tower Component Detection and Visualization Dashboard
======================================================================
Professional engineering dashboard for aerial & ground tower structural inspection.

Pipeline:
  01. Image Upload
  02. Image Quality Assessment (Laplacian Blur & Exposure Analysis)
  03. YOLO Component Detection (monopole_tower & supporting_tower)
  04. Analytics, Confidence Computation & Annotated Output Display
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import streamlit as st

# Reuse existing Phase 2 pipeline functions and constants
from phase2_tower_detection import (
    load_yolo_model,
    check_image_quality,
    calculate_average_confidence,
    DEFAULT_WEIGHTS_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_BLUR_THRESHOLD,
    DEFAULT_DARK_THRESHOLD,
    DEFAULT_BRIGHT_THRESHOLD,
    DEFAULT_CLASSES
)

# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="TowerVision | AI Tower Component Detection",
    page_icon="🗼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM ENGINEERING/TELEMETRY DARK THEME CSS
# ==========================================
st.markdown("""
<style>
    /* Global Background & Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #07111F !important;
        color: #F8FAFC !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: #07111F !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    [data-testid="stSidebar"] {
        background-color: #0B1424 !important;
        border-right: 1px solid #1E2E48 !important;
    }

    /* Hide default Streamlit decoration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Top Navigation Bar */
    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 20px;
        background: #0B1626;
        border: 1px solid #1E2E48;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-logo {
        width: 38px;
        height: 38px;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(34, 211, 238, 0.15) 0%, rgba(59, 130, 246, 0.25) 100%);
        border: 1px solid #22D3EE;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #22D3EE;
    }
    .brand-name {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.1;
    }
    .brand-sub {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        color: #94A3B8;
        text-transform: uppercase;
        margin: 0;
        font-weight: 500;
    }
    .system-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.35);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: #4ADE80;
        font-family: 'JetBrains Mono', monospace;
    }
    .status-pulse {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #22C55E;
        box-shadow: 0 0 8px #22C55E;
    }

    /* Process Timeline */
    .timeline-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #0B1424;
        border: 1px solid #1E2E48;
        border-radius: 8px;
        padding: 10px 18px;
        margin-bottom: 24px;
    }
    .timeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        color: #64748B;
        text-transform: uppercase;
    }
    .timeline-step.active {
        color: #22D3EE;
    }
    .timeline-step.completed {
        color: #38BDF8;
    }
    .timeline-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 2px 6px;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.05);
    }
    .timeline-step.active .timeline-num {
        background: rgba(34, 211, 238, 0.2);
        color: #22D3EE;
        border: 1px solid rgba(34, 211, 238, 0.4);
    }
    .timeline-arrow {
        color: #334155;
        font-size: 0.8rem;
    }

    /* Section Cards */
    .telemetry-card {
        background: #0E1B2E;
        border: 1px solid #1E2E48;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    .card-header-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #22D3EE;
        text-transform: uppercase;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 12px;
    }
    .card-subtitle {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 16px;
    }

    /* Diagnostic Quality Card */
    .diagnostic-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-top: 10px;
    }
    .diagnostic-item {
        background: #091322;
        border: 1px solid #1A2942;
        border-radius: 6px;
        padding: 10px 14px;
    }
    .diag-label {
        font-size: 0.7rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }
    .diag-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 600;
        color: #F8FAFC;
    }
    .diag-status-ok {
        color: #22C55E;
        font-weight: 600;
    }
    .diag-status-bad {
        color: #EF4444;
        font-weight: 600;
    }

    /* Detection Result Cards */
    .detection-card {
        background: #0E1B2E;
        border: 1px solid #1E2E48;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .det-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94A3B8;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .det-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.45rem;
        font-weight: 700;
        color: #22D3EE;
    }
    .det-count-badge {
        font-size: 0.72rem;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(34, 211, 238, 0.12);
        color: #67E8F9;
        font-weight: 600;
        float: right;
    }

    /* Custom Streamlit Primary Button & Download Button Override */
    div.stButton > button[kind="primary"], div.stDownloadButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #22D3EE !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 16px rgba(34, 211, 238, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
        border-color: #67E8F9 !important;
        box-shadow: 0 0 24px rgba(34, 211, 238, 0.45) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
    }

    /* File Uploader Dark Theme */
    [data-testid="stFileUploader"] section {
        background-color: #0E1B2E !important;
        border: 1px dashed #243550 !important;
        border-radius: 8px !important;
        color: #94A3B8 !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #22D3EE !important;
    }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] div {
        color: #94A3B8 !important;
    }

    /* Tabs Styling */
    [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #1E2E48 !important;
    }
    [data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 500 !important;
    }
    [aria-selected="true"] {
        color: #22D3EE !important;
        font-weight: 600 !important;
    }

    /* Empty state styling */
    .empty-state-box {
        background: #0B1626;
        border: 1px dashed #243550;
        border-radius: 10px;
        padding: 36px 20px;
        text-align: center;
        margin: 14px 0;
    }
    .empty-title {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #F8FAFC;
        margin-bottom: 6px;
    }
    .empty-desc {
        font-size: 0.82rem;
        color: #94A3B8;
        margin-bottom: 12px;
    }
    .empty-tags {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #38BDF8;
        background: rgba(34, 211, 238, 0.08);
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        border: 1px solid rgba(34, 211, 238, 0.2);
    }

    /* Notification Box */
    .completion-banner {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.35);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .completion-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #4ADE80;
        margin: 0;
    }
    .completion-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        margin: 0;
    }

    .rejection-banner {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
    }
    .rejection-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #F87171;
        margin-bottom: 4px;
    }
    .rejection-sub {
        font-size: 0.8rem;
        color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# CACHED MODEL LOADER
# ==========================================
@st.cache_resource(show_spinner=False)
def get_cached_model(weights_path: str):
    """Loads and caches the trained YOLO model in memory."""
    return load_yolo_model(Path(weights_path))


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 12px; border-bottom: 1px solid #1E2E48; margin-bottom: 18px;">
        <span style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; color: #22D3EE; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">
            CONTROL PANEL
        </span>
        <div style="font-size: 1.05rem; font-weight: 600; color: #F8FAFC;">Inspection Config</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Object Detection Threshold**")
    conf_threshold = st.slider(
        "Confidence Gate (conf)",
        min_value=0.10,
        max_value=1.00,
        value=0.60,
        step=0.05,
        help="Initial default 0.60 eliminates low-confidence duplicate bounding boxes."
    )

    st.markdown("---")
    st.markdown("**Image Quality Gates**")

    blur_thresh = st.number_input(
        "Sharpness Threshold (Laplacian Var)",
        min_value=10.0,
        max_value=2500.0,
        value=float(DEFAULT_BLUR_THRESHOLD),
        step=25.0,
        help="Images below this variance of Laplacian are rejected as blurry."
    )

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        dark_thresh = st.number_input(
            "Min Brightness",
            min_value=0.0,
            max_value=120.0,
            value=float(DEFAULT_DARK_THRESHOLD),
            step=5.0,
            help="Grayscale mean below this is rejected as underexposed."
        )
    with col_sb2:
        bright_thresh = st.number_input(
            "Max Brightness",
            min_value=130.0,
            max_value=255.0,
            value=float(DEFAULT_BRIGHT_THRESHOLD),
            step=5.0,
            help="Grayscale mean above this is rejected as overexposed."
        )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace; line-height: 1.6;">
        <span style="color: #22D3EE; font-weight: 600;">ACTIVE MODEL</span><br>
        • Architecture: Ultralytics YOLO<br>
        • Weights: <code>best.pt</code><br>
        • Classes: <code>monopole_tower</code>, <code>supporting_tower</code>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TOP NAVIGATION BAR
# ==========================================
st.markdown("""
<div class="top-nav">
    <div class="brand-wrap">
        <div class="brand-logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="12 2 2 22 22 22 12 2"></polygon>
                <line x1="12" y1="6" x2="12" y2="22"></line>
                <line x1="7" y1="16" x2="17" y2="16"></line>
                <line x1="9" y1="11" x2="15" y2="11"></line>
            </svg>
        </div>
        <div>
            <div class="brand-name">TowerVision</div>
            <div class="brand-sub">AI Tower Component Detection & Telemetry</div>
        </div>
    </div>
    <div class="system-badge">
        <span class="status-pulse"></span>
        <span>SYSTEM READY</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==========================================
# TIMELINE / PROCESS STATUS
# ==========================================
def render_timeline(current_stage: int):
    """
    Renders the 4-step engineering timeline.
    current_stage: 1 (Upload), 2 (Quality Check), 3 (Object Detection), 4 (Result Ready)
    """
    def step_cls(num: int):
        if num < current_stage:
            return "completed"
        elif num == current_stage:
            return "active"
        return ""

    st.markdown(f"""
    <div class="timeline-container">
        <div class="timeline-step {step_cls(1)}">
            <span class="timeline-num">01</span>
            <span>Image Upload</span>
        </div>
        <span class="timeline-arrow">›</span>
        <div class="timeline-step {step_cls(2)}">
            <span class="timeline-num">02</span>
            <span>Quality Check</span>
        </div>
        <span class="timeline-arrow">›</span>
        <div class="timeline-step {step_cls(3)}">
            <span class="timeline-num">03</span>
            <span>Object Detection</span>
        </div>
        <span class="timeline-arrow">›</span>
        <div class="timeline-step {step_cls(4)}">
            <span class="timeline-num">04</span>
            <span>Result Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Initialize Session State
if "raw_image" not in st.session_state:
    st.session_state.raw_image = None
if "image_name" not in st.session_state:
    st.session_state.image_name = ""
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "stage" not in st.session_state:
    st.session_state.stage = 1


# ==========================================
# SECTION 1 — IMAGE INPUT
# ==========================================
st.markdown("""
<div class="card-header-label">SECTION 01</div>
<div class="card-title">Inspection Image</div>
<div class="card-subtitle">Upload a tower photograph for automated structural classification.</div>
""", unsafe_allow_html=True)

input_tab1, input_tab2 = st.tabs(["📁 Upload Image File", "🗃️ Select Test Dataset Sample"])

with input_tab1:
    uploaded_file = st.file_uploader(
        "Drag and drop tower photograph",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            decoded_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if decoded_img is None:
                st.error("Invalid or unreadable image file. Please provide a standard JPG or PNG.")
            else:
                st.session_state.raw_image = decoded_img
                st.session_state.image_name = uploaded_file.name
                st.session_state.stage = 1
        except Exception as e:
            st.error(f"Error reading uploaded image buffer: {e}")

with input_tab2:
    test_dir = Path("test/images")
    if test_dir.exists():
        sample_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.JPG")))
        sample_map = {f.name: f for f in sample_files}
        if sample_map:
            # Default to index 2 (supporting tower with ~0.96 conf)
            selected_name = st.selectbox(
                "Choose an image from the test set",
                options=list(sample_map.keys()),
                index=2,
                label_visibility="collapsed"
            )
            if st.button("Load Selected Sample", use_container_width=False):
                sample_path = sample_map[selected_name]
                st.session_state.raw_image = cv2.imread(str(sample_path))
                st.session_state.image_name = selected_name
                st.session_state.stage = 1
                st.session_state.pipeline_result = None

# Empty State or Active Preview
raw_image = st.session_state.raw_image
image_name = st.session_state.image_name

if raw_image is None:
    render_timeline(1)
    st.markdown("""
    <div class="empty-state-box">
        <div class="empty-title">TOWER INSPECTION</div>
        <div class="empty-desc">Upload an image or choose a dataset sample above to begin structural analysis.</div>
        <div class="empty-tags">SUPPORTED: JPG • JPEG • PNG</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display preview with technical metadata
    col_img_prev, col_meta = st.columns([1.3, 1])
    with col_img_prev:
        rgb_prev = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        st.image(rgb_prev, caption=f"Active Input: {image_name}", use_container_width=True)

    with col_meta:
        st.markdown(f"""
        <div class="telemetry-card" style="margin-top: 10px;">
            <div class="card-header-label">INPUT TELEMETRY</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; line-height: 2.1; color: #CBD5E1;">
                <div><span style="color: #94A3B8;">FILENAME:</span> {image_name}</div>
                <div><span style="color: #94A3B8;">RESOLUTION:</span> {raw_image.shape[1]} × {raw_image.shape[0]} px</div>
                <div><span style="color: #94A3B8;">CHANNELS:</span> {raw_image.shape[2]} (BGR Color)</div>
                <div><span style="color: #94A3B8;">BUFFER SIZE:</span> {raw_image.nbytes / 1024:.1f} KB</div>
                <div><span style="color: #94A3B8;">STATUS:</span> <span style="color: #38BDF8;">LOADED & READY</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # SECTION 2 — EXECUTION CONTROL
    # ==========================================
    st.markdown("---")
    col_btn, col_hint = st.columns([1, 2])
    with col_btn:
        execute_clicked = st.button("RUN INSPECTION", type="primary", use_container_width=True)
    with col_hint:
        if not execute_clicked and st.session_state.pipeline_result is None:
            st.markdown("""
            <div style="font-size: 0.85rem; color: #94A3B8; padding-top: 10px; font-family: 'JetBrains Mono', monospace;">
                ● Ready for inspection. Click <strong>RUN INSPECTION</strong> to start the pipeline.
            </div>
            """, unsafe_allow_html=True)

    # Process Execution Pipeline
    if execute_clicked:
        st.session_state.stage = 2
        with st.spinner("Analyzing image quality and running structural inference..."):
            # Step 1: Quality Check
            quality_result = check_image_quality(
                image=raw_image,
                blur_threshold=blur_thresh,
                dark_threshold=dark_thresh,
                bright_threshold=bright_thresh
            )

            # Check if rejected
            if not quality_result["passed"]:
                st.session_state.pipeline_result = {
                    "quality_check": quality_result,
                    "passed": False,
                    "detections": [],
                    "annotated_image": None,
                    "conf_summary": None,
                    "output_path": None
                }
                st.session_state.stage = 2
            else:
                st.session_state.stage = 3
                # Step 2: YOLO Detection
                model = get_cached_model(str(DEFAULT_WEIGHTS_PATH))
                results = model.predict(source=raw_image, conf=conf_threshold, verbose=False)
                first_res = results[0]

                detections = []
                for box in first_res.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    conf = float(box.conf[0].item())
                    xyxy = [round(float(c), 2) for c in box.xyxy[0].tolist()]

                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf,
                        "bbox": xyxy
                    })

                annotated_bgr = first_res.plot()

                # Save output image
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                out_path = DEFAULT_OUTPUT_DIR / f"annotated_{Path(image_name).name}"
                cv2.imwrite(str(out_path), annotated_bgr)

                # Step 3: Confidence Metrics
                conf_summary = calculate_average_confidence(detections, target_classes=DEFAULT_CLASSES)

                st.session_state.pipeline_result = {
                    "quality_check": quality_result,
                    "passed": True,
                    "detections": detections,
                    "annotated_bgr": annotated_bgr,
                    "conf_summary": conf_summary,
                    "output_path": out_path
                }
                st.session_state.stage = 4

    # ==========================================
    # DISPLAY RESULTS WHEN AVAILABLE
    # ==========================================
    res = st.session_state.pipeline_result
    if res is not None:
        render_timeline(st.session_state.stage)

        # ------------------------------------------
        # SECTION 3 — IMAGE QUALITY DIAGNOSTIC CARD
        # ------------------------------------------
        qc = res["quality_check"]
        is_accepted = qc["passed"]

        st.markdown("""
        <div class="card-header-label">SECTION 02</div>
        <div class="card-title">Image Quality Assessment</div>
        """, unsafe_allow_html=True)

        if is_accepted:
            status_dot_html = '<span style="color: #22C55E;">● ACCEPTED</span>'
            exposure_desc = "Normal"
            exposure_cls = "diag-status-ok"
        else:
            status_dot_html = '<span style="color: #EF4444;">● REJECTED</span>'
            if qc["exposure_details"]["is_underexposed"]:
                exposure_desc = "Underexposed"
            elif qc["exposure_details"]["is_overexposed"]:
                exposure_desc = "Overexposed"
            else:
                exposure_desc = "Normal"
            exposure_cls = "diag-status-bad" if exposure_desc != "Normal" else "diag-status-ok"

        sharpness_cls = "diag-status-ok" if not qc["blur_details"]["is_blurry"] else "diag-status-bad"

        st.markdown(f"""
        <div class="telemetry-card">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E2E48; padding-bottom: 10px; margin-bottom: 12px;">
                <span style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">IMAGE QUALITY</span>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 700;">{status_dot_html}</span>
            </div>
            <div class="diagnostic-grid">
                <div class="diagnostic-item">
                    <div class="diag-label">Sharpness (Variance of Laplacian)</div>
                    <div class="diag-val {sharpness_cls}">{qc['blur_score']:.1f}</div>
                    <div style="font-size: 0.68rem; color: #64748B;">Threshold: ≥ {qc['blur_threshold']:.1f}</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diag-label">Brightness (Grayscale Mean)</div>
                    <div class="diag-val">{qc['brightness']:.1f}</div>
                    <div style="font-size: 0.68rem; color: #64748B;">Acceptable: [{qc['dark_threshold']:.0f} - {qc['bright_threshold']:.0f}]</div>
                </div>
                <div class="diagnostic-item">
                    <div class="diag-label">Exposure Status</div>
                    <div class="diag-val {exposure_cls}">{exposure_desc}</div>
                    <div style="font-size: 0.68rem; color: #64748B;">Lighting Evaluation</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Rejection Gate
        if not is_accepted:
            st.markdown(f"""
            <div class="rejection-banner">
                <div class="rejection-title">⛔ Inspection Halted: Quality Validation Failed</div>
                <div class="rejection-sub">
                    <strong>Reason:</strong> {qc['reason']}<br>
                    <em>In accordance with safety specifications, images that fail quality validation are not submitted to the neural network.</em>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ------------------------------------------
            # SECTION 7 — COMPLETION BANNER
            # ------------------------------------------
            st.markdown("""
            <div class="completion-banner">
                <div style="color: #22C55E; font-size: 1.4rem;">✔</div>
                <div>
                    <div class="completion-title">Inspection Complete</div>
                    <div class="completion-sub">Tower structural component analysis completed successfully.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ------------------------------------------
            # SECTION 4 & 5 — DETECTION RESULTS & ANALYTICS
            # ------------------------------------------
            st.markdown("""
            <div class="card-header-label">SECTION 03</div>
            <div class="card-title">Detection Results & Visual Analytics</div>
            """, unsafe_allow_html=True)

            col_left, col_right = st.columns([1.4, 1])

            with col_left:
                annotated_rgb = cv2.cvtColor(res["annotated_bgr"], cv2.COLOR_BGR2RGB)
                st.image(
                    annotated_rgb,
                    caption=f"Annotated Output: {image_name} (Confidence Gate: {conf_threshold})",
                    use_container_width=True
                )

            with col_right:
                conf_sum = res["conf_summary"]
                mono_avg = conf_sum["class_averages"].get("monopole_tower", "No detections")
                supp_avg = conf_sum["class_averages"].get("supporting_tower", "No detections")
                mono_cnt = conf_sum["class_counts"].get("monopole_tower", 0)
                supp_cnt = conf_sum["class_counts"].get("supporting_tower", 0)

                disp_mono = f"{mono_avg*100:.1f}%" if isinstance(mono_avg, float) else mono_avg
                disp_supp = f"{supp_avg*100:.1f}%" if isinstance(supp_avg, float) else supp_avg

                # Monopole Tower Card
                st.markdown(f"""
                <div class="detection-card">
                    <span class="det-count-badge">{mono_cnt} DETECTED</span>
                    <div class="det-title">MONOPOLE TOWER</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Confidence</div>
                    <div class="det-conf-val">{disp_mono}</div>
                </div>
                """, unsafe_allow_html=True)

                # Supporting Tower Card
                st.markdown(f"""
                <div class="detection-card">
                    <span class="det-count-badge">{supp_cnt} DETECTED</span>
                    <div class="det-title">SUPPORTING TOWER</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Confidence</div>
                    <div class="det-conf-val">{disp_supp}</div>
                </div>
                """, unsafe_allow_html=True)

                # Total Detections & Overall Avg
                ovr = conf_sum["overall"]
                disp_ovr = f"{ovr*100:.1f}%" if isinstance(ovr, float) else ovr

                st.markdown(f"""
                <div class="detection-card" style="border-color: rgba(34, 211, 238, 0.3);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div class="det-title">TOTAL DETECTIONS</div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #F8FAFC;">
                                {conf_sum['total_detections']}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div class="det-title">OVERALL CONFIDENCE</div>
                            <div class="det-conf-val" style="font-size: 1.3rem;">{disp_ovr}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Download Button
                is_success, buffer = cv2.imencode(".jpg", res["annotated_bgr"])
                if is_success:
                    st.download_button(
                        label="📥 Download Annotated Image",
                        data=buffer.tobytes(),
                        file_name=f"annotated_{Path(image_name).name}",
                        mime="image/jpeg",
                        type="primary",
                        use_container_width=True
                    )

            # Detailed Bounding Boxes Table
            if res["detections"]:
                st.markdown("""
                <div style="margin-top: 14px; margin-bottom: 6px; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; color: #94A3B8; text-transform: uppercase; font-family: 'JetBrains Mono', monospace;">
                    BOUNDING BOX TELEMETRY
                </div>
                """, unsafe_allow_html=True)

                table_data = []
                for idx, det in enumerate(res["detections"], 1):
                    table_data.append({
                        "#": idx,
                        "Component Class": det["class_name"],
                        "Confidence": f"{det['confidence']*100:.2f}%",
                        "Box Coordinates [x1, y1, x2, y2]": f"[{det['bbox'][0]}, {det['bbox'][1]}, {det['bbox'][2]}, {det['bbox'][3]}]"
                    })
                st.dataframe(table_data, use_container_width=True)

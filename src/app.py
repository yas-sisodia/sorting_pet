import time
import cv2
import numpy as np
import streamlit as st

from engine import SortingInferenceEngine
from stream import ThreadedCamera

# Page configuration for an industrial SCADA look
st.set_page_config(
    page_title="Abnoba Systems | Optical Sorter Telemetry",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for high-contrast industrial control displays
st.markdown("""
<style>
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .status-active {
        color: #3fb950;
        font-weight: 700;
    }
    .status-standby {
        color: #f85149;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_hardware():
    """Initializes camera thread and CoreML/ONNX session once across re-runs."""
    camera = ThreadedCamera(src=0, target_fps=30).start()
    engine = SortingInferenceEngine()
    return camera, engine


camera, engine = load_hardware()

# --- Sidebar Controls ---
st.sidebar.image("https://abnoba-systems.com/wp-content/uploads/2023/10/cropped-Abnoba_Logo_RGB_transparent_ohne_Subline-1.png", width=180)
st.sidebar.markdown("### ⚙️ Line Calibration")
conveyor_speed = st.sidebar.slider("Conveyor Speed (m/s)", 0.5, 3.0, 1.5, 0.1)
camera_distance = st.sidebar.slider("Sensor-to-Valve (m)", 0.2, 1.2, 0.6, 0.05)
conf_threshold = st.sidebar.slider("Rejection Threshold", 0.30, 0.95, 0.40, 0.05)
st.sidebar.divider()
run_stream = st.sidebar.toggle("Start Optical Stream", value=True)

# Update engine calibration on the fly
engine.CONVEYOR_SPEED_MPS = conveyor_speed
engine.CAMERA_TO_ACTUATOR_M = camera_distance

# --- Main Dashboard Header ---
st.title("♻️ Optical Material Sorter — Edge Diagnostics")
st.caption("Target System: High-Speed Polymer Classification & Actuator Sequencing")

col_vid, col_telemetry = st.columns([1.6, 1.0])

with col_vid:
    video_placeholder = st.empty()

with col_telemetry:
    st.markdown("#### ⚡ Actuator Telemetry")
    m1, m2 = st.columns(2)
    metric_latency = m1.empty()
    metric_fps = m2.empty()

    m3, m4 = st.columns(2)
    metric_class = m3.empty()
    metric_delay = m4.empty()

    st.markdown("#### 📡 Real-time PLC Output")
    plc_code_box = st.empty()

# Persistent state for actuator logs
if "log_history" not in st.session_state:
    st.session_state.log_history = []

# --- Continuous Inference & Display Loop ---
while run_stream:
    frame_id, frame = camera.read_latest()
    if frame is None:
        time.sleep(0.01)
        continue

    # Run ONNX CoreML inference
    result, latency = engine.infer(frame)
    conf = result["confidence"]
    label = result["material_class"]
    action = result["action"]
    delay_ms = result["actuator_trigger_delay_ms"]

    # Overlay bounding box & target tags
    display_frame = frame.copy()
    if conf >= conf_threshold and result["bbox_xyxy"] != [0, 0, 0, 0]:
        x1, y1, x2, y2 = result["bbox_xyxy"]
        box_color = (0, 255, 0) if "AIR_BLAST" in action else (0, 0, 255)

        # Draw target bounds
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)

        # Tag banner
        tag = f"{label} | Conf: {conf * 100:.1f}%"
        cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + 280, y1), box_color, -1)
        cv2.putText(display_frame, tag, (x1 + 6, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Convert BGR to RGB for Streamlit rendering
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    # Update metric widgets
    metric_latency.metric("Inference Latency", f"{latency:.2f} ms")
    metric_fps.metric("Throughput", f"{1000.0 / max(latency, 0.1):.1f} FPS")
    metric_class.metric("Detected Polymer", label)
    metric_delay.metric("Pneumatic Delay", f"{delay_ms:.1f} ms")

    # Render formatted PLC output payload
    plc_payload = {
        "frame_id": frame_id,
        "class": label,
        "confidence": conf,
        "actuator_action": action,
        "valve_trigger_delay_ms": delay_ms,
        "edge_accelerator": "Apple Silicon (CoreML/NE)"
    }
    plc_code_box.json(plc_payload)

    # Gentle throttle to match sensor cycle
    time.sleep(0.02)
import os
import io
import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# =============================
# App & Theme Setup
# =============================
st.set_page_config(
    page_title="Face Mask Detection — Pro",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------- CSS Injection (Glassmorphism + Modern UI) ---------
CUSTOM_CSS = r"""
<style>
/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&family=Inter:wght@400;600;800&display=swap');

:root {
  --bg: #0f1220;
  --card: rgba(255,255,255,0.06);
  --glass: rgba(255,255,255,0.08);
  --border: rgba(255,255,255,0.18);
  --text: #E6E8F2;
  --muted: #A5ACBA;
  --primary: #7C3AED; /* Violet */
  --primary-2: #22D3EE; /* Cyan */
  --success: #22C55E;
  --danger: #EF4444;
}

/* Global */
html, body, [class^="css"], [class*="css"]  { font-family: 'Inter','Cairo',system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; }

/* Main background gradient */
.stApp {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(124,58,237,0.25) 0%, rgba(15,18,32,0) 50%),
              radial-gradient(1000px 500px at 100% 10%, rgba(34,211,238,0.18) 0%, rgba(15,18,32,0) 40%),
              linear-gradient(180deg, #0b0f1c 0%, #0f1220 100%);
  color: var(--text);
}

/* Header/Hero */
.hero {
  position: relative;
  padding: 28px 28px;
  border-radius: 24px;
  background: linear-gradient(135deg, rgba(124,58,237,0.25), rgba(34,211,238,0.18));
  border: 1px solid var(--border);
  box-shadow: 0 10px 40px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
  margin-bottom: 40px; 
}
.hero .title { font-size: 36px; font-weight: 800; letter-spacing: 0.3px; }
.hero .subtitle { font-size: 15px; color: var(--muted); margin-top: 6px; }
.badge {
  display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
  border: 1px solid var(--border); background: var(--glass); font-size: 12px; color: var(--text);
}

/* Cards */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.card h4 { margin: 0 0 10px 0; font-weight: 700; }
.card .muted { color: var(--muted); font-size: 13px; }

/* Buttons */
.btn-primary {
  display: inline-flex; align-items: center; gap: 10px; padding: 10px 16px; border-radius: 14px; border: 1px solid var(--border);
  background: linear-gradient(135deg, rgba(124,58,237,0.65), rgba(34,211,238,0.45));
  color: white; font-weight: 700; text-decoration: none; transition: transform 120ms ease, filter 120ms ease;
}
.btn-primary:hover { transform: translateY(-1px); filter: brightness(1.05); }

/* Pills */
.pill { display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px; border:1px solid var(--border); background: var(--glass); font-size:12px; }

/* Hide default Streamlit header/footer */
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Small tweaks */
.block-container { padding-top: 1.2rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================
# Paths & Config
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "Face_Mask_Model.h5")
IMG_SIZE = (224, 224)
CLASS_NAMES = ["with_mask", "without_mask"]

# Haar Cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# =============================
# Model Loading (cached)
# =============================
@st.cache_resource(show_spinner=True)
def load_mask_model():
    return load_model(MODEL_PATH)

model = load_mask_model()

# =============================
# Inference helpers
# =============================
def predict_face(face_img, model, threshold=0.5):
    img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    x = img_to_array(img)[None, ...].astype(np.float32)
    x = preprocess_input(x)
    prob = model.predict(x, verbose=0).ravel()[0]
    if prob >= threshold:
        label = "without_mask"; conf = prob
    else:
        label = "with_mask"; conf = 1 - prob
    return label, float(conf)

def process_frame(img, model, threshold=0.5, box_thickness=2, draw_confidence=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        label, conf = predict_face(face_roi, model, threshold=threshold)
        color = (34, 197, 94) if label == "with_mask" else (239, 68, 68)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, box_thickness)
        if draw_confidence:
            text = f"{label} · {conf*100:.1f}%"
            cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return img, len(faces)

# =============================
# Hero / Header
# =============================
with st.container():
    st.markdown(
        """
        <div class="hero">
          <div class="badge">😷 Real-time CV • MobileNetV2</div>
          <div class="title">Face Mask Detection — <span style="background: linear-gradient(90deg,#a78bfa,#22d3ee); -webkit-background-clip: text; background-clip: text; color: transparent;">Pro Edition</span></div>
          <div class="subtitle">Upload an image, take a snapshot, or run real-time webcam detection. Modern UI + Glassmorphism + Advanced controls.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================
# Sidebar Controls (Transparent + Blur)
# =============================
with st.sidebar:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            background: rgba(15, 18, 32, 0.55) !important;
            backdrop-filter: blur(18px) !important;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Threshold (sensitivity for 'no mask')", 0.10, 0.90, 0.50, 0.01)
    box_thickness = st.slider("Box Thickness", 1, 6, 2)
    draw_conf = st.checkbox("Show Confidence", value=True)

    st.divider()

    # ✅ شرح مكان الـ Info
    st.markdown(
        """
        <div style="padding:12px; font-size:14px; line-height:1.6; color:#E6E8F2; font-weight:500;">
        📖 <b>About Project</b><br>
        This Face Mask Detection app is built with &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <span style="color:#a78bfa;">Deep Learning</span> and 
        <span style="color:#22d3ee;">Computer Vision</span>.  
        It allows you to <b>upload images</b>, <b>take snapshots</b>, or run <b>real-time webcam detection</b>.  
        The goal is to demonstrate how AI can be used in <i>public safety</i> with a modern UI.
        </div>
        """,
        unsafe_allow_html=True
    )

# KPI Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='card'><h4>🚀 Status</h4><div class='muted'>Model Loaded</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown(f"<div class='card'><h4>🧩 Cascade</h4><div class='muted'>{os.path.basename(CASCADE_PATH)}</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='card'><h4>🖼️ Input Size</h4><div class='muted'>{IMG_SIZE[0]}×{IMG_SIZE[1]}</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='card'><h4>🎛️ Threshold</h4><div class='muted'>{int(threshold*100)}%</div></div>", unsafe_allow_html=True)

st.markdown("\n")

# =============================
# Tabs for modes
# =============================
upload_tab, camera_tab, live_tab = st.tabs(["📁 Upload Image", "📸 Camera Snapshot", "🟢 Live Webcam (WebRTC)"])

# -------- Upload Image --------
with upload_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    file = st.file_uploader("Upload a face image (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=False)
    if file is not None:
        img = np.array(Image.open(file).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_img, n_faces = process_frame(img_bgr.copy(), model, threshold=threshold, box_thickness=box_thickness, draw_confidence=draw_conf)
        rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption=f"Result — {n_faces} face(s) detected", use_container_width=True)

        # Download button
        buffered = io.BytesIO()
        Image.fromarray(rgb).save(buffered, format="PNG")
        st.download_button(
            "Download Result",
            data=buffered.getvalue(),
            file_name="mask_detection_result.png",
            mime="image/png",
        )
    else:
        st.markdown("<span class='muted'>Upload an image to start analysis…</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------- Camera Snapshot --------
with camera_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    cam_img = st.camera_input("Take a snapshot")
    if cam_img is not None:
        img = np.array(Image.open(cam_img).convert("RGB"))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_img, n_faces = process_frame(img_bgr.copy(), model, threshold=threshold, box_thickness=box_thickness, draw_confidence=draw_conf)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Result — {n_faces} face(s) detected", use_container_width=True)
    else:
        st.markdown("<span class='muted'>Capture a snapshot to start analysis…</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------- Live Webcam (WebRTC) --------
with live_tab:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.threshold = threshold
            self.box_thickness = box_thickness
            self.draw_confidence = draw_conf
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result_img, _ = process_frame(
                img, self.model,
                threshold=self.threshold,
                box_thickness=self.box_thickness,
                draw_confidence=self.draw_confidence,
            )
            return av.VideoFrame.from_ndarray(result_img, format="bgr24")

    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    webrtc_ctx = webrtc_streamer(
        key="mask-detect-pro",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

    # ✅ CSS للتحكم في حجم شاشة الكاميرا
    st.markdown(
        """
        <style>
        video {
            width: 100% !important;
            border-radius: 16px;
            margin: auto;
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="display:flex; gap:10px; align-items:center; margin-top:8px; font-size:13px;">
          <div class="pill">🟢 Real-time Stream</div>
          <div class="pill">🎛️ Control via Sidebar</div>
          <div class="pill">💡 Green = With Mask</div>
          <div class="pill">🔴 Red = Without Mask</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
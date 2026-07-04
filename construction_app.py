import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import yt_dlp
from pathlib import Path
import base64
from PIL import Image

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Construction Site Safety Detection",
    page_icon="🏗️",
    layout="wide"
)

# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# ─────────────────────────────────────────
# PPE Classes & Colors
# ─────────────────────────────────────────
CLASSES = {
    0: 'Helmet',
    1: 'No Helmet',
    2: 'Safety Vest',
    3: 'No Safety Vest',
    4: 'Gloves',
    5: 'No Gloves',
    6: 'Safety Boots',
    7: 'Person'
}

VIOLATION_CLASSES = ['No Helmet', 'No Safety Vest', 'No Gloves']

COLORS = {
    'Helmet':         (0, 255, 0),
    'No Helmet':      (0, 0, 255),
    'Safety Vest':    (0, 255, 0),
    'No Safety Vest': (0, 0, 255),
    'Gloves':         (0, 255, 0),
    'No Gloves':      (0, 0, 255),
    'Safety Boots':   (0, 255, 0),
    'Person':         (255, 165, 0)
}

# ─────────────────────────────────────────
# Detection Function
# ─────────────────────────────────────────
def detect_frame(frame, conf_threshold=0.5):
    results = model(frame, conf=conf_threshold)
    violations = []
    annotated = frame.copy()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = CLASSES.get(cls_id, 'Unknown')
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = COLORS.get(label, (255, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f'{label} {conf:.2f}',
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if label in VIOLATION_CLASSES:
                violations.append(label)
                cv2.putText(annotated, '⚠ VIOLATION',
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return annotated, violations


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/hard-hat.png", width=80)
st.sidebar.title("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 PPE Classes Detected")
for cls in CLASSES.values():
    color = "🟢" if cls not in VIOLATION_CLASSES else "🔴"
    st.sidebar.markdown(f"{color} {cls}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 👩‍💻 About")
st.sidebar.markdown("""
**Construction Site Safety Surveillance**
- Built with YOLOv8
- Detects PPE violations in real time
- Supports image, video, webcam, YouTube
""")

# ─────────────────────────────────────────
# Main Page
# ─────────────────────────────────────────
st.title("🏗️ Construction Site Safety Surveillance System")
st.markdown("**Detect PPE violations in real-time using YOLOv8**")
st.markdown("---")

# Detection Mode Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Image Detection",
    "🎬 Video Detection",
    "📡 Webcam Detection",
    "▶️ YouTube Detection"
])

# ─────────────────────────────────────────
# TAB 1 — Image Detection
# ─────────────────────────────────────────
with tab1:
    st.subheader("📷 Upload Image for PPE Detection")
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"], key="img_upload"
    )

    if uploaded_image:
        col1, col2 = st.columns(2)

        # Original image
        image = Image.open(uploaded_image)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with col1:
            st.markdown("**Original Image**")
            st.image(image, use_column_width=True)

        # Detected image
        with st.spinner("Running detection..."):
            annotated, violations = detect_frame(frame, conf_threshold)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown("**Detection Result**")
            st.image(annotated_rgb, use_column_width=True)

        # Results
        st.markdown("---")
        if violations:
            st.error(f"⚠️ **{len(violations)} VIOLATION(S) DETECTED!**")
            for v in set(violations):
                st.warning(f"🔴 {v}")
        else:
            st.success("✅ No violations detected! All PPE requirements met.")

# ─────────────────────────────────────────
# TAB 2 — Video Detection
# ─────────────────────────────────────────
with tab2:
    st.subheader("🎬 Upload Video for PPE Detection")
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload"
    )

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_video.read())
            tmp_path = tmp.name

        st.video(uploaded_video)

        if st.button("🚀 Run Detection on Video"):
            cap = cv2.VideoCapture(tmp_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            result_path = os.path.join(tempfile.gettempdir(), 'result_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

            all_violations = []
            progress_bar = st.progress(0)
            frame_placeholder = st.empty()
            status_text = st.empty()

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                annotated, violations = detect_frame(frame, conf_threshold)
                out.write(annotated)
                all_violations.extend(violations)

                # Show every 10th frame
                if frame_count % 10 == 0:
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(annotated_rgb, use_column_width=True)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}...")

                progress_bar.progress(min(frame_count / total_frames, 1.0))
                frame_count += 1

            cap.release()
            out.release()

            st.markdown("---")
            st.success("✅ Video processing complete!")

            if all_violations:
                st.error(f"⚠️ **{len(all_violations)} total violation(s) detected!**")
                for v in set(all_violations):
                    count = all_violations.count(v)
                    st.warning(f"🔴 {v}: detected in {count} frames")
            else:
                st.success("✅ No violations detected in the video!")

            os.unlink(tmp_path)

# ─────────────────────────────────────────
# TAB 3 — Webcam Detection
# ─────────────────────────────────────────
with tab3:
    st.subheader("📡 Real-Time Webcam Detection")
    st.info("💡 Click **Start** to begin real-time PPE detection from your webcam.")

    run_webcam = st.checkbox("▶️ Start Webcam Detection")
    frame_window = st.image([])
    violation_display = st.empty()

    if run_webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not access webcam. Please check your camera.")
        else:
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame.")
                    break

                annotated, violations = detect_frame(frame, conf_threshold)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_window.image(annotated_rgb, use_column_width=True)

                if violations:
                    violation_display.error(
                        f"⚠️ VIOLATIONS: {', '.join(set(violations))}"
                    )
                else:
                    violation_display.success("✅ No violations detected")

            cap.release()

# ─────────────────────────────────────────
# TAB 4 — YouTube Detection
# ─────────────────────────────────────────
with tab4:
    st.subheader("▶️ YouTube Video PPE Detection")
    st.info("💡 Paste a YouTube URL to detect PPE violations in the video.")

    youtube_url = st.text_input("🔗 Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("🚀 Download & Detect") and youtube_url:
        with st.spinner("Downloading YouTube video..."):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    ydl_opts = {
                        'format': 'best[height<=480]',
                        'outtmpl': os.path.join(tmpdir, 'video.%(ext)s'),
                        'quiet': True
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([youtube_url])

                    video_files = list(Path(tmpdir).glob('video.*'))
                    if not video_files:
                        st.error("❌ Could not download video. Check the URL.")
                    else:
                        video_path = str(video_files[0])
                        cap = cv2.VideoCapture(video_path)
                        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        result_path = os.path.join(tempfile.gettempdir(), 'yt_result.mp4')
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

                        all_violations = []
                        frame_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        max_frames = fps * 60  # Max 60 seconds
                        frame_count = 0

                        while cap.isOpened() and frame_count < max_frames:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            annotated, violations = detect_frame(frame, conf_threshold)
                            out.write(annotated)
                            all_violations.extend(violations)

                            if frame_count % 10 == 0:
                                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(annotated_rgb, use_column_width=True)

                            progress_bar.progress(min(frame_count / max_frames, 1.0))
                            frame_count += 1

                        cap.release()
                        out.release()

                        st.success("✅ Detection complete!")
                        if all_violations:
                            st.error(f"⚠️ **{len(all_violations)} violation(s) detected!**")
                            for v in set(all_violations):
                                st.warning(f"🔴 {v}")
                        else:
                            st.success("✅ No violations found!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

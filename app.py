# app.py
import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import os

st.set_page_config(page_title="YOLO Video Detection", layout="wide")
st.title("🚗 YOLO Video Object Detection App")

# Upload video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Load YOLO model
model = YOLO("yolov8n.pt")  # Ensure this file is in your directory or provide full path

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    stframe = st.empty()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(tempfile.gettempdir(), "detected_output.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    st.info("Processing video... This may take a while ⏳")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=640, conf=0.5)

        annotated_frame = results[0].plot()

        out.write(annotated_frame)
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    out.release()

    st.success("✅ Detection Complete!")
    with open(output_path, 'rb') as f:
        st.download_button("📥 Download Detected Video", f, file_name="detected_output.mp4")

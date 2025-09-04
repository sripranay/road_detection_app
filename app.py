import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# -------------------------------
# Load models
# -------------------------------
road_model = YOLO("models/best.pt")        # trained model (damages, speed breakers)
vehicle_model = YOLO("models/yolov8s.pt")  # pretrained COCO model

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Road Detection", layout="wide")
st.title("🚦 Road Detection with Live Camera & Uploads")

option = st.radio("Choose input:", ["Upload Image", "Upload Video", "Live Camera"])

# -------------------------------
# Helper function for predictions
# -------------------------------
def detect_objects(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run both models
    road_results = road_model(img, verbose=False)
    vehicle_results = vehicle_model(img, verbose=False)

    # Annotate detections
    img = road_results[0].plot()
    img = vehicle_results[0].plot(img)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------
# Upload Image
# -------------------------------
if option == "Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)

        road_results = road_model(img)
        vehicle_results = vehicle_model(img)

        annotated = road_results[0].plot()
        annotated = vehicle_results[0].plot(annotated)

        st.image(annotated, channels="BGR", use_container_width=True)

# -------------------------------
# Upload Video
# -------------------------------
elif option == "Upload Video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = f"temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(uploaded.read())

        cap = cv2.VideoCapture(tfile)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            road_results = road_model(frame)
            vehicle_results = vehicle_model(frame)
            annotated = road_results[0].plot()
            annotated = vehicle_results[0].plot(annotated)

            st.image(annotated, channels="BGR", use_container_width=True)
        cap.release()

# -------------------------------
# Live Camera
# -------------------------------
elif option == "Live Camera":
    webrtc_streamer(
        key="road-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=detect_objects,
        media_stream_constraints={"video": {"facingMode": "environment"}},  # back camera on phone
    )

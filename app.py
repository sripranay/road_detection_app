import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# -------------------------------
# Load models
# -------------------------------
road_model = YOLO("models/best.pt")        # custom model for road damages, speed breakers
vehicle_model = YOLO("models/yolov8s.pt")  # pretrained YOLOv8s for vehicles, people

# -------------------------------
# UI Setup
# -------------------------------
st.set_page_config(page_title="Road Detection App", layout="wide")
st.title("🚦 Road Detection with Live Camera & Uploads")

option = st.sidebar.radio("Choose Input Mode", ["Upload Image", "Upload Video", "Live Camera"])

# -------------------------------
# Helper Function: Draw Results
# -------------------------------
def process_and_draw(image):
    results_road = road_model(image)
    results_vehicle = vehicle_model(image)

    # Combine both results
    annotated = results_road[0].plot()
    annotated = results_vehicle[0].plot(annotated)

    return annotated

# -------------------------------
# Upload Image
# -------------------------------
if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img, channels="BGR", caption="Uploaded Image", use_container_width=True)

        if st.button("Run Detection"):
            output = process_and_draw(img)
            st.image(output, channels="BGR", caption="Detections", use_container_width=True)

# -------------------------------
# Upload Video
# -------------------------------
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            output = process_and_draw(frame)
            stframe.image(output, channels="BGR", use_container_width=True)

        cap.release()

# -------------------------------
# Live Camera
# -------------------------------
elif option == "Live Camera":

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            output = process_and_draw(img)
            return av.VideoFrame.from_ndarray(output, format="bgr24")

    RTC_CONFIGURATION = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }

    webrtc_streamer(
        key="road-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"facingMode": {"exact": "environment"}},  # 👈 back camera on phones
            "audio": False,
        },
        video_processor_factory=VideoProcessor,
    )

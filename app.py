import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO("models/best.pt")  # keep your weights in /models folder

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Road Detection App", layout="wide")
st.title("🚦 Road Detection App")

menu = ["Live Camera", "Upload Image", "Upload Video"]
choice = st.sidebar.selectbox("Choose mode", menu)

# -------------------------------
# Live camera detection
# -------------------------------
if choice == "Live Camera":
    st.subheader("Live Camera Detection")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
    )

# -------------------------------
# Image upload detection
# -------------------------------
elif choice == "Upload Image":
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img)
        annotated = results[0].plot()

        st.image(annotated, channels="BGR", caption="Detected Objects")

# -------------------------------
# Video upload detection
# -------------------------------
elif choice == "Upload Video":
    st.subheader("Upload a Video")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated = results[0].plot()

            stframe.image(annotated, channels="BGR")

        cap.release()

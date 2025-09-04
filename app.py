import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# Load both models
road_model = YOLO("models/best.pt")         # road damages/speed breakers
vehicle_model = YOLO("models/yolov8s.pt")   # vehicles + humans

st.set_page_config(page_title="Road Detection App", layout="wide")
st.title("🚦 Road Detection with Live Camera & Uploads")

# ----- Image Upload -----
uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_img:
    bytes_data = uploaded_img.read()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    results1 = road_model(img)
    results2 = vehicle_model(img)

    combined = img.copy()
    for r in results1 + results2:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = r.names[int(box.cls[0])]
            cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(combined, cls, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    st.image(combined, channels="BGR", use_container_width=True)

# ----- Video Upload -----
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

        results1 = road_model(frame)
        results2 = vehicle_model(frame)

        combined = frame.copy()
        for r in results1 + results2:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = r.names[int(box.cls[0])]
                cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(combined, cls, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        stframe.image(combined, channels="BGR", use_container_width=True)

    cap.release()

# ----- Live Camera -----
st.subheader("Live Camera Detection")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results1 = road_model(img)
    results2 = vehicle_model(img)

    combined = img.copy()
    for r in results1 + results2:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = r.names[int(box.cls[0])]
            cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(combined, cls, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return combined

webrtc_streamer(
    key="road-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_transform=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {"width": {"ideal": 320}, "height": {"ideal": 240}},  # Lower res for phone
        "audio": False
    }
)

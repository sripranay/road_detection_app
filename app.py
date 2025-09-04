import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# -------------------------------
# Load models
# -------------------------------
road_model = YOLO("models/best.pt")        # custom trained model (road damages, speed breakers)
vehicle_model = YOLO("models/yolov8s.pt")  # pretrained YOLOv8s for vehicles + people

# -------------------------------
# Helper: Run Detection
# -------------------------------
def run_detection(image, use_road_model=True, use_vehicle_model=True):
    annotated = image.copy()

    if use_road_model:
        results = road_model(image)
        for r in results[0].boxes:
            cls_id = int(r.cls)
            label = road_model.names[cls_id]
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
            cv2.putText(annotated, label, (xyxy[0], xyxy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if use_vehicle_model:
        results = vehicle_model(image)
        for r in results[0].boxes:
            cls_id = int(r.cls)
            label = vehicle_model.names[cls_id]
            xyxy = r.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 2)
            cv2.putText(annotated, label, (xyxy[0], xyxy[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return annotated

# -------------------------------
# WebRTC Live Camera
# -------------------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = run_detection(img, use_road_model=True, use_vehicle_model=True)
        return img

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Road Detection App", layout="wide")
st.title("🚦 Road Detection App - Vehicles, Humans & Damages")

option = st.sidebar.radio("Choose Input", ["Upload Image", "Upload Video", "Live Camera"])

if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        annotated = run_detection(image)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated = run_detection(frame)
            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        cap.release()

elif option == "Live Camera":
    webrtc_streamer(
        key="road-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_transformer_factory=VideoTransformer,
    )

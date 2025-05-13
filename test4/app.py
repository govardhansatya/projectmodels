import streamlit as st
import cv2
import base64
import time
import numpy as np
from backend.agents.detection_agent import ObjectDetectionAgent
from backend.agents.vision_context_agent import VisionContextAgent
from backend.db.mongo_logger import MongoLogger
from PIL import Image
from datetime import datetime

# Initialize modules
yolo = ObjectDetectionAgent
agent = VisionContextAgent
logger = MongoLogger

# UI
st.set_page_config(page_title="Smart Room Assistant", layout="wide")
st.title("🏠 Smart Room Assistant Dashboard")

# Streamlit sidebar
st.sidebar.title("⚙️ Controls")
run_stream = st.sidebar.checkbox("Start Webcam")
motion_threshold = st.sidebar.slider("Motion Sensitivity", 0, 100, 30)
show_history = st.sidebar.checkbox("📘 Show Action Timeline")

# Main placeholders
video_placeholder = st.empty()
action_log = st.empty()

# Helper
def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode()

# Motion Detection Helper
def detect_motion(prev_frame, curr_frame, threshold):
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    motion_score = np.sum(thresh) / 255
    return motion_score > 5000  # You can adjust this threshold

# Start webcam
if run_stream:
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    _, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (640, 480))

    while True:
        _, frame2 = cap.read()
        frame2 = cv2.resize(frame2, (640, 480))
        motion_detected = detect_motion(frame1, frame2, motion_threshold)

        frame1 = frame2.copy()
        video_placeholder.image(frame2, channels="BGR")

        if motion_detected:
            st.toast("🎯 Motion Detected!", icon="📹")
            encoded_img = encode_frame(frame2)
            detections = yolo.process(encoded_img)

            decision = agent.process_frame(encoded_img, detections)
            st.toast(f"🤖 Action: {decision}", icon="✅")

        if not cap.isOpened() or not run_stream:
            break

    cap.release()
    cv2.destroyAllWindows()

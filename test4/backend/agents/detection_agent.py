from ultralytics import YOLO
from .reminder_agent import ReminderAgent
from .memory import HistoryMemory
import base64
import cv2
import numpy as np

class ObjectDetectionAgent:
    def __init__(self):
        self.model = YOLO("models/yolov8n.pt")
        self.reminder = ReminderAgent()
        self.memory = HistoryMemory()

    def process(self, image_b64: str):
        img_array = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        results = self.model(frame)[0]
        objects = list(set([self.model.names[int(cls)] for cls in results.boxes.cls]))

        self.memory.save(f"Detected: {objects}")

        # Trigger tasks based on detected objects
        for obj in objects:
            if obj in ["cup", "bottle"]:
                self.reminder.execute(f"Don't forget to drink water - I saw a {obj}")

        return objects

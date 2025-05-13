from agents.vision_context_agent import VisionContextAgent



import cv2
import base64
from agents.detection_agent import ObjectDetectionAgent
from agents.motion_agent import MotionDetectionAgent

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = ObjectDetectionAgent()
motion_agent = MotionDetectionAgent()

_, frame1 = cap.read()

print("[Live Monitor] Starting webcam feed...")
while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    motion_detected = motion_agent.detect_motion(frame1, frame2)

    if motion_detected:
        print("[Live Monitor] Motion detected — running object detection...")

        # Encode the frame for detection
        _, buffer = cv2.imencode('.jpg', frame2)
        jpg_as_text = base64.b64encode(buffer).decode()

        detected_objects = detector.process(jpg_as_text)
        print(f"[Live Monitor] Detected: {detected_objects}")

    # Display the video feed
    cv2.imshow("Smart Room Monitor", frame2)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    frame1 = frame2

cap.release()
cv2.destroyAllWindows()

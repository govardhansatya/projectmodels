import time

class MotionDetectionAgent:
    def __init__(self):
        self.last_motion_time = 0
        self.threshold_seconds = 5

    def detect_motion(self, frame1, frame2):
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        motion_score = cv2.countNonZero(thresh)

        if motion_score > 5000:
            now = time.time()
            if now - self.last_motion_time > self.threshold_seconds:
                self.last_motion_time = now
                print("[MotionAgent] Significant motion detected.")
                return True
        return False

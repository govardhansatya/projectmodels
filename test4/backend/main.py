import subprocess
import uvicorn
from fastapi import FastAPI
from agents.vision_context_agent import VisionContextAgent
from agents.detection_agent import ObjectDetectionAgent
from mongo_logger import MongoLogger
from pydantic import BaseModel
import base64

# --- Initialize components
app = FastAPI()
agent = VisionContextAgent
detector = ObjectDetectionAgent
logger = MongoLogger

# --- Pydantic model
class FrameInput(BaseModel):
    image_b64: str

# --- API Routes
@app.get("/")
def health_check():
    return {"status": "Smart Room Assistant is live!"}

@app.post("/analyze/")
def analyze_frame(data: FrameInput):
    detections = detector.process(data.image_b64)
    decision = agent.process_frame(data.image_b64, detections)
    return {"detections": detections, "decision": decision}

@app.get("/actions/")
def get_actions():
    return logger.get_action_history()

# --- Launch Streamlit UI
def launch_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py"])

# --- Entrypoint
if __name__ == "__main__":
    launch_streamlit()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

import base64
import google.generativeai as genai
from memory import HistoryMemory
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import os
from dotenv import load_dotenv  
load_dotenv()
# Load environment variables from .env file
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
from mongo_logger import MongoLogger
class VisionContextAgent:
    def __init__(self):
        self.memory = HistoryMemory()
        self._setup_gemini()
        self.blip_processor, self.blip_model = self._load_blip()
        self.logger = MongoLogger()

    def _setup_gemini(self):
        genai.configure(api_key=os.getenv(GEMINI_API_KEY))
        try:
            self.model = genai.GenerativeModel("gemini-1.5-pro-vision-latest")
        except Exception as e:
            print("[Gemini Init] Error:", e)
            self.model = None

    def _load_blip(self):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        return processor, model

    def _blip_caption(self, image_bytes):
        image = Image.open(io.BytesIO(base64.b64decode(image_bytes))).convert("RGB")
        inputs = self.blip_processor(image, return_tensors="pt")
        out = self.blip_model.generate(**inputs)
        return self.blip_processor.decode(out[0], skip_special_tokens=True)

    def generate_scene_caption(self, image_b64):
        try:
            img_data = base64.b64decode(image_b64)
            caption = self.model.generate_content([
                "Describe this room scene in detail.",
                genai.types.content.Blob(data=img_data, mime_type="image/jpeg")
            ])
            return caption.text
        except Exception as e:
            print("[Gemini Vision Fallback] Using BLIP — Error:", e)
            return self._blip_caption(image_b64)

    def reason_over_context(self, caption: str, detections: list):
        history = self.memory.get_history()
        try:
            prompt = (
                "You are a smart home AI. Use the current scene and historical context to infer useful actions.\n\n"
                f"Scene Caption: {caption}\n"
                f"Detected Objects: {', '.join(detections)}\n"
                f"History: {history}\n\n"
                "Return any suggested reminders, smart home actions, or logs."
            )

            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print("[Gemini Reasoning Error]", e)
            return "Unable to reason about context at the moment."

    def process_frame(self, image_b64, detections):
        caption = self.generate_scene_caption(image_b64)
        print(f"[Scene Caption]: {caption}")
        decision = self.reason_over_context(caption, detections)
        print(f"[Smart Decision]: {decision}")
        self.memory.save(f"Caption: {caption} | Action: {decision}")
        return decision

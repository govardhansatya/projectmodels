from pymongo import MongoClient
from datetime import datetime
import os

class MongoLogger:
    def __init__(self, db_name="smart_room", collection_name="actions"):
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def log_action(self, caption, detections, decision):
        doc = {
            "timestamp": datetime.utcnow(),
            "caption": caption,
            "detections": detections,
            "decision": decision
        }
        self.collection.insert_one(doc)

    def get_action_history(self, limit=20):
        return list(self.collection.find().sort("timestamp", -1).limit(limit))

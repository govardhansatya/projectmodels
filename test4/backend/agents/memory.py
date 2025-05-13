from pymongo import MongoClient
from datetime import datetime

class HistoryMemory:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["smart_room"]
        self.collection = self.db["history"]

    def save(self, event: str):
        self.collection.insert_one({"event": event, "timestamp": datetime.utcnow()})

    def get_history(self):
        return [entry["event"] for entry in self.collection.find().sort("timestamp", -1).limit(20)]

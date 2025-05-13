import streamlit as st
from pymongo import MongoClient
import os

def load_actions(limit=20):
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(uri)
    db = client["smart_room"]
    collection = db["actions"]
    return list(collection.find().sort("timestamp", -1).limit(limit))

def format_action(entry):
    return f"🕒 {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} \n\n" \
           f"📷 Caption: {entry['caption']}\n" \
           f"🎯 Detected: {', '.join(entry['detections'])}\n" \
           f"🧠 Decision: {entry['decision']}\n"

st.set_page_config(page_title="Action Timeline", layout="wide")
st.title("📘 Smart Action Timeline")

actions = load_actions()

for entry in actions:
    st.markdown("---")
    st.markdown(format_action(entry))

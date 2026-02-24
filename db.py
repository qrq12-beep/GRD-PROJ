import os
import cv2
import time
import base64
import hashlib
import logging
import numpy as np
import pandas as pd

from flask import Flask, render_template, jsonify, request
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque
from typing import Optional

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("violence_system")

# -------------------- DATABASE SETUP --------------------
DB_DIR = Path("violence_db")
DB_DIR.mkdir(exist_ok=True)

MASTER_CSV = DB_DIR / "master_all_events.csv"
STATS_CSV = DB_DIR / "daily_stats.csv"

CSV_FILES = {
    "civil_unrest": DB_DIR / "civil_unrest.csv",
}

# -------------------- DATA MODEL --------------------
@dataclass
class ViolenceEvent:
    event_id: str
    title: str
    description: str
    source_url: str
    source_name: str
    country: str
    region: str
    latitude: Optional[float]
    longitude: Optional[float]
    event_date: str
    collection_timestamp: str
    category: str
    severity: int
    casualty_estimate: Optional[int]
    keywords_matched: str
    raw_hash: str


# -------------------- HELPER FUNCTIONS --------------------
def generate_event_id(url: str, title: str, date: str) -> str:
    raw = f"{url}|{title}|{date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_existing_ids(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["event_id"])
        return set(df["event_id"].astype(str))
    except Exception:
        return set()


def append_to_csv(csv_path: Path, events):
    if not events:
        return
    rows = [asdict(e) for e in events]
    df = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def update_daily_stats(event):
    today = datetime.now().strftime("%Y-%m-%d")
    summary = {
        "date": today,
        "time": datetime.now().strftime("%H:%M:%S"),
        "severity": event.severity,
        "category": event.category,
    }
    df = pd.DataFrame([summary])
    write_header = not STATS_CSV.exists()
    df.to_csv(STATS_CSV, mode="a", header=write_header, index=False)


def log_camera_violence(confidence: int):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str = datetime.now().strftime("%Y-%m-%d")

    title = "Live Camera Fight Detection"
    description = f"Fight detected. Confidence: {confidence}%"

    event_id = generate_event_id("camera_live", title, now_str)
    raw_hash = hashlib.md5(description.encode()).hexdigest()

    event = ViolenceEvent(
        event_id=event_id,
        title=title,
        description=description,
        source_url="camera_live",
        source_name="Live CCTV",
        country="Unknown",
        region="Local",
        latitude=None,
        longitude=None,
        event_date=date_str,
        collection_timestamp=now_str,
        category="civil_unrest",
        severity=min(5, max(3, confidence // 20)),
        casualty_estimate=None,
        keywords_matched="camera_detection",
        raw_hash=raw_hash,
    )

    existing_ids = load_existing_ids(MASTER_CSV)
    if event.event_id not in existing_ids:
        append_to_csv(MASTER_CSV, [event])
        append_to_csv(CSV_FILES["civil_unrest"], [event])
        update_daily_stats(event)
        log.info("Camera violence event recorded.")


# -------------------- FIGHT DETECTOR --------------------
class SmartFightDetector:
    def __init__(self):
        self.prev_frame = None
        self.motion_history = deque(maxlen=15)
        self.fight_confidence = 0

    def detect_fight(self, frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False, 0, 0

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.prev_frame, curr_gray)

        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        motion_score = np.count_nonzero(thresh) / (frame.shape[0] * frame.shape[1]) * 100

        self.motion_history.append(motion_score)
        avg_motion = np.mean(self.motion_history)

        if avg_motion > 20:
            self.fight_confidence = min(100, self.fight_confidence + 5)
        else:
            self.fight_confidence = max(0, self.fight_confidence - 3)

        self.prev_frame = curr_gray

        fight_detected = self.fight_confidence > 60
        return fight_detected, int(avg_motion), int(self.fight_confidence)


# -------------------- CAMERA SYSTEM --------------------
class VideoCamera:
    def __init__(self):
        self.detector = SmartFightDetector()
        self.last_logged_time = 0

    def get_frame(self, frame_data):
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return None, False

        frame = cv2.resize(frame, (640, 480))

        fight, motion, confidence = self.detector.detect_fight(frame)

        if fight and confidence > 60:
            if time.time() - self.last_logged_time > 10:
                log_camera_violence(confidence)
                self.last_logged_time = time.time()

        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes(), fight


# -------------------- FLASK APP --------------------
app = Flask(__name__)
camera = VideoCamera()


@app.route('/')
def index():
    return "Violence Detection System Running"


@app.route('/process_frame', methods=['POST'])
def process_frame():
    image_data = request.files['frame'].read()
    frame_bytes, fight = camera.get_frame(image_data)

    if frame_bytes:
        b64 = base64.b64encode(frame_bytes).decode('utf-8')
        return jsonify({
            "success": True,
            "frame": b64,
            "fight_detected": fight
        })

    return jsonify({"success": False})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
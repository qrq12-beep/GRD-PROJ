"""
Sentinel Web Server — serves the HTML frontend and streams detection data.
Run:  python server.py [--source 0] [--port 5000]
"""

import cv2
import numpy as np
import json
import time
import threading
import argparse
from datetime import datetime
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit

# ── Import the detection engine ──
import importlib, sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Lazy-import the app module (same directory)
app_module = importlib.import_module("app")

# Load models
run_v = app_module.MODE in ("both", "violence")
run_l = app_module.MODE in ("both", "litter")

v_model = app_module.load_violence_model(
    app_module.VIOLENCE_H5_PATH, app_module.VIOLENCE_ONNX_PATH) if run_v else None
l_model = app_module.load_litter_model(
    app_module.LITTER_MODEL_PATH) if run_l else None
f_model = app_module.load_fire_model(
    app_module.FIRE_MODEL_PATH) if app_module.ENABLE_FIRE else None
p_model = app_module.load_person_model()

app_module.warmup(v_model, l_model, f_model, p_model)

import os as _os

# ── Flask setup — serve HTML files from the same directory as server.py ──
_here      = _os.path.dirname(_os.path.abspath(__file__))
flask_app  = Flask(__name__, template_folder=_here, static_folder=_here)
socketio = SocketIO(flask_app, cors_allowed_origins="*")

# Allow index.html to be served without a templates/ subfolder
flask_app.jinja_env.auto_reload = True

# Shared detection state (updated by inference)
detection_state = {
    "is_violent":    False, "v_conf":      0.0,
    "is_littering":  False, "litter_dets": [],
    "is_fire":       False, "is_smoke":    False, "fire_dets": [],
    "person_count":  0,     "person_dets": [],
    "fps":           0.0,   "frame":       0,
}
state_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────────────────
@flask_app.route('/')
def index():
    return render_template('index.html')

@flask_app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@flask_app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@flask_app.route('/alert-triage')
def alert_triage():
    return render_template('alert-triage.html')


@flask_app.route('/api/state')
def api_state():
    with state_lock:
        data = dict(detection_state)
    data['timestamp'] = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    return jsonify(data)


import base64

# ──────────────────────────────────────────────────────────────────────────────
#  Frame processing function
# ──────────────────────────────────────────────────────────────────────────────
def process_frame(frame, v_model, l_model, f_model, p_model):
    # Violence
    frame_buf = deque(maxlen=app_module.SEQUENCE_LEN)
    frame_buf.append(app_module.preprocess_frame_violence(frame))
    is_violent, v_conf = app_module.predict_violence(v_model, frame_buf) if v_model else (False, 0.0)
    
    # Litter
    is_littering, litter_dets = app_module.predict_litter(l_model, frame) if l_model else (False, [])
    
    # Fire
    is_fire, is_smoke, fire_dets = app_module.predict_fire(f_model, frame) if f_model else (False, False, [])
    
    # Persons
    person_count, person_dets = app_module.predict_persons(p_model, frame) if p_model else (0, [])
    
    return {
        'is_violent': is_violent,
        'v_conf': v_conf,
        'is_littering': is_littering,
        'litter_dets': litter_dets,
        'is_fire': is_fire,
        'is_smoke': is_smoke,
        'fire_dets': fire_dets,
        'person_count': person_count,
        'person_dets': person_dets
    }

# ──────────────────────────────────────────────────────────────────────────────
#  WebSocket for frame processing
# ──────────────────────────────────────────────────────────────────────────────
@socketio.on('frame')
def handle_frame(data):
    try:
        # Decode base64 frame
        frame_data = base64.b64decode(data['frame'].split(',')[1])
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Process frame
        result = process_frame(frame, v_model, l_model, f_model, p_model)
        
        # Update state
        with state_lock:
            detection_state.update(result)
            detection_state['fps'] = 0.0  # Not tracking fps here
            detection_state['frame'] += 1
        
        # Annotate frame
        annotated = app_module.draw_overlay(
            frame,
            result['is_violent'], result['v_conf'],
            result['is_littering'], result['litter_dets'],
            result['is_fire'], result['is_smoke'], result['fire_dets'],
            result['person_count'], result['person_dets'],
            0.0  # fps
        )
        
        # Encode annotated frame to base64
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode('utf-8')
        
        # Send back result
        emit('result', {
            'is_violent': result['is_violent'],
            'violence_confidence': result['v_conf'],
            'is_littering': result['is_littering'],
            'litter_detections': [
                {
                    'label': d['label'],
                    'confidence': d['conf'],
                    'bbox': d['bbox']
                } for d in result['litter_dets']
            ],
            'is_fire': result['is_fire'],
            'is_smoke': result['is_smoke'],
            'fire_detections': result['fire_dets'],
            'person_count': result['person_count'],
            'person_detections': result['person_dets'],
            'annotated_frame': annotated_b64,
            'timestamp': datetime.now().isoformat(),
            'fps': 0.0
        })
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('error', {'message': str(e)})

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sentinel Web Server")
    parser.add_argument('--source', default='0', help='Camera index or video path (not used in client mode)')
    parser.add_argument('--port',   default=5000, type=int)
    parser.add_argument('--host',   default='0.0.0.0')
    args = parser.parse_args()

    print(f"\n[Sentinel] Web interface → http://localhost:{args.port}\n")
    socketio.run(flask_app, host=args.host, port=args.port, debug=False)
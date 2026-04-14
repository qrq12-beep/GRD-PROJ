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

# ── Import the detection engine ──
import importlib, sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Lazy-import the app module (same directory)
app_module = importlib.import_module("app")

import os as _os

# ── Flask setup — serve HTML files from the same directory as server.py ──
_here      = _os.path.dirname(_os.path.abspath(__file__))
flask_app  = Flask(__name__, template_folder=_here, static_folder=_here)

# Allow index.html to be served without a templates/ subfolder
flask_app.jinja_env.auto_reload = True

# Shared detection state (updated by inference thread)
detection_state = {
    "is_violent":    False, "v_conf":      0.0,
    "is_littering":  False, "litter_dets": [],
    "is_fire":       False, "is_smoke":    False, "fire_dets": [],
    "person_count":  0,     "person_dets": [],
    "fps":           0.0,   "frame":       0,
}
state_lock = threading.Lock()
latest_frame = [None]
frame_lock   = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
#  Video stream generator  (MJPEG)
# ──────────────────────────────────────────────────────────────────────────────
def _make_placeholder():
    """Return a 640x480 'Connecting...' frame as JPEG bytes."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Connecting to camera...", (120, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


def gen_frames():
    _placeholder = _make_placeholder()
    while True:
        with frame_lock:
            f = latest_frame[0]
        if f is None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                   _placeholder + b'\r\n')
            time.sleep(0.5)
            continue
        _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')
        time.sleep(0.03)


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


@flask_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/api/state')
def api_state():
    with state_lock:
        data = dict(detection_state)
    data['timestamp'] = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    return jsonify(data)


# ──────────────────────────────────────────────────────────────────────────────
#  Detection thread  (mirrors app.run() but writes to shared state)
# ──────────────────────────────────────────────────────────────────────────────
def detection_thread(source):
    import queue
    from collections import deque

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

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    # ── Camera fallback: if index 0 fails, try 1 then 2 ─────────────────
    if not cap.isOpened() and isinstance(src, int):
        for fallback_idx in range(1, 3):
            print(f"[WARN] Camera {src} failed — trying index {fallback_idx} ...")
            cap = cv2.VideoCapture(fallback_idx)
            if cap.isOpened():
                print(f"[INFO] Using camera index {fallback_idx}")
                break

    # ── Set camera to a reliable resolution ──────────────────────────────
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps_cap  = app_module.FPS_CAP
    in_q     = queue.Queue(maxsize=2)
    out_q    = queue.Queue(maxsize=2)
    log_f    = app_module.open_log(app_module.LOG_FILE)
    worker   = app_module.InferenceWorker(
        v_model, l_model, f_model, p_model, in_q, out_q, log_f)
    worker.start()

    frame_count = 0
    prev_gray   = None
    fps_counter = 0
    t_fps       = time.time()
    fps_display = 0.0
    frame_interval = (1.0 / fps_cap) if fps_cap > 0 else 0.0
    t_last = 0.0

    # Last known detection results (shown while waiting for next inference)
    last = dict(is_violent=False, v_conf=0.0,
                is_littering=False, litter_dets=[],
                is_fire=False, is_smoke=False, fire_dets=[],
                person_count=0, person_dets=[])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_interval > 0:
            now = time.time()
            wait = frame_interval - (now - t_last)
            if wait > 0:
                time.sleep(wait)
        t_last = time.time()

        frame_count += 1
        fps_counter += 1
        if fps_counter >= 30:
            fps_display = fps_counter / (time.time() - t_fps)
            t_fps       = time.time()
            fps_counter = 0

        curr_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        if app_module.has_motion(prev_gray, curr_gray):
            prev_gray = curr_gray
            if not in_q.full():
                in_q.put_nowait((frame.copy(), frame_count))

        try:
            result = out_q.get_nowait()
            (last['is_violent'], last['v_conf'],
             last['is_littering'], last['litter_dets'],
             last['is_fire'], last['is_smoke'], last['fire_dets'],
             last['person_count'], last['person_dets']) = result
        except queue.Empty:
            pass

        annotated = app_module.draw_overlay(
            frame.copy(),
            last['is_violent'],   last['v_conf'],
            last['is_littering'], last['litter_dets'],
            last['is_fire'],      last['is_smoke'], last['fire_dets'],
            last['person_count'], last['person_dets'],
            fps_display,
        )

        # ── Write annotated frame into the MJPEG buffer ──────────────────
        with frame_lock:
            latest_frame[0] = annotated

        with state_lock:
            detection_state.update(last)
            detection_state['fps']   = round(fps_display, 1)
            detection_state['frame'] = frame_count

    worker.running = False
    worker.join(timeout=3)
    cap.release()
    if log_f:
        log_f.close()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sentinel Web Server")
    parser.add_argument('--source', default='0', help='Camera index or video path')
    parser.add_argument('--port',   default=5000, type=int)
    parser.add_argument('--host',   default='0.0.0.0')
    args = parser.parse_args()

    # Start detection in background thread
    t = threading.Thread(target=detection_thread, args=(args.source,), daemon=True)
    t.start()

    print(f"\n[Sentinel] Web interface → http://localhost:{args.port}\n")
    flask_app.run(host=args.host, port=args.port, threaded=True,
                  use_reloader=False, debug=False)
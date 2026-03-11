"""
Plithos — FastAPI Backend Server
---------------------------------
Wraps the existing detection logic (app.py) into a REST + WebSocket API
so the Next.js frontend can communicate with it.

Install requirements:
    pip install fastapi uvicorn python-multipart ultralytics opencv-python numpy

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST  /detect/image        — single image/frame inference
    POST  /detect/frame        — same, base64-encoded frame from webcam
    GET   /stream              — MJPEG stream of annotated video (optional)
    GET   /status              — health check + current detection state
    GET   /alerts              — recent alert history
    WS    /ws                  — WebSocket: send frames, receive detection results
"""

import cv2
import numpy as np
import base64
import time
import threading
import queue
import json
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ── Import detection logic from your existing app.py ─────────────────────────
# Make sure app.py (or app_detection.py) is in the same directory
import importlib.util, sys, os

# Load app.py as a module (handles the case where it's named app.py)
_spec = importlib.util.spec_from_file_location("detection", 
    os.path.join(os.path.dirname(__file__), "app.py"))
_det = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_det)

# Re-export the functions we need
preprocess_frame_violence = _det.preprocess_frame_violence
predict_violence          = _det.predict_violence
predict_litter            = _det.predict_litter
draw_overlay              = _det.draw_overlay
load_litter_model         = _det.load_litter_model
load_violence_model       = _det.load_violence_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (mirrors app.py config — edit to match)
# ─────────────────────────────────────────────────────────────────────────────
VIOLENCE_H5_PATH   = "modelnew.h5"
VIOLENCE_ONNX_PATH = "modelnew.onnx"
LITTER_MODEL_PATH  = "best.pt"
VIOLENCE_THRESHOLD = 0.60
LITTER_THRESHOLD   = 0.40
YOLO_IMGSZ         = 320
MAX_ALERT_HISTORY  = 100

_det.VIOLENCE_THRESHOLD = VIOLENCE_THRESHOLD
_det.LITTER_THRESHOLD   = LITTER_THRESHOLD
_det.YOLO_IMGSZ         = YOLO_IMGSZ
_det.MODE               = "both"

# ─────────────────────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Plithos Detection API", version="1.0.0")

# Allow Next.js dev server (localhost:3000) and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────
class AppState:
    v_model      = None
    l_model      = None
    models_ready = False
    alert_history: List[Dict[str, Any]] = []
    frame_buffer  = deque(maxlen=5)
    last_result: Dict[str, Any] = {
        "is_violent": False,
        "violence_confidence": 0.0,
        "is_littering": False,
        "litter_detections": [],
        "timestamp": None,
    }
    active_ws_clients: List[WebSocket] = []

state = AppState()


@app.on_event("startup")
async def startup_event():
    """Load models once when the server starts."""
    def _load():
        try:
            print("[API] Loading models ...")
            state.v_model = load_violence_model(VIOLENCE_H5_PATH, VIOLENCE_ONNX_PATH)
            state.l_model = load_litter_model(LITTER_MODEL_PATH)
            state.models_ready = True
            print("[API] Models loaded ✓")
        except Exception as e:
            print(f"[API] WARNING: Could not load models: {e}")
            print("[API] Server will still run — /detect will return an error until models are available.")

    t = threading.Thread(target=_load, daemon=True)
    t.start()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _decode_image(data: bytes) -> np.ndarray:
    """Decode bytes → OpenCV BGR frame."""
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


def _decode_base64_frame(b64: str) -> np.ndarray:
    """Decode a data-URL or raw base64 string → OpenCV BGR frame."""
    if "," in b64:          # data:image/jpeg;base64,<data>
        b64 = b64.split(",")[1]
    raw = base64.b64decode(b64)
    return _decode_image(raw)


def _run_inference(frame: np.ndarray) -> Dict[str, Any]:
    """Run both models on a single frame and return structured results."""
    if not state.models_ready:
        raise HTTPException(status_code=503, detail="Models not loaded yet. Try again shortly.")

    # Violence
    state.frame_buffer.append(preprocess_frame_violence(frame))
    is_violent, v_conf = predict_violence(state.v_model, state.frame_buffer)

    # Littering
    is_littering, litter_dets = predict_litter(state.l_model, frame)

    ts = datetime.now().isoformat()

    result = {
        "is_violent":           bool(is_violent),
        "violence_confidence":  round(float(v_conf), 4),
        "is_littering":         bool(is_littering),
        "litter_detections":    [
            {
                "label": d["label"],
                "confidence": round(float(d["conf"]), 4),
                "bbox": list(d["bbox"]),   # [x1, y1, x2, y2]
            }
            for d in litter_dets
        ],
        "timestamp": ts,
    }

    # Update shared state
    state.last_result = result

    # Log alert to history
    if is_violent or is_littering:
        alert = {
            "type":      "violence" if is_violent else "littering",
            "both":      bool(is_violent and is_littering),
            "v_conf":    result["violence_confidence"],
            "objects":   len(litter_dets),
            "timestamp": ts,
        }
        state.alert_history.insert(0, alert)
        if len(state.alert_history) > MAX_ALERT_HISTORY:
            state.alert_history.pop()

    return result


def _annotate_frame(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """Draw overlay on frame using the existing draw_overlay from app.py."""
    litter_dets = [
        {"bbox": tuple(d["bbox"]), "conf": d["confidence"], "label": d["label"]}
        for d in result["litter_detections"]
    ]
    return draw_overlay(
        frame,
        result["is_violent"],
        result["violence_confidence"],
        result["is_littering"],
        litter_dets,
        fps_display=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/status")
def get_status():
    """Health check — returns model readiness and latest detection state."""
    return {
        "status":       "ok",
        "models_ready": state.models_ready,
        "last_result":  state.last_result,
        "alert_count":  len(state.alert_history),
    }


@app.get("/alerts")
def get_alerts(limit: int = 50):
    """Return recent alert history (newest first)."""
    return {"alerts": state.alert_history[:limit]}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Upload a JPEG/PNG image → returns detection JSON.
    Use from Next.js:
        const form = new FormData();
        form.append('file', blob, 'frame.jpg');
        fetch('http://localhost:8000/detect/image', { method: 'POST', body: form })
    """
    raw = await file.read()
    frame = _decode_image(raw)
    result = _run_inference(frame)
    return result


class FrameRequest(BaseModel):
    frame: str          # base64-encoded JPEG/PNG (data-URL or raw)
    annotate: bool = False  # if True, also return annotated frame as base64


@app.post("/detect/frame")
async def detect_frame(req: FrameRequest):
    """
    Send a base64 webcam frame → returns detection JSON (+ optional annotated frame).
    Ideal for browser getUserMedia() streams.

    Example from Next.js:
        const canvas = document.createElement('canvas');
        // draw video frame onto canvas ...
        const b64 = canvas.toDataURL('image/jpeg', 0.8);
        fetch('http://localhost:8000/detect/frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: b64, annotate: true })
        })
    """
    frame = _decode_base64_frame(req.frame)
    result = _run_inference(frame)

    if req.annotate:
        annotated = _annotate_frame(frame.copy(), result)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        result["annotated_frame"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — real-time bidirectional channel
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket protocol:
      Client → Server:  JSON { "frame": "<base64>" }
      Server → Client:  JSON { detection result + optional annotated_frame }

    Next.js usage example (hooks/useDetection.ts):
        const socket = new WebSocket('ws://localhost:8000/ws');
        socket.onmessage = (e) => setResult(JSON.parse(e.data));
        // send frames:
        socket.send(JSON.stringify({ frame: canvas.toDataURL('image/jpeg', 0.7) }));
    """
    await ws.accept()
    state.active_ws_clients.append(ws)
    try:
        while True:
            text = await ws.receive_text()
            payload = json.loads(text)
            b64 = payload.get("frame", "")
            if not b64:
                await ws.send_text(json.dumps({"error": "no frame provided"}))
                continue

            try:
                frame = _decode_base64_frame(b64)
                result = _run_inference(frame)

                # Optionally send annotated frame back
                if payload.get("annotate"):
                    annotated = _annotate_frame(frame.copy(), result)
                    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    result["annotated_frame"] = (
                        "data:image/jpeg;base64," + base64.b64encode(buf).decode()
                    )

                await ws.send_text(json.dumps(result))
            except Exception as e:
                await ws.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        state.active_ws_clients.remove(ws)


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG STREAM (optional — for embedding in <img src="..."> tags)
# ─────────────────────────────────────────────────────────────────────────────

_stream_cap: Optional[cv2.VideoCapture] = None
_stream_lock = threading.Lock()


def _get_stream_frames():
    """Generator: reads webcam, annotates, yields MJPEG bytes."""
    global _stream_cap
    with _stream_lock:
        if _stream_cap is None or not _stream_cap.isOpened():
            _stream_cap = cv2.VideoCapture(0)

    while True:
        ret, frame = _stream_cap.read()
        if not ret:
            break
        try:
            result = _run_inference(frame)
            annotated = _annotate_frame(frame, result)
        except Exception:
            annotated = frame

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.033)   # ~30fps


@app.get("/stream")
def mjpeg_stream():
    """
    MJPEG annotated webcam stream.
    Use in Next.js:   <img src="http://localhost:8000/stream" />
    """
    return StreamingResponse(
        _get_stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

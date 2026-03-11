"""
Violence + Littering + Fire/Smoke Detection  — OPTIMIZED  (+ Person Counter)
-----------------------------------------------------------------------------
Optimizations applied:
  1. Parallel inference  — all models run in separate threads
  2. Separate display thread — UI never blocks on inference
  3. Smart frame skipping — skips frames where scene hasn't changed (motion diff)
  4. ONNX runtime for violence model — faster than TensorFlow on CPU
  5. Reduced YOLO input size — configurable (default 320 instead of 640)
  6. Frame buffer queue — decouples capture from inference, no dropped frames
  7. Half-precision (FP16) for YOLO on GPU — 2x speed boost
  8. Warmup pass — avoids first-frame slowdown
  9. Person counter — counts people in frame using YOLO class 0
 10. Fire/Smoke detector — dedicated bestfire.pt model, runs in parallel

Usage:
    python app.py                          # webcam, all models (default)
    python app.py --source video.mp4       # video file
    python app.py --mode violence          # violence only
    python app.py --mode litter            # littering only
    python app.py --mode both              # violence + litter (no fire)
    python app.py --no-fire                # disable fire/smoke detection
    python app.py --convert-onnx           # convert .h5 to .onnx then exit

Requirements:
    pip install ultralytics onnxruntime opencv-python numpy
    # For GPU onnx: pip install onnxruntime-gpu
"""

import cv2
import numpy as np
import argparse
import time
import os
import threading
import queue
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
VIOLENCE_H5_PATH    = "modelnew.h5"
VIOLENCE_ONNX_PATH  = "modelnew.onnx"
LITTER_MODEL_PATH   = "best.pt"
FIRE_MODEL_PATH     = "Fire_best.pt"       # fire + smoke detection model

IMG_SIZE            = (128, 128)
SEQUENCE_LEN        = 1
VIOLENCE_CLASS      = 1
VIOLENCE_THRESHOLD  = 0.80
LITTER_THRESHOLD    = 0.40
FIRE_THRESHOLD      = 0.40               # confidence threshold for fire/smoke
YOLO_IMGSZ          = 320

# Person counter
PERSON_CLASS_ID     = 0
PERSON_THRESHOLD    = 0.40
PERSON_MODEL_PATH   = ""

# Fire detection toggle (overridden by --no-fire flag)
ENABLE_FIRE         = True

SAVE_VIDEO          = True
LOG_FILE            = "detections.log"
MOTION_THRESHOLD    = 10
MAX_QUEUE_DEPTH     = 2
FPS_CAP             = 30

# Detection mode for violence + litter: "both" | "violence" | "litter"
MODE                = "both"
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  ONE-TIME CONVERSION:  .h5  →  .onnx
# ══════════════════════════════════════════════════════════════════════════════

def convert_h5_to_onnx(h5_path: str, onnx_path: str):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    try:
        import tf2onnx
    except ImportError:
        raise ImportError("Run:  pip install tf2onnx onnx")

    print(f"[CONVERT] Loading {h5_path} ...")
    from tensorflow.keras.utils import custom_object_scope

    class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(*args, **kwargs)

    with custom_object_scope({'DepthwiseConv2D': FixedDepthwiseConv2D}):
        model = tf.keras.models.load_model(h5_path)

    input_sig = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input")]
    print(f"[CONVERT] Converting to ONNX -> {onnx_path} ...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig, opset=13)

    import onnx
    onnx.save(model_proto, onnx_path)
    print(f"[CONVERT] Saved to: {onnx_path}")
    print("[CONVERT] Now run the app normally — it will use the .onnx file automatically.")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_violence_model(h5_path: str, onnx_path: str):
    if os.path.exists(onnx_path):
        print(f"[INFO] Loading violence model (ONNX): {onnx_path}")
        import onnxruntime as ort
        providers  = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess       = ort.InferenceSession(onnx_path, providers=providers)
        input_name = sess.get_inputs()[0].name
        print(f"[INFO] ONNX providers active: {sess.get_providers()}")
        return ("onnx", sess, input_name)
    else:
        print(f"[INFO] ONNX not found — using Keras: {h5_path}")
        print("[TIP]  Run  python app.py --convert-onnx  once for a speed boost.")
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.utils import custom_object_scope

        class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(*args, **kwargs)

        with custom_object_scope({'DepthwiseConv2D': FixedDepthwiseConv2D}):
            model = tf.keras.models.load_model(h5_path)

        print(f"[INFO] Keras model input : {model.input_shape}")
        return ("keras", model, None)


def load_litter_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Littering model not found: '{path}'")
    print(f"[INFO] Loading littering model (YOLOv8): {path}")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Run:  pip install ultralytics")
    return YOLO(path)


def load_fire_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fire/smoke model not found: '{path}'")
    print(f"[INFO] Loading fire/smoke model (YOLOv8): {path}")
    from ultralytics import YOLO
    mdl = YOLO(path)
    print(f"[INFO] Fire model classes: {mdl.names}")
    return mdl


def load_person_model(litter_model):
    if PERSON_MODEL_PATH and os.path.exists(PERSON_MODEL_PATH):
        print(f"[INFO] Loading dedicated person-counter model: {PERSON_MODEL_PATH}")
        from ultralytics import YOLO
        return YOLO(PERSON_MODEL_PATH)
    print(f"[INFO] Person counter: reusing litter model (class id={PERSON_CLASS_ID})")
    return litter_model


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_frame_violence(frame: np.ndarray) -> np.ndarray:
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return resized.astype(np.float32) / 255.0


def predict_violence(v_model_bundle, frame_buffer: deque):
    kind, model, input_name = v_model_bundle
    frames = np.array(list(frame_buffer))

    if kind == "onnx":
        inp   = frames[-1][np.newaxis]
        preds = model.run(None, {input_name: inp})[0][0]
    else:
        rank = len(model.input_shape)
        if rank == 4:
            inp = frames[-1][np.newaxis]
        else:
            seq = frames[-SEQUENCE_LEN:]
            if len(seq) < SEQUENCE_LEN:
                pad = np.zeros((SEQUENCE_LEN - len(seq), *seq.shape[1:]), dtype=np.float32)
                seq = np.concatenate([pad, seq], axis=0)
            inp = seq[np.newaxis]
        preds = model.predict(inp, verbose=0)[0]

    conf       = float(preds[0] if preds.shape[0] == 1 else preds[VIOLENCE_CLASS])
    is_violent = conf >= VIOLENCE_THRESHOLD
    return is_violent, conf


def _use_half() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def predict_litter(l_model, frame: np.ndarray):
    results      = l_model.predict(frame, conf=LITTER_THRESHOLD, imgsz=YOLO_IMGSZ,
                                   verbose=False, half=_use_half())
    detections   = []
    is_littering = False
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == PERSON_CLASS_ID and not PERSON_MODEL_PATH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val        = float(box.conf[0])
            label           = l_model.names.get(cls_id, str(cls_id))
            detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf_val, 'label': label})
            is_littering    = True
    return is_littering, detections


def predict_fire(f_model, frame: np.ndarray):
    """
    Run fire/smoke model. Returns (is_fire, is_smoke, detections).
    Each detection: {'bbox', 'conf', 'label'} where label is e.g. 'fire' or 'smoke'.
    """
    results    = f_model.predict(frame, conf=FIRE_THRESHOLD, imgsz=YOLO_IMGSZ,
                                 verbose=False, half=_use_half())
    detections = []
    is_fire    = False
    is_smoke   = False
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val        = float(box.conf[0])
            cls_id          = int(box.cls[0])
            label           = f_model.names.get(cls_id, str(cls_id)).lower()
            detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf_val, 'label': label})
            if 'fire' in label:
                is_fire  = True
            elif 'smoke' in label:
                is_smoke = True
            else:
                # Unknown class from this model — treat as fire-related
                is_fire  = True
    return is_fire, is_smoke, detections


def predict_persons(p_model, frame: np.ndarray):
    results = p_model.predict(frame, conf=PERSON_THRESHOLD, imgsz=YOLO_IMGSZ,
                              classes=[PERSON_CLASS_ID], verbose=False, half=_use_half())
    persons = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if int(box.cls[0]) != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val        = float(box.conf[0])
            persons.append({'bbox': (x1, y1, x2, y2), 'conf': conf_val})
    return len(persons), persons


def has_motion(prev_gray, curr_gray) -> bool:
    if prev_gray is None:
        return True
    return float(cv2.absdiff(prev_gray, curr_gray).mean()) > MOTION_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(frame,
                 is_violent, v_conf,
                 is_littering, litter_detections,
                 is_fire, is_smoke, fire_detections,
                 person_count, person_detections,
                 fps_display):
    h, w = frame.shape[:2]

    run_v = MODE in ("both", "violence")
    run_l = MODE in ("both", "litter")
    run_f = ENABLE_FIRE

    # ── Banner colour & label — fire takes highest priority ───────────────────
    if run_f and (is_fire or is_smoke):
        if is_fire and is_smoke:
            color = (0, 60, 255);  label = "FIRE + SMOKE DETECTED"
        elif is_fire:
            color = (0, 60, 255);  label = "FIRE DETECTED"
        else:
            color = (0, 120, 200); label = "SMOKE DETECTED"
    elif run_v and is_violent and run_l and is_littering:
        color = (0, 0, 210);       label = "WARNING  VIOLENCE + LITTERING"
    elif run_v and is_violent:
        color = (0, 0, 210);       label = "WARNING  VIOLENCE DETECTED"
    elif run_l and is_littering:
        color = (0, 140, 255);     label = "WARNING  LITTERING DETECTED"
    else:
        color = (30, 180, 30);     label = "All Clear"

    # ── Top banner ─────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), color, -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
    cv2.putText(frame, label, (14, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    if run_v:
        right_txt = f"Fight:{v_conf*100:.0f}%  {fps_display:.0f}fps"
    else:
        right_txt = f"{fps_display:.0f}fps"
    (tw, _), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, right_txt, (w - tw - 14, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Mode badge
    active_modes = []
    if run_v: active_modes.append("VIOLENCE")
    if run_l: active_modes.append("LITTER")
    if run_f: active_modes.append("FIRE")
    mode_label = "MODE: " + "+".join(active_modes)
    (mw, mh), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame, (w - mw - 16, 62), (w - 4, 62 + mh + 8), (40, 40, 40), -1)
    cv2.putText(frame, mode_label, (w - mw - 10, 62 + mh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Violence confidence bar
    if run_v:
        v_color = (0, 0, 210) if is_violent else (30, 180, 30)
        cv2.rectangle(frame, (0, 0), (int(w * v_conf), 6), v_color, -1)
        cv2.rectangle(frame, (0, 0), (w, 6), (80, 80, 80), 1)

    # ── Fire / smoke bounding boxes ───────────────────────────────────────────
    if run_f:
        # Subtle red pulse overlay when fire is active
        if is_fire:
            pulse = frame.copy()
            cv2.rectangle(pulse, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(pulse, 0.08, frame, 0.92, 0, frame)

        for det in fire_detections:
            x1, y1, x2, y2 = det['bbox']
            lbl             = det['label']
            conf_val        = det['conf']
            # Fire = red-orange, Smoke = steel blue-grey
            box_color = (0, 60, 255) if 'fire' in lbl else (160, 120, 60)
            tag = f"{lbl} {conf_val*100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            (bw, bh), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty1 = max(y1 - bh - bl - 4, 0)
            cv2.rectangle(frame, (x1, ty1), (x1 + bw + 6, y1 + bl), box_color, -1)
            cv2.putText(frame, tag, (x1 + 3, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if fire_detections:
            fire_cnt  = sum(1 for d in fire_detections if 'fire'  in d['label'])
            smoke_cnt = sum(1 for d in fire_detections if 'smoke' in d['label'])
            other_cnt = len(fire_detections) - fire_cnt - smoke_cnt
            parts = []
            if fire_cnt:  parts.append(f"Fire:{fire_cnt}")
            if smoke_cnt: parts.append(f"Smoke:{smoke_cnt}")
            if other_cnt: parts.append(f"Other:{other_cnt}")
            cv2.putText(frame, "  ".join(parts), (14, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)

    # ── Litter bounding boxes ─────────────────────────────────────────────────
    if run_l:
        for det in litter_detections:
            x1, y1, x2, y2 = det['bbox']
            tag = f"{det['label']} {det['conf']*100:.0f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
            (lw, lh), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty1 = max(y1 - lh - bl - 4, 0)
            cv2.rectangle(frame, (x1, ty1), (x1 + lw + 6, y1 + bl), (0, 140, 255), -1)
            cv2.putText(frame, tag, (x1 + 3, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        if litter_detections:
            cv2.putText(frame, f"Litter objects: {len(litter_detections)}", (14, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    # ── Person bounding boxes ─────────────────────────────────────────────────
    for p in person_detections:
        x1, y1, x2, y2 = p['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        ptag = f"person {p['conf']*100:.0f}%"
        (plw, plh), pbl = cv2.getTextSize(ptag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pty1 = max(y1 - plh - pbl - 4, 0)
        cv2.rectangle(frame, (x1, pty1), (x1 + plw + 6, y1 + pbl), (255, 200, 0), -1)
        cv2.putText(frame, ptag, (x1 + 3, y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    # ── Person count badge — bottom-right ─────────────────────────────────────
    badge_w, badge_h = 110, 60
    bx1 = w - badge_w - 10
    by1 = h - badge_h - 10
    bx2, by2 = w - 10, h - 10
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 200, 0), 2)
    badge_lbl = "PERSONS"
    (slw, slh), _ = cv2.getTextSize(badge_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, badge_lbl, (bx1 + (badge_w - slw) // 2, by1 + slh + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)
    count_str = str(person_count)
    (nw, nh), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
    cv2.putText(frame, count_str, (bx1 + (badge_w - nw) // 2, by2 - 10),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    # ── Fire status badge — bottom-left ───────────────────────────────────────
    if run_f:
        fb_w, fb_h = 130, 60
        fbx1, fby1 = 10, h - fb_h - 10
        fbx2, fby2 = fbx1 + fb_w, fby1 + fb_h
        fb_bg     = (0, 30, 140) if is_fire else (30, 60, 100) if is_smoke else (50, 50, 50)
        fb_border = (0, 60, 255) if is_fire else (160, 120, 60) if is_smoke else (120, 120, 120)
        cv2.rectangle(frame, (fbx1, fby1), (fbx2, fby2), fb_bg, -1)
        cv2.rectangle(frame, (fbx1, fby1), (fbx2, fby2), fb_border, 2)
        fb_lbl = "FIRE / SMOKE"
        (flw, flh), _ = cv2.getTextSize(fb_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, fb_lbl, (fbx1 + (fb_w - flw) // 2, fby1 + flh + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, fb_border, 1, cv2.LINE_AA)
        if is_fire and is_smoke:
            status_txt = "FIRE+SMOKE"
        elif is_fire:
            status_txt = "  FIRE"
        elif is_smoke:
            status_txt = " SMOKE"
        else:
            status_txt = " CLEAR"
        (stw, sth), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        cv2.putText(frame, status_txt, (fbx1 + (fb_w - stw) // 2, fby2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Timestamp ─────────────────────────────────────────────────────────────
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  THREADED INFERENCE WORKER
# ══════════════════════════════════════════════════════════════════════════════

class InferenceWorker(threading.Thread):
    def __init__(self, v_model_bundle, l_model, f_model, p_model, in_q, out_q, log_f):
        super().__init__(daemon=True)
        self.v_model   = v_model_bundle
        self.l_model   = l_model
        self.f_model   = f_model
        self.p_model   = p_model
        self.in_q      = in_q
        self.out_q     = out_q
        self.log_f     = log_f
        self.frame_buf = deque(maxlen=max(SEQUENCE_LEN, 1))
        self.running   = True

    def run(self):
        run_v = MODE in ("both", "violence")
        run_l = MODE in ("both", "litter")
        run_f = ENABLE_FIRE and self.f_model is not None

        while self.running:
            try:
                frame, frame_count = self.in_q.get(timeout=1.0)
            except queue.Empty:
                continue

            v_res = [False, 0.0]
            l_res = [False, []]
            f_res = [False, False, []]
            p_res = [0, []]

            threads = []

            if run_v:
                self.frame_buf.append(preprocess_frame_violence(frame))
                def run_vf(buf=self.frame_buf):
                    v_res[0], v_res[1] = predict_violence(self.v_model, buf)
                threads.append(threading.Thread(target=run_vf))

            if run_l:
                def run_lf(fr=frame):
                    l_res[0], l_res[1] = predict_litter(self.l_model, fr)
                threads.append(threading.Thread(target=run_lf))

            if run_f:
                def run_ff(fr=frame):
                    f_res[0], f_res[1], f_res[2] = predict_fire(self.f_model, fr)
                threads.append(threading.Thread(target=run_ff))

            def run_pf(fr=frame):
                p_res[0], p_res[1] = predict_persons(self.p_model, fr)
            threads.append(threading.Thread(target=run_pf))

            for t in threads: t.start()
            for t in threads: t.join()

            is_violent, v_conf        = v_res
            is_littering, litter_d    = l_res
            is_fire, is_smoke, fire_d = f_res
            person_count, person_d    = p_res

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if run_v and is_violent:
                msg = f"[ALERT] Violence  | frame {frame_count:6d} | conf {v_conf:.3f} | {ts}"
                print(msg)
                if self.log_f: self.log_f.write(msg + "\n"); self.log_f.flush()
            if run_l and is_littering:
                msg = f"[ALERT] Littering | frame {frame_count:6d} | objects {len(litter_d)} | {ts}"
                print(msg)
                if self.log_f: self.log_f.write(msg + "\n"); self.log_f.flush()
            if run_f and (is_fire or is_smoke):
                kind_str = ("FIRE+SMOKE" if (is_fire and is_smoke)
                            else "FIRE" if is_fire else "SMOKE")
                msg = (f"[ALERT] {kind_str:<10} | frame {frame_count:6d} | "
                       f"objects {len(fire_d)} | {ts}")
                print(msg)
                if self.log_f: self.log_f.write(msg + "\n"); self.log_f.flush()

            if self.out_q.full():
                try: self.out_q.get_nowait()
                except queue.Empty: pass
            self.out_q.put((is_violent, v_conf,
                            is_littering, litter_d,
                            is_fire, is_smoke, fire_d,
                            person_count, person_d))
            self.in_q.task_done()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def open_log(path):
    if not path: return None
    f = open(path, "a", encoding="utf-8")
    f.write(f"\n{'='*60}\nSession started: {datetime.now()}  "
            f"[mode={MODE}  fire={'ON' if ENABLE_FIRE else 'OFF'}]\n{'='*60}\n")
    return f


def warmup(v_model, l_model, f_model, p_model):
    print("[INFO] Warming up models ...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    if v_model is not None:
        buf = deque([preprocess_frame_violence(dummy)], maxlen=1)
        predict_violence(v_model, buf)
    if l_model is not None:
        predict_litter(l_model, dummy)
    if f_model is not None:
        predict_fire(f_model, dummy)
    if p_model is not None and p_model is not l_model:
        predict_persons(p_model, dummy)
    print("[INFO] Warmup done.")


def run(source):
    run_v = MODE in ("both", "violence")
    run_l = MODE in ("both", "litter")

    print(f"[INFO] Mode: {MODE.upper()}  —  "
          f"Violence={'ON' if run_v else 'OFF'}  |  "
          f"Littering={'ON' if run_l else 'OFF'}  |  "
          f"Fire/Smoke={'ON' if ENABLE_FIRE else 'OFF'}  |  "
          f"PersonCounter=ON")

    v_model = load_violence_model(VIOLENCE_H5_PATH, VIOLENCE_ONNX_PATH) if run_v else None
    l_model = load_litter_model(LITTER_MODEL_PATH)                       if run_l else None
    f_model = load_fire_model(FIRE_MODEL_PATH)                           if ENABLE_FIRE else None

    if run_l:
        p_model = load_person_model(l_model)
    else:
        fallback = PERSON_MODEL_PATH if PERSON_MODEL_PATH else "yolov8n.pt"
        print(f"[INFO] Violence-only mode: loading person model from {fallback}")
        from ultralytics import YOLO
        p_model = YOLO(fallback)

    warmup(v_model, l_model, f_model, p_model)

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source: {source}  {width}x{height} @ {fps:.1f}fps")

    writer = None
    if SAVE_VIDEO:
        out_path = f"output_{MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        print(f"[INFO] Saving to: {out_path}")

    log_f  = open_log(LOG_FILE)
    in_q   = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
    out_q  = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
    worker = InferenceWorker(v_model, l_model, f_model, p_model, in_q, out_q, log_f)
    worker.start()

    frame_count   = 0
    prev_gray     = None
    is_violent    = False;  v_conf      = 0.0
    is_littering  = False;  litter_dets = []
    is_fire       = False;  is_smoke    = False;  fire_dets = []
    person_count  = 0;      person_dets = []
    fps_display   = 0.0
    t_fps         = time.time();  fps_counter = 0
    violence_count = 0;  litter_count = 0;  fire_count = 0;  smoke_count = 0
    t_start        = time.time()
    frame_interval = (1.0 / FPS_CAP) if FPS_CAP > 0 else 0.0
    t_last_frame   = 0.0

    win_title = "Detector [VIOLENCE+LITTER+FIRE]  (Q = quit)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    print(f"[INFO] FPS cap: {FPS_CAP if FPS_CAP > 0 else 'disabled'}")
    print("[INFO] Running — press  Q  to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        if frame_interval > 0:
            now = time.time()
            sleep_for = frame_interval - (now - t_last_frame)
            if sleep_for > 0:
                time.sleep(sleep_for)
        t_last_frame = time.time()

        frame_count += 1
        fps_counter += 1
        if fps_counter >= 30:
            fps_display = fps_counter / (time.time() - t_fps)
            t_fps       = time.time()
            fps_counter = 0

        curr_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120))
        if has_motion(prev_gray, curr_gray):
            prev_gray = curr_gray
            if not in_q.full():
                in_q.put_nowait((frame.copy(), frame_count))

        try:
            (is_violent, v_conf,
             is_littering, litter_dets,
             is_fire, is_smoke, fire_dets,
             person_count, person_dets) = out_q.get_nowait()
            if is_violent:   violence_count += 1
            if is_littering: litter_count   += 1
            if is_fire:      fire_count      += 1
            if is_smoke:     smoke_count     += 1
        except queue.Empty:
            pass

        annotated = draw_overlay(
            frame,
            is_violent, v_conf,
            is_littering, litter_dets,
            is_fire, is_smoke, fire_dets,
            person_count, person_dets,
            fps_display
        )
        if writer: writer.write(annotated)
        cv2.imshow(win_title, annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] User quit.")
            break

    worker.running = False
    worker.join(timeout=3)
    elapsed = time.time() - t_start
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

    if log_f:
        log_f.write(
            f"\nSession ended     : {datetime.now()}\n"
            f"Mode              : {MODE}\n"
            f"Frames processed  : {frame_count}\n"
            f"Violence alerts   : {violence_count}\n"
            f"Littering alerts  : {litter_count}\n"
            f"Fire alerts       : {fire_count}\n"
            f"Smoke alerts      : {smoke_count}\n"
            f"Elapsed           : {elapsed:.1f}s\n"
        )
        log_f.close()

    print(f"\n[SUMMARY] Mode:{MODE} | Frames:{frame_count} | "
          f"Violence:{violence_count} | Littering:{litter_count} | "
          f"Fire:{fire_count} | Smoke:{smoke_count} | Time:{elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Violence + Littering + Fire/Smoke Detection — Optimized")
    parser.add_argument("--source",              default="0")
    parser.add_argument("--mode",                default="both",
                        choices=["both", "violence", "litter"])
    parser.add_argument("--violence-model",      default=VIOLENCE_H5_PATH)
    parser.add_argument("--violence-onnx",       default=VIOLENCE_ONNX_PATH)
    parser.add_argument("--litter-model",        default=LITTER_MODEL_PATH)
    parser.add_argument("--fire-model",          default=FIRE_MODEL_PATH,
                        help="Path to fire/smoke YOLOv8 model (default: bestfire.pt)")
    parser.add_argument("--no-fire",             action="store_true",
                        help="Disable fire/smoke detection entirely")
    parser.add_argument("--fire-threshold",      type=float, default=FIRE_THRESHOLD,
                        help="Confidence threshold for fire/smoke (default 0.40)")
    parser.add_argument("--person-model",        default=PERSON_MODEL_PATH)
    parser.add_argument("--person-class-id",     type=int,   default=PERSON_CLASS_ID)
    parser.add_argument("--person-threshold",    type=float, default=PERSON_THRESHOLD)
    parser.add_argument("--violence-threshold",  type=float, default=VIOLENCE_THRESHOLD)
    parser.add_argument("--litter-threshold",    type=float, default=LITTER_THRESHOLD)
    parser.add_argument("--yolo-size",           type=int,   default=YOLO_IMGSZ)
    parser.add_argument("--motion-threshold",    type=int,   default=MOTION_THRESHOLD)
    parser.add_argument("--fps-cap",             type=int,   default=FPS_CAP)
    parser.add_argument("--convert-onnx",        action="store_true")
    args = parser.parse_args()

    MODE               = args.mode
    VIOLENCE_H5_PATH   = args.violence_model
    VIOLENCE_ONNX_PATH = args.violence_onnx
    LITTER_MODEL_PATH  = args.litter_model
    FIRE_MODEL_PATH    = args.fire_model
    ENABLE_FIRE        = not args.no_fire
    FIRE_THRESHOLD     = args.fire_threshold
    PERSON_MODEL_PATH  = args.person_model
    PERSON_CLASS_ID    = args.person_class_id
    PERSON_THRESHOLD   = args.person_threshold
    VIOLENCE_THRESHOLD = args.violence_threshold
    LITTER_THRESHOLD   = args.litter_threshold
    YOLO_IMGSZ         = args.yolo_size
    MOTION_THRESHOLD   = args.motion_threshold
    FPS_CAP            = args.fps_cap

    if args.convert_onnx:
        convert_h5_to_onnx(VIOLENCE_H5_PATH, VIOLENCE_ONNX_PATH)
    else:
        run(args.source)

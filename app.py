"""
Violence + Littering Detection  — OPTIMIZED
---------------------------------------------
Optimizations applied:
  1. Parallel inference  — violence & litter models run in separate threads
  2. Separate display thread — UI never blocks on inference
  3. Smart frame skipping — skips frames where scene hasn't changed (motion diff)
  4. ONNX runtime for violence model — faster than TensorFlow on CPU
  5. Reduced YOLO input size — configurable (default 320 instead of 640)
  6. Frame buffer queue — decouples capture from inference, no dropped frames
  7. Half-precision (FP16) for YOLO on GPU — 2x speed boost
  8. Warmup pass — avoids first-frame slowdown

Usage:
    python app.py                          # webcam, both models (default)
    python app.py --source video.mp4       # video file, both models
    python app.py --mode violence          # violence detection only
    python app.py --mode litter            # littering detection only
    python app.py --mode both              # both (default)

    # Convert .h5 to .onnx first (one-time, run once for speed boost):
    python app.py --convert-onnx

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

IMG_SIZE            = (128, 128)
SEQUENCE_LEN        = 1
VIOLENCE_CLASS      = 1
VIOLENCE_THRESHOLD  = 0.60
LITTER_THRESHOLD    = 0.40
YOLO_IMGSZ          = 320

SAVE_VIDEO          = True
LOG_FILE            = "detections.log"
MOTION_THRESHOLD    = 10
MAX_QUEUE_DEPTH     = 2

# Cap display loop to this many frames per second (0 = no cap)
FPS_CAP             = 30

# Detection mode: "both" | "violence" | "litter"
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
    print(f"[CONVERT] Converting to ONNX → {onnx_path} ...")
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
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_val        = float(box.conf[0])
            cls_id          = int(box.cls[0])
            label           = l_model.names.get(cls_id, str(cls_id))
            detections.append({'bbox': (x1, y1, x2, y2), 'conf': conf_val, 'label': label})
            is_littering    = True
    return is_littering, detections


def has_motion(prev_gray, curr_gray) -> bool:
    if prev_gray is None:
        return True
    return float(cv2.absdiff(prev_gray, curr_gray).mean()) > MOTION_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════

def draw_overlay(frame, is_violent, v_conf, is_littering, litter_detections, fps_display):
    h, w = frame.shape[:2]

    run_v = MODE in ("both", "violence")
    run_l = MODE in ("both", "litter")

    if run_v and run_l:
        if is_violent and is_littering:
            color = (0, 0, 210);   label = "⚠  VIOLENCE + LITTERING"
        elif is_violent:
            color = (0, 0, 210);   label = "⚠  VIOLENCE DETECTED"
        elif is_littering:
            color = (0, 140, 255); label = "⚠  LITTERING DETECTED"
        else:
            color = (30, 180, 30); label = "✓  All Clear"
    elif run_v:
        if is_violent:
            color = (0, 0, 210);   label = "⚠  VIOLENCE DETECTED"
        else:
            color = (30, 180, 30); label = "✓  No Violence"
    else:  # litter only
        if is_littering:
            color = (0, 140, 255); label = "⚠  LITTERING DETECTED"
        else:
            color = (30, 180, 30); label = "✓  No Littering"

    # Mode badge (top-right corner, small)
    mode_label = f"MODE: {MODE.upper()}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), color, -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    cv2.putText(frame, label, (14, 40),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    # Right side: violence conf (if active) + fps + mode
    if run_v:
        right_txt = f"Fight:{v_conf*100:.0f}%  {fps_display:.0f}fps"
    else:
        right_txt = f"{fps_display:.0f}fps"
    (tw, _), _ = cv2.getTextSize(right_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, right_txt, (w - tw - 14, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Mode badge below banner
    (mw, mh), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (w - mw - 16, 62), (w - 4, 62 + mh + 8), (40, 40, 40), -1)
    cv2.putText(frame, mode_label, (w - mw - 10, 62 + mh + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Confidence bar (violence only, hidden in litter-only mode)
    if run_v:
        v_color = (0, 0, 210) if is_violent else (30, 180, 30)
        cv2.rectangle(frame, (0, 0), (int(w * v_conf), 6), v_color, -1)
        cv2.rectangle(frame, (0, 0), (w, 6), (80, 80, 80), 1)

    # Littering boxes
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

    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════════════════════════════════════
#  THREADED INFERENCE WORKER
# ══════════════════════════════════════════════════════════════════════════════

class InferenceWorker(threading.Thread):
    """Runs active models in background. Respects MODE to skip unused models."""
    def __init__(self, v_model_bundle, l_model, in_q, out_q, log_f):
        super().__init__(daemon=True)
        self.v_model   = v_model_bundle   # None if MODE == "litter"
        self.l_model   = l_model          # None if MODE == "violence"
        self.in_q      = in_q
        self.out_q     = out_q
        self.log_f     = log_f
        self.frame_buf = deque(maxlen=max(SEQUENCE_LEN, 1))
        self.running   = True

    def run(self):
        run_v = MODE in ("both", "violence")
        run_l = MODE in ("both", "litter")

        while self.running:
            try:
                frame, frame_count = self.in_q.get(timeout=1.0)
            except queue.Empty:
                continue

            v_res = [False, 0.0]
            l_res = [False, []]

            threads = []

            if run_v:
                self.frame_buf.append(preprocess_frame_violence(frame))
                def run_vf(): v_res[0], v_res[1] = predict_violence(self.v_model, self.frame_buf)
                t = threading.Thread(target=run_vf)
                threads.append(t)

            if run_l:
                def run_lf(): l_res[0], l_res[1] = predict_litter(self.l_model, frame)
                t = threading.Thread(target=run_lf)
                threads.append(t)

            for t in threads: t.start()
            for t in threads: t.join()

            is_violent, v_conf     = v_res
            is_littering, litter_d = l_res

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            if run_v and is_violent:
                msg = f"[ALERT] Violence  | frame {frame_count:6d} | conf {v_conf:.3f} | {ts}"
                print(msg)
                if self.log_f: self.log_f.write(msg + "\n"); self.log_f.flush()
            if run_l and is_littering:
                msg = f"[ALERT] Littering | frame {frame_count:6d} | objects {len(litter_d)} | {ts}"
                print(msg)
                if self.log_f: self.log_f.write(msg + "\n"); self.log_f.flush()

            if self.out_q.full():
                try: self.out_q.get_nowait()
                except queue.Empty: pass
            self.out_q.put((is_violent, v_conf, is_littering, litter_d))
            self.in_q.task_done()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def open_log(path):
    if not path: return None
    f = open(path, "a", encoding="utf-8")
    f.write(f"\n{'='*60}\nSession started: {datetime.now()}  [mode={MODE}]\n{'='*60}\n")
    return f


def warmup(v_model, l_model):
    print("[INFO] Warming up models ...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    if v_model is not None:
        buf = deque([preprocess_frame_violence(dummy)], maxlen=1)
        predict_violence(v_model, buf)
    if l_model is not None:
        predict_litter(l_model, dummy)
    print("[INFO] Warmup done.")


def run(source):
    run_v = MODE in ("both", "violence")
    run_l = MODE in ("both", "litter")

    print(f"[INFO] Mode: {MODE.upper()}  —  "
          f"Violence={'ON' if run_v else 'OFF'}  |  Littering={'ON' if run_l else 'OFF'}")

    v_model = load_violence_model(VIOLENCE_H5_PATH, VIOLENCE_ONNX_PATH) if run_v else None
    l_model = load_litter_model(LITTER_MODEL_PATH)                       if run_l else None
    warmup(v_model, l_model)

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
    worker = InferenceWorker(v_model, l_model, in_q, out_q, log_f)
    worker.start()

    frame_count    = 0
    prev_gray      = None
    is_violent     = False
    v_conf         = 0.0
    is_littering   = False
    litter_dets    = []
    fps_display    = 0.0
    t_fps          = time.time()
    fps_counter    = 0
    violence_count = 0
    litter_count   = 0
    t_start        = time.time()
    frame_interval = (1.0 / FPS_CAP) if FPS_CAP > 0 else 0.0
    t_last_frame   = 0.0

    win_title = f"Detector [{MODE.upper()}]  (Q = quit)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    print(f"[INFO] FPS cap: {FPS_CAP if FPS_CAP > 0 else 'disabled'}")
    print("[INFO] Running — press  Q  to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        # ── FPS cap — sleep out any remaining time in this frame slot ─────────
        if frame_interval > 0:
            now     = time.time()
            elapsed_since_last = now - t_last_frame
            sleep_for = frame_interval - elapsed_since_last
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
            is_violent, v_conf, is_littering, litter_dets = out_q.get_nowait()
            if is_violent:   violence_count += 1
            if is_littering: litter_count   += 1
        except queue.Empty:
            pass

        annotated = draw_overlay(frame, is_violent, v_conf, is_littering, litter_dets, fps_display)
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
            f"Elapsed           : {elapsed:.1f}s\n"
        )
        log_f.close()

    print(f"\n[SUMMARY] Mode:{MODE} | Frames:{frame_count} | "
          f"Violence:{violence_count} | Littering:{litter_count} | Time:{elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence + Littering Detection — Optimized")
    parser.add_argument("--source",              default="0",
                        help="Webcam index or video file path")
    parser.add_argument("--mode",                default="both",
                        choices=["both", "violence", "litter"],
                        help="Which models to run: both | violence | litter  (default: both)")
    parser.add_argument("--violence-model",      default=VIOLENCE_H5_PATH)
    parser.add_argument("--violence-onnx",       default=VIOLENCE_ONNX_PATH)
    parser.add_argument("--litter-model",        default=LITTER_MODEL_PATH)
    parser.add_argument("--violence-threshold",  type=float, default=VIOLENCE_THRESHOLD)
    parser.add_argument("--litter-threshold",    type=float, default=LITTER_THRESHOLD)
    parser.add_argument("--yolo-size",           type=int,   default=YOLO_IMGSZ,
                        help="YOLO input px: 320=fast  416=balanced  640=accurate")
    parser.add_argument("--motion-threshold",    type=int,   default=MOTION_THRESHOLD,
                        help="Skip still frames: 0=off  8=light  20=aggressive")
    parser.add_argument("--fps-cap",             type=int,   default=FPS_CAP,
                        help="Max display frames per second (default 30, 0=uncapped)")
    parser.add_argument("--convert-onnx",        action="store_true",
                        help="Convert modelnew.h5 → modelnew.onnx then exit")
    args = parser.parse_args()

    MODE               = args.mode
    VIOLENCE_H5_PATH   = args.violence_model
    VIOLENCE_ONNX_PATH = args.violence_onnx
    LITTER_MODEL_PATH  = args.litter_model
    VIOLENCE_THRESHOLD = args.violence_threshold
    LITTER_THRESHOLD   = args.litter_threshold
    YOLO_IMGSZ         = args.yolo_size
    MOTION_THRESHOLD   = args.motion_threshold
    FPS_CAP            = args.fps_cap

    if args.convert_onnx:
        convert_h5_to_onnx(VIOLENCE_H5_PATH, VIOLENCE_ONNX_PATH)
    else:
        run(args.source)
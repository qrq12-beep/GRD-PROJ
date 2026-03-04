"""
Violence Detection using modelnew.h5
-------------------------------------
Supports: video file or webcam
Usage:
    python app.py                          # webcam
    python app.py --source video.mp4       # video file
    python app.py --source 0               # webcam (explicit)
"""

import cv2
import numpy as np
import argparse
import time
import os
from collections import deque
from datetime import datetime

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH      = "modelnew.h5"
IMG_SIZE        = (128, 128)      # must match model input
SEQUENCE_LEN    = 1               # single-frame model
VIOLENCE_CLASS  = 1
THRESHOLD       = 0.60
SAVE_VIDEO      = True
LOG_FILE        = "detections.log"
INFER_EVERY_N   = 2               # run inference every N frames (1=every frame, 2=every other, etc.)
                                  # lower = smoother but slower; raise if laggy
# ─────────────────────────────────────────────────────────────────────────────


def load_violence_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: '{path}'\n"
            "Make sure modelnew.h5 is in the same folder as this script."
        )
    print(f"[INFO] Loading model from: {path}")

    from tensorflow.keras.utils import custom_object_scope

    # Fix for older models saved with 'groups' in DepthwiseConv2D
    class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(*args, **kwargs)

    with custom_object_scope({'DepthwiseConv2D': FixedDepthwiseConv2D}):
        model = tf.keras.models.load_model(path)

    print(f"[INFO] Model input shape : {model.input_shape}")
    print(f"[INFO] Model output shape: {model.output_shape}")
    return model


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize & normalise a single BGR frame → float32 (does NOT alter display frame)."""
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return resized.astype(np.float32) / 255.0


def predict(model, frame_buffer: deque):
    frames = np.array(list(frame_buffer))
    rank   = len(model.input_shape)

    if rank == 4:
        inp = frames[-1][np.newaxis]           # (1, H, W, C)
    elif rank == 5:
        seq = frames
        if len(seq) < SEQUENCE_LEN:
            pad = np.zeros((SEQUENCE_LEN - len(seq), *seq.shape[1:]), dtype=np.float32)
            seq = np.concatenate([pad, seq], axis=0)
        else:
            seq = seq[-SEQUENCE_LEN:]
        inp = seq[np.newaxis]                  # (1, T, H, W, C)
    else:
        raise ValueError(f"Unexpected model input rank: {rank}")

    preds = model.predict(inp, verbose=0)[0]

    if preds.shape[0] == 1:
        conf       = float(preds[0])
        is_violent = conf >= THRESHOLD
    else:
        conf       = float(preds[VIOLENCE_CLASS])
        is_violent = conf >= THRESHOLD

    return is_violent, conf


def draw_overlay(frame: np.ndarray, is_violent: bool, conf: float) -> np.ndarray:
    """Draw overlay directly on the FULL original frame — no cropping."""
    h, w   = frame.shape[:2]
    label  = "⚠ VIOLENCE DETECTED" if is_violent else "✓ No Violence"
    color  = (0, 0, 210) if is_violent else (30, 180, 30)   # BGR

    # ── Top banner (semi-transparent) ────────────────────────────────────────
    banner_h = 58
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    # Label
    cv2.putText(frame, label, (14, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)

    # Confidence % on the right
    conf_txt = f"{conf * 100:.1f}%"
    (tw, _), _ = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(frame, conf_txt, (w - tw - 14, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Confidence bar (very top, 6 px tall) ─────────────────────────────────
    bar_w = int(w * conf)
    cv2.rectangle(frame, (0, 0), (bar_w, 6), color, -1)
    cv2.rectangle(frame, (0, 0), (w,     6), (80, 80, 80), 1)   # border

    # ── Timestamp (bottom-left) ───────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, ts, (14, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)

    return frame


def open_log(path):
    if path is None:
        return None
    f = open(path, "a", encoding="utf-8")
    f.write(f"\n{'='*60}\nSession started: {datetime.now()}\n{'='*60}\n")
    return f


def run(source):
    model = load_violence_model(MODEL_PATH)

    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Try to set maximum buffer size = 1 to reduce latency on webcam
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source  : {source}")
    print(f"[INFO] Size    : {width}x{height} @ {fps:.1f} fps")

    writer = None
    if SAVE_VIDEO:
        out_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {out_path}")

    log_f          = open_log(LOG_FILE)
    frame_buffer   = deque(maxlen=max(SEQUENCE_LEN, 1))
    frame_count    = 0
    violence_count = 0
    t_start        = time.time()
    is_violent     = False
    conf           = 0.0

    # Pre-warm: create a named window so it appears full-size immediately
    cv2.namedWindow("Violence Detector  (Q = quit)", cv2.WINDOW_NORMAL)

    print("[INFO] Running — press  Q  to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        frame_count += 1
        frame_buffer.append(preprocess_frame(frame))

        # ── Run inference every INFER_EVERY_N frames ──────────────────────────
        if frame_count % INFER_EVERY_N == 0:
            is_violent, conf = predict(model, frame_buffer)

            if is_violent:
                violence_count += 1
                ts  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                msg = f"[ALERT] Violence | frame {frame_count:6d} | conf {conf:.3f} | {ts}"
                print(msg)
                if log_f:
                    log_f.write(msg + "\n")
                    log_f.flush()

        # ── Draw on FULL frame (no resizing/cropping for display) ─────────────
        annotated = draw_overlay(frame, is_violent, conf)   # modifies frame in-place copy

        if writer:
            writer.write(annotated)

        cv2.imshow("Violence Detector  (Q = quit)", annotated)

        # waitKey(1) = fastest possible; increase to ~int(1000/fps) to match source speed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] User quit.")
            break

    # ── Clean up ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if log_f:
        log_f.write(
            f"\nSession ended    : {datetime.now()}\n"
            f"Frames processed : {frame_count}\n"
            f"Violence frames  : {violence_count}\n"
            f"Elapsed time     : {elapsed:.1f}s\n"
        )
        log_f.close()

    print(f"\n[SUMMARY] Frames: {frame_count} | Violence: {violence_count} | Time: {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence Detection — modelnew.h5")
    parser.add_argument("--source",    default="0",         help="Webcam index or video file path")
    parser.add_argument("--model",     default=MODEL_PATH,  help="Path to .h5 model")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Confidence threshold 0-1")
    parser.add_argument("--every",     type=int,   default=INFER_EVERY_N,
                        help="Run inference every N frames (default 2, lower=more accurate/slower)")
    args = parser.parse_args()

    MODEL_PATH    = args.model
    THRESHOLD     = args.threshold
    INFER_EVERY_N = args.every

    run(args.source)
from flask import Flask, render_template, jsonify, request
import cv2
from ultralytics import YOLO
import numpy as np
import time
import base64
from pose_fight import PoseFightDetector

app = Flask(__name__)

# Load pose model
model = YOLO('yolov8n-pose.pt')
model.fuse()

fight_detector = PoseFightDetector()


class VideoCamera:
    def __init__(self):
        self.last_process_time = 0
        self.process_interval = 0.05  # ~20 FPS cap

    def get_frame(self, frame_data):
        try:
            current_time = time.time()

            # Don't fake "no fight" if throttled
            if current_time - self.last_process_time < self.process_interval:
                return None, [], None

            self.last_process_time = current_time

            # Decode frame
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Resize for speed
            frame = cv2.resize(frame, (640, 480))

            # Pose detection
            results = model(
                frame,
                conf=0.4,
                verbose=False,
                device='cpu'
            )

            # 🔥 Fight detection FIRST
            fight_detected = fight_detector.detect(results)

            # Debug prints
            print("Fight status:", fight_detected)
            if fight_detected:
                print("🔥 FIGHT DETECTED")

            # Draw skeleton
            annotated_frame = results[0].plot()

            # Overlay fight warning
            if fight_detected:
                cv2.putText(
                    annotated_frame,
                    "FIGHT DETECTED",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )

            # Encode frame
            _, buffer = cv2.imencode(
                '.jpg',
                annotated_frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            )
            frame_bytes = buffer.tobytes()

            # Optional detections list
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    detections.append({
                        'class': model.names[cls],
                        'confidence': round(conf * 100, 2)
                    })

            return frame_bytes, detections, fight_detected

        except Exception as e:
            print("Frame error:", e)
            return None, [], False


camera = VideoCamera()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        image_data = request.files['frame'].read()

        frame_bytes, detections, fight_detected = camera.get_frame(image_data)

        if frame_bytes:
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
            return jsonify({
                'success': True,
                'frame': frame_base64,
                'detections': detections,
                'fight': fight_detected
            })

        return jsonify({'success': False, 'throttled': True})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import time

app = Flask(__name__)

# Load YOLO model with optimizations
model = YOLO('yolov8n.pt')  # Using nano model for speed
model.fuse()  # Fuse layers for faster inference

class VideoCamera:
    def __init__(self):
        self.last_process_time = 0
        self.min_process_interval = 0.05  # Process max 20 FPS to reduce load
        
    def get_frame(self, frame_data):
        """Process frame from webcam"""
        try:
            # Throttle processing
            current_time = time.time()
            if current_time - self.last_process_time < self.min_process_interval:
                return None, []
            
            self.last_process_time = current_time
            
            # Decode the base64 frame data
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Resize frame for faster processing (optional, adjust as needed)
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                frame = cv2.resize(frame, (640, int(height * scale)))
            
            # Run YOLO detection with optimizations
            results = model(
                frame, 
                conf=0.45,      # Slightly lower confidence for more detections
                iou=0.5,        # IoU threshold for NMS
                verbose=False,  # Disable verbose output
                device='cpu'    # Use CPU (change to 'cuda' if GPU available)
            )
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Encode frame back to jpg with lower quality for speed
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
            ret, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            frame_bytes = buffer.tobytes()
            
            # Get detection info
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    detections.append({
                        'class': model.names[cls],
                        'confidence': round(conf * 100, 2)
                    })
            
            return frame_bytes, detections
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None, []

camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    from flask import request
    try:
        # Get the image data from the request
        image_data = request.files['frame'].read()
        
        frame_bytes, detections = camera.get_frame(image_data)
        
        if frame_bytes:
            import base64
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
            return jsonify({
                'success': True,
                'frame': frame_base64,
                'detections': detections
            })
        else:
            # Return empty response if throttled
            return jsonify({'success': False, 'throttled': True})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Use threaded mode for better performance
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
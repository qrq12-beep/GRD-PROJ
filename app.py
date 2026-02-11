from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import time
import base64
from collections import deque

app = Flask(__name__)

class SmartFightDetector:
    """Ultra-fast 60 FPS fight detection using multi-feature analysis"""
    
    def __init__(self):
        self.prev_frame = None
        self.motion_history = deque(maxlen=15)
        self.fight_confidence = 0
        self.frame_count = 0
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        
    def detect_fight(self, frame):
        """Multi-feature fight detection"""
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # Feature 1: Frame differencing (fastest)
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return False, 0, self.fight_confidence
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.prev_frame, curr_gray)
        
        # Threshold and count changed pixels
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        changed_pixels = np.count_nonzero(thresh)
        frame_diff_score = min(100, (changed_pixels / (h * w)) * 200)
        
        # Feature 2: Foreground detection
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        fg_pixels = np.count_nonzero(fg_mask)
        fg_score = min(100, (fg_pixels / (h * w)) * 300)
        
        # Feature 3: Contour activity
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len([c for c in contours if cv2.contourArea(c) > 50])
        contour_score = min(100, contour_count * 15)
        
        # Weighted combination
        combined_score = (
            frame_diff_score * 0.45 +
            fg_score * 0.35 +
            contour_score * 0.20
        )
        
        # Update history and confidence
        self.motion_history.append(combined_score)
        avg_motion = np.mean(self.motion_history)
        
        # Smart hysteresis for fight detection
        if avg_motion > 50:
            self.fight_confidence = min(100, self.fight_confidence + 6)
        else:
            self.fight_confidence = max(0, self.fight_confidence - 4)
        
        fight_detected = self.fight_confidence > 55
        
        self.prev_frame = curr_gray
        return fight_detected, int(avg_motion), int(self.fight_confidence)


class VideoCamera:
    def __init__(self):
        self.last_process_time = 0
        self.process_interval = 1.0 / 60.0  # 60 FPS
        self.fight_detector = SmartFightDetector()
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
    def get_frame(self, frame_data):
        try:
            current_time = time.time()
            
            # Throttle to 60 FPS
            if current_time - self.last_process_time < self.process_interval:
                return None, [], False
            
            self.last_process_time = current_time
            
            # Decode
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return None, [], False
            
            # Resize for speed
            frame = cv2.resize(frame, (640, 480))
            
            # Detect
            fight_detected, motion_score, confidence = self.fight_detector.detect_fight(frame)
            
            # FPS calc
            self.fps_counter += 1
            if current_time - self.fps_time >= 1.0:
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.fps_time = current_time
            
            # Annotate
            annotated = frame.copy()
            cv2.rectangle(annotated, (5, 5), (400, 120), (20, 20, 20), -1)
            
            # FPS
            cv2.putText(annotated, f'FPS: {self.current_fps}', (15, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Motion bar
            bar_w = int((motion_score / 100) * 150)
            cv2.rectangle(annotated, (15, 45), (165, 65), (100, 100, 100), 2)
            if bar_w > 0:
                col = (0, 0, 255) if fight_detected else (0, 255, 0)
                cv2.rectangle(annotated, (15, 45), (15 + bar_w, 65), col, -1)
            cv2.putText(annotated, f'Motion: {motion_score}%', (180, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Confidence
            conf_col = (0, 255, 0) if confidence < 50 else (0, 165, 255) if confidence < 70 else (0, 0, 255)
            cv2.putText(annotated, f'Alert: {confidence}%', (15, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_col, 2)
            
            # Alert
            if fight_detected:
                cv2.rectangle(annotated, (2, 2), (638, 478), (0, 0, 255), 4)
                cv2.rectangle(annotated, (80, 200), (560, 280), (0, 0, 0), -1)
                cv2.putText(annotated, 'FIGHT DETECTED!', (120, 245),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv2.rectangle(annotated, (80, 200), (560, 280), (0, 0, 255), 3)
            
            # Encode
            _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            
            return buf.tobytes(), [{'class': 'smart_detection', 'confidence': motion_score}], fight_detected
        
        except Exception as e:
            print(f"Error: {e}")
            return None, [], False


camera = VideoCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        image_data = request.files['frame'].read()
        frame_bytes, detections, fight = camera.get_frame(image_data)
        
        if frame_bytes:
            b64 = base64.b64encode(frame_bytes).decode('utf-8')
            return jsonify({'success': True, 'frame': b64, 'detections': detections, 'fight': bool(fight)})
        
        return jsonify({'success': False, 'throttled': True})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

import threading
from flask import Flask, render_template, Response, jsonify
import cv2
import app as plithos_app  # Import our refactored app.py

app = Flask(__name__)

# Generator function to stream frames from our detection app
def generate_frames():
    # We will run the plithos app in web mode.
    # We pass the default source '0' (webcam).
    # Since run() is a generator when web_mode is True, we iterate over it.
    for frame in plithos_app.run(source="0", web_mode=True):
        if frame is None:
            continue
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield the output frame in the byte format required by multipart/x-mixed-replace
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    """API endpoint to get the latest detection statistics."""
    # Return the global stats updated by the inference thread in app.py
    return jsonify({
        "mode": plithos_app.MODE.upper(),
        "is_violent": plithos_app.WEB_STATS.get("is_violent", False),
        "violence_conf": round(plithos_app.WEB_STATS.get("violence_conf", 0.0), 2),
        "is_littering": plithos_app.WEB_STATS.get("is_littering", False),
        "litter_count": plithos_app.WEB_STATS.get("litter_count", 0),
        "person_count": plithos_app.WEB_STATS.get("person_count", 0),
        "fps": round(plithos_app.WEB_STATS.get("fps", 0.0), 1)
    })

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

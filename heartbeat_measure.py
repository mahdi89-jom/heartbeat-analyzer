import cv2
import numpy as np
import base64
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize variables for heart rate calculation
last_heartbeat_time = None
heartbeat_intervals = []
MIN_HEART_RATE = 40  # Minimum expected heart rate (bpm)
MAX_HEART_RATE = 180  # Maximum expected heart rate (bpm)

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_frame.shape

    # Crop the center of the face
    center_x, center_y = width // 2, height // 2
    crop_size = min(width, height) // 2  # Adjust the crop size as needed
    x1, y1 = center_x - crop_size // 2, center_y - crop_size // 2
    x2, y2 = center_x + crop_size // 2, center_y + crop_size // 2

    cropped_frame = gray_frame[y1:y2, x1:x2]
    resized_frame = cv2.resize(cropped_frame, (64, 64))
    normalized_frame = resized_frame / 255.0  # Normalize
    return normalized_frame.flatten()

def predict_heartbeat(frame):
    preprocessed_frame = preprocess_frame(frame)

    # Implement a more sophisticated heart rate detection algorithm here
    # For demonstration, let's use a simple mean change detection approach
    # This is a placeholder and should be replaced with a real algorithm
    intensity_change = np.mean(np.diff(preprocessed_frame))  
    return intensity_change

def calculate_heart_rate(heartbeat_intervals):
    if len(heartbeat_intervals) < 2:
        return None  # Not enough data to calculate heart rate
    average_interval = np.mean(heartbeat_intervals)
    heart_rate = 60 / average_interval  # Heart rate in beats per minute

    # Ignore out-of-range heart rates
    if heart_rate < MIN_HEART_RATE or heart_rate > MAX_HEART_RATE:
        return None
    return heart_rate

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    global last_heartbeat_time, heartbeat_intervals
    
    image_data = request.json.get('image')
    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)
    # Convert bytes to OpenCV image
    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    result = predict_heartbeat(frame)
    
    current_time = time.time()
    
    # Define a threshold for heartbeat detection
    threshold = 0.5
    if result > threshold:
        if last_heartbeat_time is not None:
            interval = current_time - last_heartbeat_time
            heartbeat_intervals.append(interval)
            # Keep the last N intervals (e.g., last 10)
            if len(heartbeat_intervals) > 10:
                heartbeat_intervals.pop(0)
        last_heartbeat_time = current_time

    heart_rate = calculate_heart_rate(heartbeat_intervals)
    
    response = {'result': result}
    if heart_rate is not None:
        response['heart_rate'] = heart_rate
    else:
        response['error'] = 'No valid heart rate detected'

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

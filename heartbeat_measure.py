import cv2
import numpy as np
import base64
import time
from flask import Flask, request, jsonify

# Initialize variables for heart rate calculation
last_heartbeat_time = None
heartbeat_intervals = []

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Crop the upper part of the face (e.g., top 1/3 of the image)
    height, width = gray_frame.shape
    crop_height = int(height / 3)
    cropped_frame = gray_frame[:crop_height, :]
    resized_frame = cv2.resize(cropped_frame, (64, 64))
    normalized_frame = resized_frame / 255.0  # Ensure normalization
    return normalized_frame.flatten()

def calculate_heart_rate(heartbeat_intervals):
    if len(heartbeat_intervals) < 2:
        return None  # Not enough data to calculate heart rate
    average_interval = np.mean(heartbeat_intervals)
    heart_rate = 60 / average_interval  # Heart rate in beats per minute
    return heart_rate

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    global last_heartbeat_time, heartbeat_intervals
    
    image_data = request.json.get('image')
    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)
    # Convert bytes to OpenCV image
    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Simulate heartbeat detection (e.g., based on simple color change)
    # For testing purposes, let's assume a heartbeat is detected
    heartbeat_detected = True  # Replace with actual detection logic if available
    
    current_time = time.time()
    
    if heartbeat_detected:
        if last_heartbeat_time is not None:
            interval = current_time - last_heartbeat_time
            heartbeat_intervals.append(interval)
            # Keep the last N intervals (e.g., last 10)
            if len(heartbeat_intervals) > 10:
                heartbeat_intervals.pop(0)
        last_heartbeat_time = current_time

    heart_rate = calculate_heart_rate(heartbeat_intervals)
    
    response = {'result': int(heartbeat_detected)}  # Simulate result as 1 (heartbeat detected)
    if heart_rate is not None:
        response['heart_rate'] = heart_rate
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

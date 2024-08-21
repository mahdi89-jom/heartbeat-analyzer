import cv2
import numpy as np
import joblib
import base64
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

try:
    # Load the pre-trained model
    clf = joblib.load('heartbeat_model.pkl')  # Adjust the path if needed
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize variables for heart rate calculation
last_heartbeat_time = None
heartbeat_intervals = []

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame / 255.0  # Ensure normalization
    return normalized_frame.flatten()

def predict_heartbeat(frame):
    try:
        preprocessed_frame = preprocess_frame(frame)
        prediction = clf.predict([preprocessed_frame])
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0  # Default to no heartbeat

def calculate_heart_rate(heartbeat_intervals):
    if len(heartbeat_intervals) < 2:
        return None  # Not enough data to calculate heart rate
    average_interval = np.mean(heartbeat_intervals)
    heart_rate = 60 / average_interval  # Heart rate in beats per minute
    return heart_rate

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    global last_heartbeat_time, heartbeat_intervals

    try:
        image_data = request.json.get('image')
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        # Convert bytes to OpenCV image
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Crop the upper part of the face (adjust as needed)
        height, width, _ = frame.shape
        cropped_frame = frame[0:int(height/2), :]  # Crop the upper half

        result = predict_heartbeat(cropped_frame)  # Use the trained model

        current_time = time.time()

        if result == 1:  # Heartbeat detected
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

        return jsonify(response)

    except Exception as e:
        print(f"Error in /analyze endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

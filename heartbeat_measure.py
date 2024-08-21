from flask import Flask, request, jsonify
import numpy as np
import cv2
import joblib
import base64
import time

# Load the pre-trained model
clf = joblib.load('heartbeat_model.pkl')

# Initialize variables for heart rate calculation
last_heartbeat_time = None
heartbeat_intervals = []

def preprocess_frame(frame):
    """
    Preprocess the image frame: convert to grayscale, resize, and normalize.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))  # Ensure the size matches the model input
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return normalized_frame.flatten()

def predict_heartbeat(frame):
    """
    Predict heartbeat presence using the pre-trained model.
    """
    preprocessed_frame = preprocess_frame(frame)
    prediction = clf.predict([preprocessed_frame])
    return prediction[0]

def calculate_heart_rate(heartbeat_intervals):
    """
    Calculate heart rate based on intervals between detected beats.
    """
    if len(heartbeat_intervals) < 2:
        return None
    average_interval = np.mean(heartbeat_intervals)
    heart_rate = 60 / average_interval  # Convert interval to beats per minute
    return heart_rate

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    """
    Handle POST requests, predict heartbeat presence, and calculate heart rate.
    """
    global last_heartbeat_time, heartbeat_intervals
    
    try:
        # Get image data from the request
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Predict heartbeat presence
        result = predict_heartbeat(frame)
        
        # Calculate heart rate if a heartbeat is detected
        current_time = time.time()
        if result == 1:  # Heartbeat detected
            if last_heartbeat_time is not None:
                interval = current_time - last_heartbeat_time
                heartbeat_intervals.append(interval)
                if len(heartbeat_intervals) > 10:
                    heartbeat_intervals.pop(0)
            last_heartbeat_time = current_time

        heart_rate = calculate_heart_rate(heartbeat_intervals)
        
        # Prepare the response
        response = {'result': result}
        if heart_rate is not None:
            response['heart_rate'] = heart_rate
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

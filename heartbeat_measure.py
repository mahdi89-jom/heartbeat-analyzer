from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib
import base64
import time

# Load the pre-trained model
clf = joblib.load('heartbeat_model.pkl')  # Update with the correct path

# Initialize variables for heart rate calculation
last_heartbeat_time = None
heartbeat_intervals = []

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    return normalized_frame.flatten()

def predict_heartbeat(frame):
    preprocessed_frame = preprocess_frame(frame)
    prediction = clf.predict([preprocessed_frame])
    return prediction[0]

def calculate_heart_rate(heartbeat_intervals):
    if len(heartbeat_intervals) < 2:
        return None
    average_interval = np.mean(heartbeat_intervals)
    heart_rate = 60 / average_interval
    return heart_rate

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    global last_heartbeat_time, heartbeat_intervals
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Image decoding failed")
        
        result = predict_heartbeat(frame)
        
        current_time = time.time()
        
        if result == 1:
            if last_heartbeat_time is not None:
                interval = current_time - last_heartbeat_time
                heartbeat_intervals.append(interval)
                if len(heartbeat_intervals) > 10:
                    heartbeat_intervals.pop(0)
            last_heartbeat_time = current_time

        heart_rate = calculate_heart_rate(heartbeat_intervals)
        
        response = {'result': result}
        if heart_rate is not None:
            response['heart_rate'] = heart_rate
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import cv2
import numpy as np
import base64
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize variables for heart rate calculation
frame_results = []

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame / 255.0  # Ensure normalization
    return normalized_frame.flatten()

def predict_heartbeat(frame):
    # Placeholder for actual prediction logic
    preprocessed_frame = preprocess_frame(frame)
    intensity_change = np.mean(np.diff(preprocessed_frame))  # Simplified analysis for heartbeat detection
    return intensity_change

def calculate_average_heart_rate(frame_results):
    if not frame_results:
        return None
    
    valid_results = [hr for hr in frame_results if 40 <= hr <= 180]  # Filter out unrealistic heart rate values
    
    if not valid_results:
        return None
    
    return np.mean(valid_results)

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    global frame_results
    
    image_data = request.json.get('image')
    
    # Decode base64 image data
    image_bytes = base64.b64decode(image_data)
    # Convert bytes to OpenCV image
    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Crop the center of the face
    h, w, _ = frame.shape
    cropped_frame = frame[h//4:h*3//4, w//4:w*3//4]  # Crop to the center
    
    # Analyze the frame
    heart_rate = predict_heartbeat(cropped_frame)
    
    # Store the heart rate for later averaging
    frame_results.append(heart_rate)
    
    # Calculate and return the average heart rate
    average_heart_rate = calculate_average_heart_rate(frame_results)
    
    response = {'heart_rate': average_heart_rate if average_heart_rate is not None else 'No valid heart rate detected'}
    
    # Clear the frame results after processing to prepare for the next set of frames
    frame_results = []
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

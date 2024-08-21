from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import time

app = Flask(__name__)

# Initialize variables for heart rate calculation
last_pulse_time = None
pulse_intervals = []

def preprocess_frame(frame):
    """
    Preprocess the image frame: convert to grayscale, resize, and normalize.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))  # Resize for consistency
    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    return normalized_frame

def calculate_heart_rate(pulse_intervals):
    """
    Calculate heart rate from pulse intervals.
    """
    if len(pulse_intervals) < 2:
        return None  # Not enough data to calculate heart rate
    average_interval = np.mean(pulse_intervals)
    heart_rate = 60 / average_interval  # Heart rate in beats per minute
    return heart_rate

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    """
    Handle POST requests, preprocess image data, and perform basic heart rate estimation.
    """
    global last_pulse_time, pulse_intervals
    
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
        
        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)
        average_intensity = np.mean(preprocessed_frame)  # Calculate average intensity
        
        current_time = time.time()
        
        # Detect changes in intensity to estimate pulse
        if last_pulse_time is not None:
            time_interval = current_time - last_pulse_time
            if average_intensity > 0.5:  # Arbitrary threshold for pulse detection
                pulse_intervals.append(time_interval)
                if len(pulse_intervals) > 10:  # Keep the last 10 intervals
                    pulse_intervals.pop(0)
        
        last_pulse_time = current_time
        
        # Calculate heart rate
        heart_rate = calculate_heart_rate(pulse_intervals)
        
        response = {
            'result': 1,  # Indicating heartbeat detected
            'heart_rate': heart_rate if heart_rate is not None else 0  # Dummy heart rate value if None
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

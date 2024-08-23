import cv2
import numpy as np
import base64
import time
from flask import Flask, request, jsonify
from scipy.signal import butter, filtfilt, find_peaks

# Initialize variables for heart rate calculation
heartbeat_intervals = []
last_heartbeat_time = None
MIN_HEART_RATE = 40  # Minimum expected heart rate (bpm)
MAX_HEART_RATE = 180  # Maximum expected heart rate (bpm)
fs = 30  # Frame rate (samples per second)

def preprocess_frame(frame):
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = gray_frame / 255.0  # Normalize
        return normalized_frame.flatten()
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def bandpass_filter(signal, fs, lowcut=0.67, highcut=3.0, order=5):
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal)
        return y
    except Exception as e:
        print(f"Error during bandpass filtering: {e}")
        return None

def detect_peaks(signal, distance):
    try:
        peaks, _ = find_peaks(signal, distance=distance)
        return peaks
    except Exception as e:
        print(f"Error during peak detection: {e}")
        return []

def calculate_heart_rate(heartbeat_intervals):
    if len(heartbeat_intervals) < 2:
        return None  # Not enough data to calculate heart rate
    try:
        average_interval = np.mean(heartbeat_intervals)
        heart_rate = 60 / average_interval  # Heart rate in beats per minute

        # Ignore out-of-range heart rates
        if heart_rate < MIN_HEART_RATE or heart_rate > MAX_HEART_RATE:
            return None
        return heart_rate
    except Exception as e:
        print(f"Error during heart rate calculation: {e}")
        return None

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    global last_heartbeat_time, heartbeat_intervals
    
    try:
        image_data = request.json.get('image')
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        preprocessed_frame = preprocess_frame(frame)
        if preprocessed_frame is None:
            return jsonify({'error': 'Error during preprocessing'}), 500

        intensity_value = np.mean(preprocessed_frame)
        heartbeat_intervals.append(intensity_value)
        
        filtered_signal = bandpass_filter(heartbeat_intervals, fs)
        if filtered_signal is None:
            return jsonify({'error': 'Error during filtering'}), 500

        peaks = detect_peaks(filtered_signal, distance=fs//2)
        heart_rate = calculate_heart_rate(peaks)
        
        response = {'intensity_value': intensity_value}
        if heart_rate is not None:
            response['heart_rate'] = heart_rate
        
        return jsonify(response)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import cv2
import numpy as np
import base64
import time
from flask import Flask, request, jsonify
from scipy.signal import butter, filtfilt, find_peaks

# Initialize variables for heart rate calculation
heartbeat_intervals = []
last_heartbeat_time = None
MIN_HEART_RATE = 40  # Minimum expected heart rate (bpm)
MAX_HEART_RATE = 180  # Maximum expected heart rate (bpm)
fs = 30  # Frame rate (samples per second)

def preprocess_frame(frame):
    try:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized_frame = gray_frame / 255.0  # Normalize
        return normalized_frame.flatten()
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def bandpass_filter(signal, fs, lowcut=0.67, highcut=3.0, order=5):
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, signal)
        return y
    except Exception as e:
        print(f"Error during bandpass filtering: {e}")
        return None

def detect_peaks(signal, distance):
    try:
        peaks, _ = find_peaks(signal, distance=distance)
        return peaks
    except Exception as e:
        print(f"Error during peak detection: {e}")
        return []

def calculate_heart_rate(heartbeat_intervals):
    if len(heartbeat_intervals) < 2:
        return None  # Not enough data to calculate heart rate
    try:
        average_interval = np.mean(heartbeat_intervals)
        heart_rate = 60 / average_interval  # Heart rate in beats per minute

        # Ignore out-of-range heart rates
        if heart_rate < MIN_HEART_RATE or heart_rate > MAX_HEART_RATE:
            return None
        return heart_rate
    except Exception as e:
        print(f"Error during heart rate calculation: {e}")
        return None

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    global last_heartbeat_time, heartbeat_intervals
    
    try:
        image_data = request.json.get('image')
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        preprocessed_frame = preprocess_frame(frame)
        if preprocessed_frame is None:
            return jsonify({'error': 'Error during preprocessing'}), 500

        intensity_value = np.mean(preprocessed_frame)
        heartbeat_intervals.append(intensity_value)
        
        filtered_signal = bandpass_filter(heartbeat_intervals, fs)
        if filtered_signal is None:
            return jsonify({'error': 'Error during filtering'}), 500

        peaks = detect_peaks(filtered_signal, distance=fs//2)
        heart_rate = calculate_heart_rate(peaks)
        
        response = {'intensity_value': intensity_value}
        if heart_rate is not None:
            response['heart_rate'] = heart_rate
        
        return jsonify(response)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    

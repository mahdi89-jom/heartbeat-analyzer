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
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    normalized_frame = gray_frame / 255.0  # Normalize
    return normalized_frame.flatten()

def bandpass_filter(signal, fs, lowcut=0.67, highcut=3.0, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def detect_peaks(signal, distance):
    peaks, _ = find_peaks(signal, distance=distance)
    return peaks

def calculate_heart_rate(heartbeat_intervals):
    if len(heartbeat_intervals) < 2:
        return None  # Not enough data to calculate heart rate
    average_interval = np.mean(heartbeat_intervals)
    heart_rate = 60 / average_interval  # Heart rate in beats per minute

    # Ignore out-of-range heart rates
    if heart_rate < MIN_HEART_RATE or heart_rate > MAX_HEART_RATE:
        return None
    return heart_rate

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_heartbeat():
    global last_heartbeat_time, heartbeat_intervals
    
    image_data = request.json.get('image')
    image_bytes = base64.b64decode(image_data)
    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    preprocessed_frame = preprocess_frame(frame)
    
    # Calculate the mean intensity
    intensity_value = np.mean(preprocessed_frame)
    
    # Append the intensity value for filtering
    heartbeat_intervals.append(intensity_value)
    
    # Use a bandpass filter to isolate heart rate frequencies
    filtered_signal = bandpass_filter(heartbeat_intervals, fs)
    
    # Detect peaks corresponding to heartbeats
    peaks = detect_peaks(filtered_signal, distance=fs//2)
    
    # Calculate heart rate based on detected peaks
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fs
        heart_rate = 60.0 / np.mean(intervals)
    else:
        heart_rate = None
    
    response = {'intensity_value': intensity_value}
    if heart_rate is not None:
        response['heart_rate'] = heart_rate
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

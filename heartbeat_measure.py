from flask import Flask, request, jsonify
from PIL import Image
import base64
import io
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get the image data from the JSON payload
        data = request.get_json()
        image_data = data['image']

        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Optionally convert the image to a format suitable for your model
        # For example, convert to numpy array and then to OpenCV format
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Process the image (e.g., extract heart rate, etc.)
        # Replace this with your actual processing code
        heart_rate = process_image(image_cv)

        # Return the result as JSON
        return jsonify({"heart_rate": heart_rate})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_image(image):
    # Placeholder function to simulate heart rate analysis
    # Replace this with your actual model inference code
    return 72.5  # Example heart rate

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

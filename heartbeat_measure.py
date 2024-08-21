from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Connection successful"}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Use the port from the environment variable or default to 5000
    app.run(host='0.0.0.0', port=port)

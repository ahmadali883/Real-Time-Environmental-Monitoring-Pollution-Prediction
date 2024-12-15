from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
from logging.handlers import RotatingFileHandler
from prometheus_client import Counter, generate_latest, Histogram

app = Flask(__name__)

# Configure logging with rotation to prevent log file from growing indefinitely
handler = RotatingFileHandler('app.log', maxBytes=1000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total Request Count')
PREDICTION_COUNT = Counter('prediction_count', 'Total Predictions Made')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Latency of requests in seconds')

# Load the best LSTM model
try:
    model = load_model('best_lstm_model.keras')  # Updated to .keras format
    app.logger.info("LSTM model loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load the model: {e}")
    model = None

# Load the scaler
try:
    scaler = joblib.load('scaler.joblib')
    app.logger.info("Scaler loaded successfully.")
except Exception as e:
    app.logger.error(f"Failed to load the scaler: {e}")
    scaler = None

@app.route('/predict', methods=['POST'])
@REQUEST_LATENCY.time()
def predict():
    REQUEST_COUNT.inc()
    if model is None or scaler is None:
        app.logger.error("Model or scaler not loaded.")
        return jsonify({'error': 'Model or scaler not loaded.'}), 500

    try:
        data = request.json
        # Expected JSON structure:
        # {
        #     "history": [5]  # AQI values with length equal to look_back
        # }
        history = data.get('history')
        if not history:
            app.logger.warning("Invalid input: 'history' field is missing.")
            return jsonify({'error': 'Invalid input. "history" field is required.'}), 400

        # Convert history to numpy array
        history = np.array(history).reshape(-1, 1)

        # Ensure history length matches look_back
        look_back = 1  # Ensure this matches the training
        if len(history) != look_back:
            app.logger.warning(f"Invalid history length: {len(history)}. Expected: {look_back}.")
            return jsonify({'error': f'Invalid history length. Expected: {look_back}'}), 400

        # Scale the data
        history_scaled = scaler.transform(history)

        # Prepare the input data
        X_input = history_scaled.reshape(1, look_back, 1)

        # Make prediction
        prediction_scaled = model.predict(X_input)
        prediction = scaler.inverse_transform(prediction_scaled)
        predicted_aqi = prediction[0][0]

        # Convert NumPy float32 to native Python float
        predicted_aqi = float(predicted_aqi)

        # Optionally, round the prediction
        predicted_aqi = round(predicted_aqi, 2)

        PREDICTION_COUNT.inc()
        app.logger.info(f"Prediction made: {predicted_aqi}")
        return jsonify({'predicted_aqi': predicted_aqi})

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/metrics', methods=['GET'])
def metrics():
    return generate_latest()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

from flask import Flask, jsonify, Response, stream_with_context
import firebase_admin
from firebase_admin import credentials, db
import time
import json
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from collections import deque

app = Flask(__name__)
CORS(app)

# --- Machine Learning Model Setup ---
try:
    model = joblib.load(r"C:\Users\gsmlv\OneDrive\Desktop\BTP\Code-1\Codes\human_activity_model.pkl")
    le = joblib.load(r"C:\Users\gsmlv\OneDrive\Desktop\BTP\Code-1\Codes\label_encoder.pkl")
    print("✅ Model and Label Encoder loaded successfully.")
except FileNotFoundError:
    print("🔴 Error: Model or Label Encoder files not found. Please ensure they are in the same directory.")
    exit()

# --- Prediction Constants & Data Window ---
TIME_STEPS = 4  # Must match the window size used during training (2 seconds @ 2Hz)
data_window = deque(maxlen=TIME_STEPS)
sensor_cols = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ']

# --- Firebase Setup ---
cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://data-1-8da90-default-rtdb.asia-southeast1.firebasedatabase.app" # Your URL is kept
})
root_ref = db.reference("/IMUData/test1")

# --- Low-Pass Filter State (moved from JS to Python) ---
ALPHA = 0.1
filtered_state = {}

def apply_low_pass_filter(raw_data):
    """Applies a low-pass filter to raw sensor data."""
    global filtered_state
    if not filtered_state: # First run
        filtered_state = raw_data.copy()
        return filtered_state
    
    for sensor in ['acc', 'gyro']:
        for axis in ['x', 'y', 'z']:
            raw_val = raw_data[sensor][axis]
            last_filtered_val = filtered_state[sensor][axis]
            filtered_state[sensor][axis] = ALPHA * raw_val + (1 - ALPHA) * last_filtered_val
    return filtered_state.copy()


def predict_activity(window):
    """Processes a window of data, extracts features, and returns a prediction."""
    if len(window) < TIME_STEPS:
        return "--" # Not enough data yet

    # Convert deque to a DataFrame for feature extraction
    df = pd.DataFrame(list(window), columns=sensor_cols)
    
    feature_row = []
    for sensor in sensor_cols:
        # Note: No separate noise reduction needed here as feature extraction is robust
        feature_row.append(df[sensor].mean())
        feature_row.append(df[sensor].std())
        feature_row.append(df[sensor].min())
        feature_row.append(df[sensor].max())
        feature_row.append(np.sqrt(np.mean(df[sensor]**2))) # RMS
    
    # Reshape for the model and predict
    features = np.array(feature_row).reshape(1, -1)
    prediction_encoded = model.predict(features)
    prediction_label = le.inverse_transform(prediction_encoded)
    
    return prediction_label[0]


def parse_data_string(value_string):
    """Helper function to parse the comma-separated data string."""
    values = value_string.split(",")
    # We now expect 6 values (AccX,Y,Z, GyroX,Y,Z), ignoring the timestamp.
    if len(values) < 6: return None
    try:
        return {
            # Timestamp key has been removed.
            "acc": {"x": int(values[0]), "y": int(values[1]), "z": int(values[2])},
            "gyro": {"x": int(values[3]), "y": int(values[4]), "z": int(values[5])}
        }
    except (ValueError, IndexError):
        return None

@app.route("/history")
def get_history():
    # This endpoint is kept for initial chart population, but real-time is the focus.
    data = root_ref.order_by_key().limit_to_last(100).get() # Reduced history for faster load
    if not data:
        return jsonify({"error": "No data", "history": []})
    
    history = []
    for _, val_string in data.items():
        parsed_data = parse_data_string(val_string)
        if parsed_data:
            filtered_data = apply_low_pass_filter(parsed_data)
            history.append({"raw": parsed_data, "filtered": filtered_data})
    return jsonify({"history": history})


@app.route("/stream")
def stream():
    """Streams new data, filtered data, and the live prediction."""
    def event_stream():
        last_sent_key = None
        while True:
            data = root_ref.order_by_key().limit_to_last(1).get()
            if data:
                current_key = list(data.keys())[0]
                if current_key != last_sent_key:
                    last_sent_key = current_key
                    data_string = data[current_key]
                    raw_data = parse_data_string(data_string)
                    
                    if raw_data:
                        # 1. Apply low-pass filter
                        filtered_data = apply_low_pass_filter(raw_data)
                        
                        # 2. Add data to prediction window
                        flat_data = [
                            raw_data['acc']['x'], raw_data['acc']['y'], raw_data['acc']['z'],
                            raw_data['gyro']['x'], raw_data['gyro']['y'], raw_data['gyro']['z']
                        ]
                        data_window.append(flat_data)
                        
                        # 3. Predict activity
                        activity_prediction = predict_activity(data_window)
                        
                        # 4. Construct payload and send
                        payload = {
                            "raw": raw_data,
                            "filtered": filtered_data,
                            "prediction": activity_prediction
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.1) 
            
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True) # Set debug=False for production


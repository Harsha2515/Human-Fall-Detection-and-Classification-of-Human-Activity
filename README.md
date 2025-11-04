# Real-time Human Fall Detection & Activity Classification  

This is an end-to-end IoT + Machine Learning project that captures human motion via a wearable IMU sensor, classifies six human activities in real time, and visualizes them on a live web dashboard.  
The system is specifically trained to detect activities such as **Sit, Stand, Walk, Sleep, Steady,** and **Fall**.

---

## Features
- **Real-time Activity Classification:** Classifies 6 different human activities, including "Fall".  
- **Live Web Dashboard:** Single-page interface displaying incoming sensor data and predictions.  
- **Dual Data View:** Shows both raw and low-pass filtered sensor data for clarity.  
- **Instant Prediction Display:** Continuously updates the current predicted activity in real time.

---

## System Architecture
1. **Hardware (Data Collection):**  
   - IMU sensor (MPU-6886) connected to NodeMCU (ESP8266).  
   - Reads 6-axis accelerometer and gyroscope data.  

2. **Data Transmission:**  
   - ESP8266 streams sensor data as JSON to **Google Firebase Realtime Database** via Wi-Fi.  

3. **Backend (Data Processing & AI):**  
   - **Flask** server loads `human_activity_model.pkl` and `label_encoder.pkl`.  
   - Connects securely to Firebase using `firebase_config.json`.  
   - Maintains a sliding window of recent readings, extracts features, and predicts activity.  
   - Streams predictions and sensor data to the frontend via **Server-Sent Events (SSE)**.  

4. **Frontend (Visualization):**  
   - `index.html` connects to `/stream` endpoint and renders live charts using **Chart.js**.  
   - Displays raw & filtered data and the current activity label.

---

## Technologies Used
**Hardware:** NodeMCU (ESP8266), MPU-6886  
**Cloud:** Google Firebase Realtime Database  
**ML:** Python, Scikit-learn (RandomForestClassifier), Pandas, NumPy, Joblib  
**Backend:** Flask, Firebase Admin SDK, Flask-CORS  
**Frontend:** HTML5, JavaScript, Chart.js, SSE  

---

## Setup & Usage

### Firebase Setup
1. Go to your Firebase project → **Project Settings > Service accounts**.  
2. Click **Generate new private key** to download a JSON file.  
3. Rename it to `firebase_config.json` and place it in the project root.  
4. In `app.py`, confirm that the `databaseURL` matches your Firebase project URL.

### Machine Learning Model
Place your trained model files in the root directory:

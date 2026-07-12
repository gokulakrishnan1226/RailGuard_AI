# System Verification & Testing Guide

This guide details how to verify the AI models, API, and ESP32 hardware integrations.

## 1. Running Real-Time Inference
To start patrolling using your local webcam:
1. Ensure your camera is plugged in.
2. In `config.json`, verify `"webcam_index"` is set to your camera's port (usually `0` or `1`).
3. Run the API server:
   ```powershell
   uvicorn api.main:app --reload
   ```
4. Access the dashboard: `http://127.0.0.1:8000/dashboard`
5. The system will process frames in real-time, draw green boxes around authorized officers, red boxes around intruders, orange boxes around animals, and highlight rail cracks. Non-blocking audio alerts will play automatically.

## 2. API Endpoint Verification
You can query endpoints using cURL or Postman:

- **Login Authentication**:
  ```bash
  curl -X POST http://127.0.0.1:8000/api/login -H "Content-Type: application/json" -d '{"username": "admin", "password": "admin123"}'
  ```
- **Fetch Alert Logs**:
  ```bash
  curl -X GET http://127.0.0.1:8000/api/alerts
  ```
- **Query Patroller Location**:
  ```bash
  curl -X GET http://127.0.0.1:8000/api/location
  ```

## 3. ESP32 Hardware Emulation & Tests
If you don't have an ESP32 board, you can mock ESP32 communication:
1. In `config.json`, leave `"enabled": false` inside the `"esp32"` dictionary.
2. The python server will skip HTTP requests to the microcontroller, avoiding network timeout lag.
3. You can still test WiFi overrides on the web UI by clicking buttons on the dashboard control console.

# RailGuard AI: Railway Track Monitoring System

RailGuard AI is a complete, production-quality, modular, and AI-powered railway track patrolling, inspection, and security system. It combines classical machine learning, deep learning computer vision architectures, a real-time FastAPI backend, a dark glassmorphic web dashboard, and ESP32 microcontroller communication.

## Quick Start in 3 Steps

### 1. Install Requirements
Open a terminal in the project directory:
```powershell
pip install -r requirements.txt
```

### 2. Train Models (ML & DL)
Generate the synthetic dataset, compare ML classifiers, train PyTorch classifiers and UNet segmenters, and save weights:
```powershell
python training/pipeline_runner.py
```
This trains all models and outputs charts (ROC curve, confusion matrix) and trained model formats (`best.pt`, `best.onnx`, `best.tflite`) inside `models/trained_models/`.

### 3. Launch Web Control Center
Run the REST API and live patrol webcam server:
```powershell
uvicorn api.main:app --reload
```
Open your browser and navigate to: [http://127.0.0.1:8000/login](http://127.0.0.1:8000/login)

Login details:
- **Username**: `admin`
- **Password**: `admin123`

---

## Detailed Manuals
- See [INSTALLATION.md](file:///d:/Projects/Railguard%20ai/docs/INSTALLATION.md) for initial setup instructions.
- See [TRAINING.md](file:///d:/Projects/Railguard%20ai/docs/TRAINING.md) for custom dataset configurations.
- See [TESTING.md](file:///d:/Projects/Railguard%20ai/docs/TESTING.md) for webcam and endpoint test examples.
- See [DEPLOYMENT.md](file:///d:/Projects/Railguard%20ai/docs/DEPLOYMENT.md) for Nginx proxying configurations.

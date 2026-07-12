# RailGuard AI: Technical Overview & Documentation

RailGuard AI is a production-grade, modular, real-time Railway Track Monitoring System designed to secure railway infrastructure, track cleanliness, and structural integrity.

## Project Architecture
The system integrates Machine Learning, Deep Learning, Computer Vision, a FastAPI REST Backend, a dark glassmorphic telemetry dashboard, and WiFi actuator communication with an ESP32 patroller.

```mermaid
graph TD
    Camera[Patrol Camera / Webcam] -->|Video Feed| FastAPI[FastAPI Server]
    ESP32[ESP32 Patroller WiFi] -->|Vibration & GPS Telemetry| FastAPI
    FastAPI -->|Stream Frames| Detector[RailGuard Detector]
    
    subgraph AI Inference Pipeline
        Detector --> Model1[Human Intrusion Model]
        Detector --> Model2[Officer Verification]
        Detector --> Model3[Animal Intrusion Model]
        Detector --> Model4[Obstacle Detector]
        Detector --> Model5[UNet Crack Segmenter]
        Detector --> Model6[Cleanliness Classifier]
        Detector --> Model7[Track Damage Classifier]
    end
    
    Detector -->|Aggregated Detections| Database[MySQL Database Logs]
    Detector -->|JSON Directives| ESP32
    Detector -->|Voice Alerts| Speaker[Local Audio TTS]
    FastAPI -->|JSON telemetry & MJPEG video| Dashboard[Glassmorphic Web UI]
```

## Folder Structure
- `database/`: Schema setup and connection pooler with mock in-memory fallback.
- `datasets/`: Preprocessing, augmentations, and synthetic data loader generator.
- `models/`: Unified wrappers and architectures for classifiers and segmenters.
- `training/`: Machine Learning and PyTorch Deep Learning train scripts and pipeline runner.
- `inference/`: Integrated detection algorithms, barcode vest scanner, uniform checker, and TTS voice alert engine.
- `api/`: REST endpoint routing, CORS handling, and video streamer.
- `website/`: Templates and static glassmorphic style sheets.
- `esp32/`: Microcontroller firmwares.
- `utils/`: Common loggers, configuration parameters, and bounding box conversions.
- `docs/`: Guides for deployment and execution.

## Next Steps
For setups, training pipelines, and deployment parameters, see:
1. [INSTALLATION.md](file:///d:/Projects/Railguard%20ai/docs/INSTALLATION.md)
2. [TRAINING.md](file:///d:/Projects/Railguard%20ai/docs/TRAINING.md)
3. [TESTING.md](file:///d:/Projects/Railguard%20ai/docs/TESTING.md)
4. [DEPLOYMENT.md](file:///d:/Projects/Railguard%20ai/docs/DEPLOYMENT.md)

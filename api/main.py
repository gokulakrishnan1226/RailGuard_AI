import os
import time
import json
import cv2
import numpy as np
import requests
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel, Field
from typing import Optional, Dict

# Global command queue for ESP32 devices
esp32_commands: Dict[str, str] = {}

from utils.config import config
from utils.logger import setup_logger
from database.connection import db_manager
from api.auth import verify_password, get_password_hash, create_access_token, decode_access_token
from inference.detector import RailGuardDetector

logger = setup_logger("api_main")
esp32_logger = setup_logger("esp32", log_file="esp32.log")
# FastAPI App configuration
app = FastAPI(
    title="RailGuard AI",
    description="Production-grade Railway Track Monitoring System API",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Locate templates & static directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(BASE_DIR, "website", "static")
templates_dir = os.path.join(BASE_DIR, "website", "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Initialize detector
detector = RailGuardDetector()

# Live Video Stream Generator Helper
class CameraFeed:
    """Accesses webcam or generates simulated railway animation if no hardware present."""
    
    def __init__(self):
        self.webcam_idx = config["inference"].get("webcam_index", 0)
        self.cap = cv2.VideoCapture(self.webcam_idx)
        self.is_simulated = False
        
        if not self.cap.isOpened():
            logger.warning(f"Webcam index {self.webcam_idx} not found. Launching track simulation camera.")
            self.is_simulated = True
            self.sim_frame_count = 0

    def generate_simulated_frame(self):
        """Generates an animated 3D-like rail track frame in memory."""
        width, height = 640, 480
        # Gray gravel background
        img = np.ones((height, width, 3), dtype=np.uint8) * 110
        # Add random gravel texture noise
        noise = np.random.randint(-10, 10, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Sleeper spacing animation
        self.sim_frame_count += 1
        offset = (self.sim_frame_count * 5) % 80
        
        # Draw wooden sleepers moving downward to simulate track movement
        for y in range(-40 + offset, height + 80, 80):
            # Scale sleeper width for perspective
            scale = 1.0 + (y / height) * 0.8
            sl_w = int(240 * scale)
            sl_h = int(18 * scale)
            x1 = (width - sl_w) // 2
            x2 = (width + sl_w) // 2
            cv2.rectangle(img, (x1, y), (x2, y + sl_h), (60, 45, 35), -1)
            
        # Draw converging tracks (perspective layout)
        cv2.line(img, (int(width * 0.38), 0), (int(width * 0.15), height), (180, 180, 180), 8)
        cv2.line(img, (int(width * 0.62), 0), (int(width * 0.85), height), (180, 180, 180), 8)
        
        # Add simulated objects periodically
        # Human alert simulation
        if 150 < (self.sim_frame_count % 500) < 250:
            # Draw orange safety vest intruder
            cx, cy = int(width * 0.4), int(height * 0.5)
            # Vest
            cv2.rectangle(img, (cx-15, cy-20), (cx+15, cy+15), (5, 97, 240), -1) # Orange BGR
            # Head
            cv2.circle(img, (cx, cy-30), 10, (150, 200, 240), -1)
            # Legs
            cv2.line(img, (cx-8, cy+15), (cx-8, cy+45), (50, 50, 50), 3)
            cv2.line(img, (cx+8, cy+15), (cx+8, cy+45), (50, 50, 50), 3)
            
        # Obstacle alert simulation
        if 350 < (self.sim_frame_count % 500) < 450:
            # Draw gray stone boulder on track
            cx, cy = int(width * 0.5), int(height * 0.6)
            cv2.circle(img, (cx, cy), 15, (60, 60, 60), -1)
            cv2.circle(img, (cx+5, cy-3), 10, (80, 80, 80), -1)
            
        # Crack simulation
        if (self.sim_frame_count % 500) < 100:
            # Jagged black crack on left track
            y_start = int(height * 0.3)
            for y_curr in range(y_start, y_start + 80, 10):
                x_offset = int((y_curr / height) * -160)
                tx = int(width * 0.38) + x_offset
                cv2.line(img, (tx, y_curr), (tx + np.random.randint(-3, 3), y_curr + 10), (10, 10, 10), 3)

        return img

    def get_frame(self):
        """Reads frame and handles failures."""
        if self.is_simulated:
            time.sleep(0.033) # 30 FPS cap
            return True, self.generate_simulated_frame()
            
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to grab camera frame. Retrying...")
            return False, None
        return True, frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

camera = CameraFeed()

# ESP32 JSON Communication Handler
def send_to_esp32(alerts_list):
    """Sends JSON alerts to ESP32WiFi listener."""
    if not config["esp32"].get("enabled", False):
        return
        
    esp_ip = config["esp32"].get("ip", "192.168.1.100")
    esp_port = config["esp32"].get("port", 80)
    url = f"http://{esp_ip}:{esp_port}/control"
    
    # Compose active commands based on detections
    payload = {
        "motor": "stop" if len(alerts_list) > 0 else "forward",
        "buzzer": True if len(alerts_list) > 0 else False,
        "led": "red" if len(alerts_list) > 0 else "green",
        "servo": 45 if "Rail Crack Detected" in alerts_list else 90
    }
    
    try:
        requests.post(url, json=payload, timeout=0.2)
    except Exception as e:
        logger.debug(f"Could not connect to ESP32: {e}")

# Pydantic schema for Login request
class LoginRequest(PydanticBaseModel):
    username: str
    password: str

# Pydantic models for ESP32 API
class ESP32Status(PydanticBaseModel):
    device_id: str
    ip_address: str
    wifi_strength: int
    heap_memory: int
    uptime: int
    status: str
    battery: int
    laser: bool
    motor: str
    gps: bool
    vibration: float

class ESP32Command(PydanticBaseModel):
    device_id: str
    command: str

def verify_esp32_api_key(request: Request):
    api_key = request.headers.get("x-api-key")
    expected_key = config["esp32"].get("api_key", "")
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API Key")
    return True

# --- HTTP ENDPOINTS & WEB ROUTES ---

@app.get("/", response_class=HTMLResponse)
def index_page():
    """Renders the login UI."""
    return HTMLResponse("<!DOCTYPE html><html><script>window.location.href='/login';</script></html>")

@app.get("/login", response_class=HTMLResponse)
def get_login(request: Request):
    """Renders the dashboard login page."""
    return templates.TemplateResponse(request, "login.html")

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request):
    """Renders the control panel telemetry dashboard."""
    return templates.TemplateResponse(request, "index.html")

@app.get("/esp32", response_class=HTMLResponse)
def get_esp32_dashboard(request: Request):
    """Renders the ESP32 control panel dashboard."""
    return templates.TemplateResponse(request, "esp32.html")

# --- REST APIs ---

@app.post("/api/login")
def api_login(req: LoginRequest):
    """Authenticates admin operator and returns session token."""
    user = db_manager.get_user_by_username(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
        
    if verify_password(req.password, user["password_hash"]):
        access_token = create_access_token(data={"sub": user["username"], "role": user["role"]})
        return {
            "status": "success",
            "access_token": access_token,
            "token_type": "bearer",
            "role": user["role"]
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password")

@app.post("/api/detect")
def api_detect(file: UploadFile = File(...)):
    """API endpoint to run prediction analysis on a single uploaded photo."""
    try:
        # Read uploaded image bytes
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
            
        # Run inference
        _, results = detector.process_frame(img)
        
        # Log to database
        for d in results["detections"]:
            db_manager.add_alert(
                detection_type=d.get("label", d.get("class")),
                confidence=d.get("confidence", 0.8),
                officer_verified=d.get("verified", False)
            )
            
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Upload detect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
def api_alerts(limit: int = 20):
    """Returns active or historical list of alerts."""
    return db_manager.get_alerts(limit=limit)

@app.get("/api/history")
def api_history(limit: int = 50):
    """Fetches list of alerts logs."""
    alerts = db_manager.get_alerts(limit=limit)
    return {"status": "success", "count": len(alerts), "history": alerts}

@app.get("/api/location")
def api_location():
    """Returns the real-time GPS coordinates of the track inspection system."""
    telemetry = db_manager.get_latest_telemetry()
    if telemetry:
        return {
            "latitude": float(telemetry["gps_latitude"]),
            "longitude": float(telemetry["gps_longitude"]),
            "timestamp": telemetry["timestamp"]
        }
    # Return Delhi station coordinates by default
    return {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

@app.get("/api/status")
def api_status():
    """Returns comprehensive track details."""
    telemetry = db_manager.get_latest_telemetry()
    
    # Run test frame to get cleanliness and damage metrics
    # In real deployment, this queries active tracking variables
    vibe = telemetry["vibration_level"] if telemetry else 1.25
    
    return {
        "cleanliness": "Highly_Dirty" if vibe > 6.0 else ("Dirty" if vibe > 3.0 else "Clean"),
        "damage": "Broken_Rail" if vibe > 8.0 else "Normal_Track",
        "vibration_level": vibe,
        "gps": {
            "latitude": telemetry["gps_latitude"] if telemetry else 28.6139,
            "longitude": telemetry["gps_longitude"] if telemetry else 77.2090
        }
    }

# --- ESP32 APIs ---

@app.post("/api/esp32/register")
def esp32_register(status: ESP32Status, _: bool = Depends(verify_esp32_api_key)):
    db_manager.update_esp32_status(**status.dict())
    return {"status": "registered"}

@app.post("/api/esp32/status")
def esp32_status_update(status: ESP32Status, _: bool = Depends(verify_esp32_api_key)):
    db_manager.update_esp32_status(**status.dict())
    return {"status": "updated"}

@app.post("/api/esp32/heartbeat")
def esp32_heartbeat(status: ESP32Status, _: bool = Depends(verify_esp32_api_key)):
    esp32_logger.debug(f"Heartbeat from {status.device_id}")
    db_manager.update_esp32_status(**status.dict())
    return {"status": "acknowledged"}

@app.get("/api/esp32/command")
def esp32_get_command(device_id: str, _: bool = Depends(verify_esp32_api_key)):
    cmd = esp32_commands.pop(device_id, "NONE")
    return {"command": cmd}

@app.post("/api/esp32/queue_command")
def esp32_queue_command(cmd: ESP32Command):
    """Internal API to queue a command from the web dashboard."""
    esp32_commands[cmd.device_id] = cmd.command
    esp32_logger.info(f"Queued command {cmd.command} for {cmd.device_id}")
    return {"status": "queued"}

@app.post("/api/esp32/telemetry")
def esp32_telemetry(status: ESP32Status, _: bool = Depends(verify_esp32_api_key)):
    db_manager.add_telemetry(vibration_level=status.vibration, motor_state=status.motor)
    db_manager.update_esp32_status(**status.dict())
    return {"status": "logged"}

@app.post("/api/esp32/alert")
def esp32_alert(payload: dict, _: bool = Depends(verify_esp32_api_key)):
    esp32_logger.warning(f"ESP32 Alert: {payload}")
    return {"status": "alert_received"}

@app.get("/api/esp32/ping")
def esp32_ping():
    return {"status": "pong"}

@app.get("/api/esp32/latest")
def esp32_get_latest_status():
    """Returns the latest status for the web dashboard."""
    status = db_manager.get_esp32_status()
    return status if status else {}

# --- VIDEO MJPEG FEED ROUTE ---

def generate_video_stream():
    """Continuously reads camera frames, runs detector models, and yields MJPEG streams."""
    last_db_log_time = 0
    
    while True:
        success, frame = camera.get_frame()
        if not success:
            continue
            
        # Get simulated GPS telemetry
        gps_lat = 28.6139 + np.sin(time.time() / 100.0) * 0.005
        gps_lon = 77.2090 + np.cos(time.time() / 100.0) * 0.005
        
        # Run detection
        processed_frame, results = detector.process_frame(frame, gps_coords=(gps_lat, gps_lon))
        
        # Limit DB writing frequency to once per 3 seconds per alert type
        now = time.time()
        if now - last_db_log_time > 3.0:
            alerts_found = results["alerts"]
            
            # Log telemetries
            vib_level = float(np.random.normal(1.2, 0.4))
            if "Obstacle on Track" in alerts_found:
                vib_level += 4.5
            if "Rail Crack Detected" in alerts_found:
                vib_level += 6.2
                
            db_manager.add_telemetry(
                vibration_level=vib_level,
                gps_lat=gps_lat,
                gps_lon=gps_lon,
                motor_state="STOPPED" if len(alerts_found) > 0 else "FORWARD",
                buzzer_state=len(alerts_found) > 0,
                servo_angle=45 if "Rail Crack Detected" in alerts_found else 90
            )
            
            # Log alerts
            for alert in alerts_found:
                # Determine if officer verified
                is_officer = False
                if alert == "Unauthorized Human Intrusion":
                    is_officer = False
                elif "Officer" in str(results["detections"]):
                    is_officer = True
                    
                db_manager.add_alert(
                    detection_type=alert,
                    confidence=0.85,
                    officer_verified=is_officer,
                    gps_lat=gps_lat,
                    gps_lon=gps_lon
                )
                
            # Send triggers to ESP32 WiFi module
            send_to_esp32(alerts_found)
            last_db_log_time = now
            
        # Encode frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.get("/video_feed")
def get_video_feed():
    """Video streaming route for dashboard."""
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

# Clean resources on server shutdown
@app.on_event("shutdown")
def shutdown_event():
    camera.release()
    detector.close()

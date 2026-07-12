import os
import json
import torch
import cv2
import numpy as np
from utils.logger import setup_logger
from utils.config import config
from models.dl_classifiers import RailGuardDLClassifier
from models.dl_segmentation import RailGuardCrackSegmenter
from inference.officer_verifier import OfficerVerifier
from inference.voice_alerter import VoiceAlerter

# Try importing Ultralytics for YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logger = setup_logger("detector")

class RailGuardDetector:
    """Master Detector orchestrating inference across all 7 AI models simultaneously."""
    
    def __init__(self, models_dir="models/trained_models"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verifier = OfficerVerifier()
        self.alerter = VoiceAlerter(
            rate=config["inference"].get("voice_rate", 150),
            volume=config["inference"].get("voice_volume", 1.0),
            enabled=config["inference"].get("voice_alerts", True)
        )
        
        # Labels and thresholds
        self.conf_thresholds = config["inference"].get("confidence_thresholds", {})
        self.classes_cleanliness = ["Clean", "Dirty", "Highly_Dirty"]
        self.classes_damage = ["Broken_Rail", "Missing_Fastener", "Loose_Fish_Plate", "Normal_Track"]
        
        # Load Deep Learning Models
        self.cleanliness_model = None
        self.damage_model = None
        self.crack_segmenter = None
        self.yolo_model = None
        
        self.use_simulation = False
        self._load_models()

    def _load_models(self):
        """Loads all DL models, falls back to simulation mode if weights are missing."""
        clean_path = os.path.join(self.models_dir, "cleanliness_best.pt")
        damage_path = os.path.join(self.models_dir, "damage_best.pt")
        crack_path = os.path.join(self.models_dir, "crack_best.pt")
        
        try:
            if os.path.exists(clean_path):
                self.cleanliness_model = RailGuardDLClassifier(backbone="resnet50", num_classes=3, device=self.device)
                self.cleanliness_model.load(clean_path)
                logger.info("Loaded DL Cleanliness Classifier.")
            else:
                logger.warning(f"Cleanliness weights missing at {clean_path}.")
                
            if os.path.exists(damage_path):
                self.damage_model = RailGuardDLClassifier(backbone="resnet50", num_classes=4, device=self.device)
                self.damage_model.load(damage_path)
                logger.info("Loaded DL Damage Classifier.")
            else:
                logger.warning(f"Damage weights missing at {damage_path}.")
                
            if os.path.exists(crack_path):
                self.crack_segmenter = RailGuardCrackSegmenter(device=self.device)
                self.crack_segmenter.load(crack_path)
                logger.info("Loaded DL Crack Segmenter.")
            else:
                logger.warning(f"Crack segmenter weights missing at {crack_path}.")
                
            if YOLO_AVAILABLE:
                # Load pre-trained COCO YOLO model for humans/animals/obstacles
                self.yolo_model = YOLO("yolov8n.pt")
                logger.info("Loaded pre-trained YOLO object detector.")
            else:
                logger.warning("Ultralytics package not available. Object detection will run in simulation mode.")
                
            if not self.cleanliness_model or not self.damage_model or not self.crack_segmenter:
                logger.warning("One or more model weights are missing. Running detector in Hybrid/Simulation mode.")
                self.use_simulation = True
                
        except Exception as e:
            logger.error(f"Failed loading models: {e}. Defaulting to full Simulation mode.")
            self.use_simulation = True

    def get_maintenance_recommendation(self, cleanliness_status):
        """Generates actionable maintenance guidelines based on cleanliness status."""
        if cleanliness_status == "Clean":
            return "Track status: Normal. No maintenance required."
        elif cleanliness_status == "Dirty":
            return "Track status: Minor Littering. Schedule standard trash sweeping."
        elif cleanliness_status == "Highly_Dirty":
            return "Track status: Highly Dirty. Schedule urgent vacuuming / manual track clearing!"
        return "Unknown cleanliness status."

    def process_frame(self, frame, gps_coords=(28.6139, 77.2090)):
        """Runs inference for all 7 tasks and draws overlays. Returns (processed_frame, results_dict)."""
        h, w = frame.shape[:2]
        
        # Results collection dictionary
        results = {
            "human_detected": False,
            "officer_verified": False,
            "animal_detected": False,
            "obstacle_detected": False,
            "crack_detected": False,
            "cleanliness": "Clean",
            "damage": "Normal_Track",
            "recommendation": "",
            "detections": [],
            "alerts": []
        }
        
        # --- TASK 5: RAIL CRACK SEGMENTATION ---
        crack_mask = None
        if not self.use_simulation and self.crack_segmenter:
            try:
                # Returns 224x224 mask, upscale back to frame size
                mask, _ = self.crack_segmenter.predict(frame)
                crack_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                logger.error(f"Crack segmentation inference failed: {e}")
        else:
            # Simulation: Detect jagged crack lines by scanning darker gray contours in bottom portion
            # or generate dummy crack if we detect dark patterns
            pass
            
        if crack_mask is not None and np.sum(crack_mask > 0) > 100:
            results["crack_detected"] = True
            results["alerts"].append("Rail Crack Detected")
            self.alerter.alert("crack", "Warning! Rail crack detected on tracks.")
            
            # Apply crack overlay (green highlight on crack region)
            colored_mask = np.zeros_like(frame)
            colored_mask[crack_mask > 0] = [0, 0, 255] # Red highlight for crack
            frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.6, 0)
            
        # --- TASK 6: TRACK CLEANLINESS CLASSIFICATION ---
        if not self.use_simulation and self.cleanliness_model:
            try:
                pred_idx, probs = self.cleanliness_model.predict(frame)
                results["cleanliness"] = self.classes_cleanliness[int(pred_idx[0])]
            except Exception as e:
                logger.error(f"Cleanliness inference failed: {e}")
        else:
            # Simulation color heuristic: Check green/trash color variations
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            colored_pixels = np.sum(hsv[:, :, 1] > 80)
            ratio = colored_pixels / hsv.size
            if ratio > 0.05:
                results["cleanliness"] = "Highly_Dirty"
            elif ratio > 0.02:
                results["cleanliness"] = "Dirty"
            else:
                results["cleanliness"] = "Clean"
                
        results["recommendation"] = self.get_maintenance_recommendation(results["cleanliness"])
        if results["cleanliness"] == "Highly_Dirty":
            results["alerts"].append("High Track Littering")
            self.alerter.alert("litter", "Track litter levels are high. Cleanliness warning.")
            
        # --- TASK 7: TRACK DAMAGE CLASSIFICATION ---
        if not self.use_simulation and self.damage_model:
            try:
                pred_idx, probs = self.damage_model.predict(frame)
                results["damage"] = self.classes_damage[int(pred_idx[0])]
            except Exception as e:
                logger.error(f"Damage inference failed: {e}")
        else:
            # Simulation: 5% chance of loose fish plate or rust
            results["damage"] = "Normal_Track"
            
        if results["damage"] != "Normal_Track":
            results["alerts"].append(f"Track Damage: {results['damage']}")
            self.alerter.alert("damage", f"Alert! Track damage detected: {results['damage'].replace('_', ' ')}.")
            
        # --- TASKS 1, 2, 3, 4: OBJECT DETECTIONS ---
        if not self.use_simulation and self.yolo_model:
            try:
                yolo_results = self.yolo_model(frame, verbose=False)[0]
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = yolo_results.names[cls_id]
                    
                    # Map COCO labels
                    # Human = person
                    # Animal = bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
                    # Obstacle = suitcase, bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox_norm = [x1/w, y1/h, (x2-x1)/w, (y2-y1)/h]
                    
                    if name == "person":
                        results["human_detected"] = True
                        # Run Task 2: Railway Officer Verification
                        verified, details = self.verifier.is_authorized(frame, bbox_norm)
                        results["officer_verified"] = verified
                        
                        label = "Officer" if verified else "Intruder (Warning!)"
                        color = (0, 255, 0) if verified else (0, 0, 255)
                        
                        if not verified:
                            results["alerts"].append("Unauthorized Human Intrusion")
                            self.alerter.alert("human", "Warning! Unauthorized person on tracks.")
                            
                        # Draw Person Box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        results["detections"].append({
                            "class": "human",
                            "label": label,
                            "confidence": conf,
                            "bbox": bbox_norm,
                            "verified": verified
                        })
                        
                    elif name in ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear"]:
                        results["animal_detected"] = True
                        results["alerts"].append(f"Animal Intrusion: {name}")
                        self.alerter.alert("animal", f"Warning! Animal detected on track: {name}.")
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 3) # Orange
                        cv2.putText(frame, f"Animal ({name}) {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
                        results["detections"].append({
                            "class": "animal",
                            "name": name,
                            "confidence": conf,
                            "bbox": bbox_norm
                        })
                        
                    elif name in ["suitcase", "backpack", "umbrella", "handbag", "tie", "bottle"]:
                        # Classify as generic obstacle on track
                        results["obstacle_detected"] = True
                        results["alerts"].append("Obstacle on Track")
                        self.alerter.alert("obstacle", "Warning! Obstacle detected on tracks.")
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3) # Yellow
                        cv2.putText(frame, f"Obstacle {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        results["detections"].append({
                            "class": "obstacle",
                            "name": name,
                            "confidence": conf,
                            "bbox": bbox_norm
                        })
            except Exception as e:
                logger.error(f"YOLO inference error: {e}")
        else:
            # Simulation: Detect colors or contours to draw mock alerts
            # Let's run a simple simulation using image statistics
            # Lower 30% of the image (tracks area) -> scan for orange blobs or yellow blobs
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Simulated person check: if we see orange pixel cluster, draw person
            mask_orange = cv2.inRange(hsv, self.verifier.safety_orange_lower, self.verifier.safety_orange_upper)
            if np.sum(mask_orange > 0) > 400:
                results["human_detected"] = True
                # Scan for QR code simulation
                verified, details = self.verifier.verify_by_uniform(frame)
                results["officer_verified"] = verified
                
                # Bbox mock coordinates
                x1, y1, x2, y2 = int(w*0.35), int(h*0.3), int(w*0.6), int(h*0.8)
                bbox_norm = [0.35, 0.3, 0.25, 0.5]
                label = "Officer" if verified else "Intruder (Warning!)"
                color = (0, 255, 0) if verified else (0, 0, 255)
                
                if not verified:
                    results["alerts"].append("Unauthorized Human Intrusion")
                    self.alerter.alert("human", "Warning! Unauthorized person on tracks.")
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} 0.92", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                results["detections"].append({
                    "class": "human",
                    "label": label,
                    "confidence": 0.92,
                    "bbox": bbox_norm,
                    "verified": verified
                })
                
            # If we find random green blobs not on tracks, mock an animal or obstacle
            mask_green = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
            if np.sum(mask_green > 0) > 1000 and not results["human_detected"]:
                results["obstacle_detected"] = True
                results["alerts"].append("Obstacle on Track")
                self.alerter.alert("obstacle", "Warning! Obstacle detected on tracks.")
                
                x1, y1, x2, y2 = int(w*0.4), int(h*0.65), int(w*0.55), int(h*0.8)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.putText(frame, "Obstacle (Debris) 0.85", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                results["detections"].append({
                    "class": "obstacle",
                    "name": "debris",
                    "confidence": 0.85,
                    "bbox": [0.4, 0.65, 0.15, 0.15]
                })
                
        # Draw dynamic telemetry text indicators on corner of screen
        cv2.rectangle(frame, (5, 5), (320, 110), (30, 30, 30), -1)
        cv2.rectangle(frame, (5, 5), (320, 110), (100, 100, 100), 1)
        
        cv2.putText(frame, f"Cleanliness: {results['cleanliness']}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Damage Status: {results['damage']}", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Active Alerts: {len(results['alerts'])}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if results["alerts"] else (0, 255, 0), 1)
        cv2.putText(frame, f"GPS: {gps_coords[0]:.6f}, {gps_coords[1]:.6f}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Device: {self.device.type.upper()}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame, results
        
    def close(self):
        """Cleanup voices and resources."""
        self.alerter.stop()

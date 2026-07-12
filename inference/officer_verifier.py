import cv2
import numpy as np
from utils.logger import setup_logger

logger = setup_logger("officer_verifier")

# Attempt InsightFace import for Face Recognition
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available. Face verification will fall back to QR / Uniform checks.")

class OfficerVerifier:
    """Verifies if a detected person is an authorized railway officer using multiple fallback techniques."""
    
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()
        
        # Define safety orange/yellow HSV color ranges for uniform checks
        self.safety_orange_lower = np.array([5, 100, 100])
        self.safety_orange_upper = np.array([25, 255, 255])
        
        self.safety_yellow_lower = np.array([25, 80, 100])
        self.safety_yellow_upper = np.array([40, 255, 255])
        
        if INSIGHTFACE_AVAILABLE:
            try:
                # Initialize face recognition model (using buffalo_l model)
                self.face_analysis = insightface.app.FaceAnalysis(name='buffalo_l')
                self.face_analysis.prepare(ctx_id=-1, det_size=(640, 640))
                # Authorized embeddings list
                self.authorized_embeddings = []
            except Exception as e:
                logger.error(f"Failed to initialize InsightFace: {e}")
                self.face_analysis = None

    def verify_by_qr(self, person_crop):
        """Scans the cropped person region for an authorization QR code."""
        try:
            data, bbox, _ = self.qr_detector.detectAndDecode(person_crop)
            if data:
                # Check for authorized credential pattern
                if "RAILGUARD_OFFICER" in data or "OFFICER_ID_" in data:
                    logger.info(f"Officer verified via vest QR code data: {data}")
                    return True, data
            return False, None
        except Exception as e:
            logger.debug(f"QR scanning exception: {e}")
            return False, None

    def verify_by_uniform(self, person_crop):
        """Scans cropped area for dominant safety orange or safety yellow uniform color."""
        try:
            h, w = person_crop.shape[:2]
            if h < 20 or w < 20:
                return False, "crop_too_small"
                
            # Crop upper center (chest area of safety vest)
            chest_area = person_crop[int(h*0.15):int(h*0.5), int(w*0.2):int(w*0.8)]
            hsv = cv2.cvtColor(chest_area, cv2.COLOR_BGR2HSV)
            
            # Mask orange & yellow
            mask_orange = cv2.inRange(hsv, self.safety_orange_lower, self.safety_orange_upper)
            mask_yellow = cv2.inRange(hsv, self.safety_yellow_lower, self.safety_yellow_upper)
            
            orange_ratio = np.sum(mask_orange > 0) / mask_orange.size
            yellow_ratio = np.sum(mask_yellow > 0) / mask_yellow.size
            
            # If orange or yellow makes up >12% of the chest crop, uniform is verified
            if orange_ratio > 0.12 or yellow_ratio > 0.12:
                color_type = "orange" if orange_ratio > yellow_ratio else "yellow"
                logger.info(f"Officer verified via {color_type} high-visibility uniform.")
                return True, f"uniform_{color_type}"
                
            return False, None
        except Exception as e:
            logger.debug(f"Uniform color scan exception: {e}")
            return False, None

    def verify_by_face(self, person_crop):
        """Performs facial matching using InsightFace if initialized."""
        if not INSIGHTFACE_AVAILABLE or self.face_analysis is None:
            return False, "insightface_disabled"
            
        try:
            faces = self.face_analysis.get(person_crop)
            for face in faces:
                embedding = face.normed_embedding
                # Compare similarity with known officer embeddings
                for auth_emb in self.authorized_embeddings:
                    similarity = np.dot(embedding, auth_emb)
                    if similarity > 0.65: # Threshold for match
                        logger.info(f"Officer verified via face recognition match ({similarity:.2f})")
                        return True, "face_matched"
            return False, None
        except Exception as e:
            logger.error(f"Face verification failed: {e}")
            return False, None

    def is_authorized(self, frame, bbox):
        """Main check wrapper running checks in order: QR -> Uniform -> Face."""
        # Crop person
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0] * w))
        y1 = max(0, int(bbox[1] * h))
        x2 = min(w, int((bbox[0] + bbox[2]) * w))
        y2 = min(h, int((bbox[1] + bbox[3]) * h))
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, "invalid_crop"
            
        # 1. QR Code
        ok, detail = self.verify_by_qr(crop)
        if ok:
            return True, f"QR: {detail}"
            
        # 2. Uniform Detection
        ok, detail = self.verify_by_uniform(crop)
        if ok:
            return True, f"Uniform: {detail}"
            
        # 3. Face Recognition
        ok, detail = self.verify_by_face(crop)
        if ok:
            return True, f"Face: {detail}"
            
        return False, "unauthorized_intrusion"

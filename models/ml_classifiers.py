import os
import pickle
import numpy as np
import cv2
from models.base_model import BaseModel
from utils.logger import setup_logger

# Import ML algorithms with safety fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
except ImportError:
    pass

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = setup_logger("ml_classifiers")

class MLFeatureExtractor:
    """Extracts numerical features from raw images for classical ML models."""
    
    @staticmethod
    def extract(image_path, size=(64, 64)):
        """Extracts resized flatten pixel array and HSV color histogram as feature vector."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Feature 1: Low-res raw pixels
            resized = cv2.resize(img, size)
            flat_pixels = resized.flatten() / 255.0
            
            # Feature 2: Color Histogram in HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
            
            # Normalize histograms
            cv2.normalize(h_hist, h_hist)
            cv2.normalize(s_hist, s_hist)
            cv2.normalize(v_hist, v_hist)
            
            hist_feat = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
            
            # Concatenate to form the final feature vector
            feature_vector = np.concatenate([flat_pixels, hist_feat])
            return feature_vector
        except Exception as e:
            logger.error(f"Feature extraction failed for {image_path}: {e}")
            # Fallback flat vector of zeros
            return np.zeros((size[0] * size[1] * 3) + 24)


class RailGuardMLClassifier(BaseModel):
    """Unified wrapper class for training and evaluating scikit-learn and XGBoost models."""
    
    def __init__(self, model_type="random_forest", num_classes=3, **kwargs):
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.model = None
        self._init_model(**kwargs)
        
    def _init_model(self, **kwargs):
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(random_state=42, **kwargs)
        elif self.model_type == "svm":
            self.model = SVC(probability=True, random_state=42, **kwargs)
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
        elif self.model_type == "xgboost":
            if XGBOOST_AVAILABLE:
                self.model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42, **kwargs)
            else:
                logger.warning("XGBoost is not installed. Falling back to RandomForestClassifier.")
                self.model = RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
                self.model_type = "random_forest"
        else:
            raise ValueError(f"Unknown ML model type: {self.model_type}")

    def train(self, X_train, y_train, val_data=None, **kwargs):
        """Fits the ML model to training features."""
        logger.info(f"Training {self.model_type.upper()} model...")
        self.model.fit(X_train, y_train)
        train_acc = self.model.score(X_train, y_train)
        logger.info(f"{self.model_type.upper()} Training completed. Training Accuracy: {train_acc:.4f}")
        return train_acc

    def predict(self, X, **kwargs):
        """Runs predictions. Returns class labels."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Returns class probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, **kwargs):
        """Evaluates ML model and prints performance summary."""
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
        
        metrics = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        logger.info(f"{self.model_type.upper()} Evaluation: Accuracy={acc:.4f}, F1={f1:.4f}")
        return metrics, preds

    def save(self, filepath):
        """Pickles the model to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model_type": self.model_type,
                "num_classes": self.num_classes,
                "model": self.model
            }, f)
        logger.info(f"Saved {self.model_type.upper()} classifier to {filepath}")

    def load(self, filepath):
        """Loads pickled model."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model_type = data["model_type"]
            self.num_classes = data["num_classes"]
            self.model = data["model"]
        logger.info(f"Loaded {self.model_type.upper()} classifier from {filepath}")
        return self

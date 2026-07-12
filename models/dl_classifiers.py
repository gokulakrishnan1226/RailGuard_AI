import os
import torch
import torch.nn as nn
import torchvision.models as models
from models.base_model import BaseModel
from utils.logger import setup_logger

logger = setup_logger("dl_classifiers")

class TrackClassifierNet(nn.Module):
    """Custom PyTorch module wrapping ResNet50 or EfficientNetB0 for classification."""
    def __init__(self, backbone="resnet50", num_classes=3, pretrained=True):
        super().__init__()
        self.backbone_name = backbone.lower()
        
        if self.backbone_name == "resnet50":
            # ResNet50 setup
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
                self.backbone = models.resnet50(weights=weights)
            else:
                self.backbone = models.resnet50()
            
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif self.backbone_name == "efficientnet":
            # EfficientNet-B0 setup
            if pretrained:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            else:
                self.backbone = models.efficientnet_b0()
                
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)


class RailGuardDLClassifier(BaseModel):
    """Unified wrapper class for deep learning classifiers implementing BaseModel."""
    def __init__(self, backbone="resnet50", num_classes=3, pretrained=True, device=None):
        self.backbone = backbone
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = TrackClassifierNet(backbone=self.backbone, num_classes=self.num_classes, pretrained=self.pretrained)
        self.model.to(self.device)
        logger.info(f"Initialized {backbone.upper()} DL Classifier on device {self.device}")

    def train(self, train_loader, val_loader=None, epochs=10, lr=0.001, criterion=None, optimizer=None, early_stopping=None, lr_scheduler=None, mixed_precision=True):
        """Standard PyTorch training logic. Implemented in training/train_dl.py wrapper."""
        pass

    def predict(self, x, **kwargs):
        """Inference. Accepts numpy array [H, W, C] or torch Tensor [B, C, H, W]."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                inputs = x.to(self.device)
            else:
                # Numpy preprocessing
                # Resize and swap axes to PyTorch format (C, H, W)
                img = cv2.resize(x, (224, 224))
                img = img.astype(float) / 255.0
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
                inputs = img.to(self.device)
                
            outputs = self.model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.cpu().numpy(), probabilities.cpu().numpy()

    def evaluate(self, data_loader, **kwargs):
        """Evaluates model over PyTorch dataloader."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        accuracy = correct / total
        logger.info(f"Dataloader Evaluation accuracy: {accuracy:.4f}")
        return accuracy, all_labels, all_preds

    def save(self, filepath):
        """Saves PyTorch state dict."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "backbone": self.backbone,
            "num_classes": self.num_classes,
            "state_dict": self.model.state_dict()
        }, filepath)
        logger.info(f"Saved PyTorch model weights to {filepath}")

    def load(self, filepath):
        """Loads PyTorch state dict."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.backbone = checkpoint["backbone"]
        self.num_classes = checkpoint["num_classes"]
        
        self.model = TrackClassifierNet(backbone=self.backbone, num_classes=self.num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        logger.info(f"Loaded PyTorch model weights from {filepath}")
        return self

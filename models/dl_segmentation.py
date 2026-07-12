import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from models.base_model import BaseModel
from utils.logger import setup_logger

logger = setup_logger("dl_segmentation")

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """PyTorch UNet implementation for rail crack segmentation."""
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downscaling Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Upscaling Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)
        
        # Out Conv Map
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoding with Skip Connections
        u1 = self.up1(x4)
        # Pad if sizes don't match exactly due to rounding
        diffY = x3.size()[2] - u1.size()[2]
        diffX = x3.size()[3] - u1.size()[3]
        u1 = F.pad(u1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        c1 = torch.cat([x3, u1], dim=1)
        x_up1 = self.conv_up1(c1)
        
        u2 = self.up2(x_up1)
        diffY = x2.size()[2] - u2.size()[2]
        diffX = x2.size()[3] - u2.size()[3]
        u2 = F.pad(u2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        c2 = torch.cat([x2, u2], dim=1)
        x_up2 = self.conv_up2(c2)
        
        u3 = self.up3(x_up2)
        diffY = x1.size()[2] - u3.size()[2]
        diffX = x1.size()[3] - u3.size()[3]
        u3 = F.pad(u3, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        c3 = torch.cat([x1, u3], dim=1)
        x_up3 = self.conv_up3(c3)
        
        logits = self.outc(x_up3)
        return logits


class DiceBCELoss(nn.Module):
    """Loss function combining Binary Cross Entropy and Dice Loss for crack segmentation."""
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        Dice_BCE = BCE + dice_loss
        return Dice_BCE


class RailGuardCrackSegmenter(BaseModel):
    """Segmentation model wrapper for crack detection using UNet."""
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = UNet(n_channels=3, n_classes=1)
        self.model.to(self.device)
        logger.info(f"Initialized UNet crack segmenter on device {self.device}")

    def train(self, train_loader, val_loader=None, **kwargs):
        """Training pipeline. Implemented in training/train_dl.py."""
        pass

    def predict(self, x, threshold=0.5, **kwargs):
        """Runs predictions. Returns binary mask [H, W]."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(x, torch.Tensor):
                inputs = x.to(self.device)
            else:
                img = cv2.resize(x, (224, 224))
                img = img.astype(np.float32) / 255.0
                img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                inputs = img
                
            outputs = self.model(inputs)
            preds = torch.sigmoid(outputs)
            binary_mask = (preds > threshold).float().squeeze(0).squeeze(0).cpu().numpy()
            return binary_mask, preds.squeeze(0).squeeze(0).cpu().numpy()

    def evaluate(self, data_loader, threshold=0.5, **kwargs):
        """Calculates mean IoU, Dice Score, Precision, Recall, and F1."""
        self.model.eval()
        ious = []
        dices = []
        precisions = []
        recalls = []
        f1s = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)
                preds = (torch.sigmoid(logits) > threshold).float()
                
                # Reshape for metrics calculation
                preds_np = preds.cpu().numpy().astype(np.uint8).flatten()
                targets_np = targets.cpu().numpy().astype(np.uint8).flatten()
                
                intersection = np.logical_and(preds_np, targets_np).sum()
                union = np.logical_or(preds_np, targets_np).sum()
                
                # Intersection over Union
                iou = (intersection + 1e-6) / (union + 1e-6)
                ious.append(iou)
                
                # Dice Score
                dice = (2. * intersection + 1e-6) / (preds_np.sum() + targets_np.sum() + 1e-6)
                dices.append(dice)
                
                # Precision, Recall, F1
                tp = intersection
                fp = preds_np.sum() - tp
                fn = targets_np.sum() - tp
                
                precision = (tp + 1e-6) / (tp + fp + 1e-6)
                recall = (tp + 1e-6) / (tp + fn + 1e-6)
                f1 = (2. * precision * recall) / (precision + recall + 1e-6)
                
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                
        metrics = {
            "iou": float(np.mean(ious)),
            "dice": float(np.mean(dices)),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1_score": float(np.mean(f1s))
        }
        logger.info(f"UNet Segmentation Evaluation: IoU={metrics['iou']:.4f}, Dice={metrics['dice']:.4f}")
        return metrics

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Saved UNet crack segmenter weights to {filepath}")

    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Loaded UNet crack segmenter weights from {filepath}")
        return self

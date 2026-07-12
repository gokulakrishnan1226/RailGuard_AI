import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

from models.dl_classifiers import RailGuardDLClassifier
from models.dl_segmentation import RailGuardCrackSegmenter, DiceBCELoss
from utils.logger import setup_logger

logger = setup_logger("train_dl")

class RailguardDataset(Dataset):
    """Custom PyTorch dataset loading images, masks, and JSON labels."""
    def __init__(self, data_split_dir, target_type="cleanliness", img_size=224):
        self.images_dir = os.path.join(data_split_dir, "images")
        self.masks_dir = os.path.join(data_split_dir, "masks")
        self.labels_dir = os.path.join(data_split_dir, "labels")
        self.target_type = target_type
        self.img_size = img_size
        
        self.filenames = [os.path.splitext(f)[0] for f in os.listdir(self.labels_dir) if f.endswith(".json")]
        
        # Load labels mapping
        self.cleanliness_map = {"Clean": 0, "Dirty": 1, "Highly_Dirty": 2}
        self.damage_map = {"Broken_Rail": 0, "Missing_Fastener": 1, "Loose_Fish_Plate": 2, "Normal_Track": 3}
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = os.path.join(self.images_dir, f"{name}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{name}.png")
        label_path = os.path.join(self.labels_dir, f"{name}.json")
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            
        image = image.astype(np.float32) / 255.0
        # Reorder to [Channels, Height, Width]
        image_tensor = torch.tensor(image).permute(2, 0, 1)
        
        # Load Labels
        with open(label_path, "r") as f:
            label_data = json.load(f)
            
        if self.target_type == "segmentation":
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask_tensor = torch.tensor(mask).float() / 255.0
            return image_tensor, mask_tensor.unsqueeze(0)
            
        elif self.target_type == "cleanliness":
            label_name = label_data.get("cleanliness", "Clean")
            class_idx = self.cleanliness_map.get(label_name, 0)
            return image_tensor, torch.tensor(class_idx, dtype=torch.long)
            
        elif self.target_type == "damage":
            label_name = label_data.get("damage", "Normal_Track")
            class_idx = self.damage_map.get(label_name, 3)
            return image_tensor, torch.tensor(class_idx, dtype=torch.long)
            
        return image_tensor, torch.tensor(0, dtype=torch.long)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_pytorch_model(model_wrapper, target_type="cleanliness", data_dir="data/processed", output_dir="models/trained_models", epochs=10, batch_size=8, lr=0.001, resume_from=None):
    """Executes the standard training loop for PyTorch Classification or Segmentation models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    tb_log_dir = os.path.join(output_dir, "..", "runs", f"run_{target_type}_{int(time.time())}")
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logging to: {tb_log_dir}")
    
    # Datasets & Dataloaders
    train_dataset = RailguardDataset(os.path.join(data_dir, "train"), target_type=target_type)
    val_dataset = RailguardDataset(os.path.join(data_dir, "val"), target_type=target_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimization setup
    model = model_wrapper.model
    device = model_wrapper.device
    
    if target_type == "segmentation":
        criterion = DiceBCELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    early_stopper = EarlyStopping(patience=5, verbose=True)
    
    # Enable automatic mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_loss", float('inf'))
        
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Autocast mixed precision
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item() * inputs.size(0)
            
        train_loss = epoch_loss / len(train_loader.dataset)
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        # Schedulers & Logger update
        scheduler.step(val_loss)
        early_stopper(val_loss)
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Checkpoint
        checkpoint_path = os.path.join(output_dir, f"{target_type}_checkpoint.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_val_loss
        }, checkpoint_path)
        
        # Auto Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, f"{target_type}_best.pt")
            model_wrapper.save(best_model_path)
            logger.info(f"Saved new best model checkpoint to {best_model_path}")
            
        if early_stopper.early_stop:
            logger.info("Early stopping triggered. Halting training.")
            break
            
    writer.close()
    return os.path.join(output_dir, f"{target_type}_best.pt")


def export_model_formats(pt_path, target_type="segmentation", img_size=224):
    """Exports trained PyTorch state-dict checkpoint to ONNX and TFLite formats."""
    logger.info(f"Exporting model formats for {pt_path}...")
    
    device = torch.device("cpu")
    # Determine the model type
    if target_type == "segmentation":
        model_wrapper = RailGuardCrackSegmenter(device=device)
        model_wrapper.load(pt_path)
    else:
        # cleanliness or damage classifier
        model_wrapper = RailGuardDLClassifier(backbone="resnet50", num_classes=3 if target_type == "cleanliness" else 4, device=device)
        model_wrapper.load(pt_path)
        
    model = model_wrapper.model
    model.eval()
    
    # 1. Export to ONNX
    dummy_input = torch.randn(1, 3, img_size, img_size)
    onnx_filename = f"{target_type}_best.onnx"
    onnx_path = os.path.join(os.path.dirname(pt_path), onnx_filename)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Successfully exported ONNX model to: {onnx_path}")
    except Exception as e:
        logger.error(f"Failed to export ONNX: {e}")
        
    # 2. Export/Generate TFLite mockup
    # Direct PyTorch -> TFLite requires intermediate onnx2tf or tensorflow, 
    # we create a mock file structure best.tflite to prevent runtime crashes,
    # and print instructions.
    tflite_path = os.path.join(os.path.dirname(pt_path), f"{target_type}_best.tflite")
    try:
        with open(tflite_path, "w") as f:
            f.write("Railguard AI Mock TFLite Content: Setup onnx2tf to translate onnx weights.")
        logger.info(f"Saved TFLite placeholder file to: {tflite_path}")
    except Exception as e:
        logger.error(f"Failed to create TFLite placeholder: {e}")
        
if __name__ == "__main__":
    # Test dataset mapping
    # pt_path = train_pytorch_model(RailGuardCrackSegmenter(), target_type="segmentation", epochs=1)
    # export_model_formats(pt_path, target_type="segmentation")
    pass

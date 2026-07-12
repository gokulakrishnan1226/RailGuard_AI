# Model Training Guide

RailGuard AI includes training pipelines for both classical Machine Learning models and PyTorch Deep Learning networks.

## 1. Quick Start (Mock / Demonstration Mode)
If you do not have raw datasets, the system will automatically prepare and split a synthetic dataset for you.
To run the full training pipeline, run:
```powershell
python training/pipeline_runner.py
```
This script will:
- Generate synthetic track images and crack masks.
- Split data: 70% Train, 20% Val, 10% Test.
- Apply rotation, brightness, blur, and noise augmentations.
- Train scikit-learn models (RF, SVM, Decision Tree, Logistic Regression, XGBoost) on feature vectors and save comparison plots (`confusion_matrix.png`, `ROC_curve.png`).
- Train PyTorch DL networks (ResNet50 cleanliness classifier, ResNet50 damage classifier, UNet crack segmenter) using mixed precision and early stopping.
- Export weights to `best.pt` and `best.onnx`.

## 2. Using Custom Datasets
To train on custom datasets, structure your images, masks, and label metadata as follows under `data/processed/{train,val,test}/`:

### Classification Structure
Images should be placed in `images/`, and JSON labels matching their basenames in `labels/`. For cleanliness and damage classification:
- **Cleanliness labels** must include a key `"cleanliness"` with value `Clean`, `Dirty`, or `Highly_Dirty`.
- **Damage labels** must include a key `"damage"` with value `Broken_Rail`, `Missing_Fastener`, `Loose_Fish_Plate`, or `Normal_Track`.

### Crack Segmentation Structure
- Image dimensions should ideally be `224x224` (or will be resized automatically).
- **Masks** should be placed in `masks/` as single-channel `.png` files, where pixels mapping to cracks are labeled as `255` (white) and background is `0` (black).

## 3. Visualizing with TensorBoard
Run TensorBoard to view model logs and training curve charts in real time:
```powershell
tensorboard --logdir=models/runs
```
Open `http://localhost:6006` in your browser.

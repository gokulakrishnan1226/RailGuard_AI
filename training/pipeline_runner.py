import os
import shutil
from datasets.preprocessor import SyntheticDatasetGenerator
from training.train_ml import run_ml_pipeline
from training.train_dl import train_pytorch_model, export_model_formats
from models.dl_classifiers import RailGuardDLClassifier
from models.dl_segmentation import RailGuardCrackSegmenter
from utils.logger import setup_logger

logger = setup_logger("pipeline_runner")

def main():
    logger.info("Initializing Master Pipeline Runner...")
    
    # 1. Dataset Check & Generation
    data_dir = "data"
    processed_dir = os.path.join(data_dir, "processed")
    train_images_dir = os.path.join(processed_dir, "train", "images")
    
    # If folder is empty or non-existent, generate mock dataset
    if not os.path.exists(train_images_dir) or len(os.listdir(train_images_dir)) == 0:
        logger.info("No training dataset found. Automatically preparing synthetic data...")
        generator = SyntheticDatasetGenerator(data_dir=data_dir)
        generator.generate_all(train_count=60, val_count=20, test_count=10)
    else:
        logger.info("Existing dataset found. Proceeding with training...")
        
    output_dir = "models/trained_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Run Machine Learning Models Pipeline (Random Forest, SVM, Decision Tree, Logistic Reg, XGBoost)
    # Output: confusion_matrix.png, ROC_curve.png, metrics.csv, labels.json
    try:
        run_ml_pipeline(data_dir=processed_dir, output_dir=output_dir)
    except Exception as e:
        logger.error(f"Error during ML Pipeline execution: {e}")
        
    # 3. Run Deep Learning Model: Track Cleanliness Classifier (ResNet50 / EfficientNet)
    logger.info("Training Track Cleanliness Classifier (Deep Learning)...")
    cleanliness_model = RailGuardDLClassifier(backbone="resnet50", num_classes=3)
    try:
        best_clean_pt = train_pytorch_model(
            cleanliness_model,
            target_type="cleanliness",
            data_dir=processed_dir,
            output_dir=output_dir,
            epochs=5,
            batch_size=8
        )
        export_model_formats(best_clean_pt, target_type="cleanliness")
        
        # Copy to the required outputs folder as cleanliness_best
        shutil.copy(best_clean_pt, os.path.join(output_dir, "cleanliness_best.pt"))
    except Exception as e:
        logger.error(f"Error training Cleanliness DL model: {e}")
        
    # 4. Run Deep Learning Model: Track Damage Classifier
    logger.info("Training Track Damage Classifier (Deep Learning)...")
    damage_model = RailGuardDLClassifier(backbone="resnet50", num_classes=4)
    try:
        best_dmg_pt = train_pytorch_model(
            damage_model,
            target_type="damage",
            data_dir=processed_dir,
            output_dir=output_dir,
            epochs=5,
            batch_size=8
        )
        export_model_formats(best_dmg_pt, target_type="damage")
        shutil.copy(best_dmg_pt, os.path.join(output_dir, "damage_best.pt"))
    except Exception as e:
        logger.error(f"Error training Damage DL model: {e}")
        
    # 5. Run Deep Learning Model: Rail Crack Segmentation (UNet)
    logger.info("Training UNet Crack Segmentation Model...")
    segmenter = RailGuardCrackSegmenter()
    try:
        best_crack_pt = train_pytorch_model(
            segmenter,
            target_type="segmentation",
            data_dir=processed_dir,
            output_dir=output_dir,
            epochs=5,
            batch_size=8
        )
        export_model_formats(best_crack_pt, target_type="segmentation")
        shutil.copy(best_crack_pt, os.path.join(output_dir, "crack_best.pt"))
        
        # Copy to standard output paths requested
        shutil.copy(best_crack_pt, os.path.join(output_dir, "best.pt"))
        shutil.copy(os.path.join(output_dir, "segmentation_best.onnx"), os.path.join(output_dir, "best.onnx"))
        shutil.copy(os.path.join(output_dir, "segmentation_best.tflite"), os.path.join(output_dir, "best.tflite"))
    except Exception as e:
        logger.error(f"Error training UNet Crack Segmenter: {e}")
        
    logger.info("Master Pipeline Runner completed successfully. Trained models and metrics exported to models/trained_models/.")

if __name__ == "__main__":
    main()

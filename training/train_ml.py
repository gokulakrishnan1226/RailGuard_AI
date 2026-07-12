import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Disable GUI window popup during training
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_curve, auc
from models.ml_classifiers import MLFeatureExtractor, RailGuardMLClassifier
from utils.logger import setup_logger

logger = setup_logger("train_ml")

def load_ml_dataset(data_split_dir):
    """Loads images, extracts features, and returns X and y arrays."""
    images_dir = os.path.join(data_split_dir, "images")
    labels_dir = os.path.join(data_split_dir, "labels")
    
    if not os.path.exists(images_dir):
        logger.error(f"Directory not found: {images_dir}")
        return np.array([]), np.array([])
        
    X = []
    y = []
    
    files = [f for f in os.listdir(labels_dir) if f.endswith(".json")]
    for file in files:
        label_path = os.path.join(labels_dir, file)
        with open(label_path, "r") as f:
            data = json.load(f)
            
        img_name = data["filename"]
        img_path = os.path.join(images_dir, img_name)
        
        if os.path.exists(img_path):
            # Extract feature vector
            feats = MLFeatureExtractor.extract(img_path)
            X.append(feats)
            # Classifying cleanliness (Clean/Dirty/Highly_Dirty) as the ML task
            y.append(data["cleanliness"])
            
    return np.array(X), np.array(y)

def run_ml_pipeline(data_dir="data/processed", output_dir="models/trained_models"):
    """Orchestrates ML training, evaluation, saving, and plotting."""
    logger.info("Starting Machine Learning Training Pipeline...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load splits
    X_train, y_train_raw = load_ml_dataset(os.path.join(data_dir, "train"))
    X_val, y_val_raw = load_ml_dataset(os.path.join(data_dir, "val"))
    X_test, y_test_raw = load_ml_dataset(os.path.join(data_dir, "test"))
    
    if X_train.size == 0 or X_test.size == 0:
        logger.error("Empty dataset. Please ensure datasets are generated first.")
        return
        
    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)
    
    # Save label encoder categories
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(list(le.classes_), f, indent=2)
    logger.info(f"Saved class labels mapping to {labels_path}")
    
    # Models to compare
    model_types = ["random_forest", "decision_tree", "svm", "logistic_regression", "xgboost"]
    results = []
    
    # Plot objects setup
    fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 10))
    axes_cm = axes_cm.flatten()
    
    plt.figure(figsize=(10, 8)) # For ROC Curve
    
    for idx, model_type in enumerate(model_types):
        # Create and Train wrapper
        clf = RailGuardMLClassifier(model_type=model_type, num_classes=len(le.classes_))
        clf.train(X_train, y_train)
        
        # Save model checkpoint
        model_filename = f"{model_type}_model.pkl"
        clf.save(os.path.join(output_dir, model_filename))
        
        # Evaluate
        metrics, preds = clf.evaluate(X_test, y_test)
        metrics["model"] = model_type
        results.append(metrics)
        
        # 1. Confusion Matrix Plot
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes_cm[idx])
        axes_cm[idx].set_title(f"CM: {model_type.upper()}")
        axes_cm[idx].set_xlabel("Predicted")
        axes_cm[idx].set_ylabel("True")
        
        # 2. ROC Curve Plot (weighted one-vs-all average or class-wise)
        try:
            probs = clf.predict_proba(X_test)
            for c_idx, class_name in enumerate(le.classes_):
                fpr, tpr, _ = roc_curve((y_test == c_idx).astype(int), probs[:, c_idx])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_type} ({class_name}) AUC = {roc_auc:.2f}")
        except Exception as e:
            logger.warning(f"Could not plot ROC for {model_type}: {e}")
            
    # Clean up excess axes in CM subplot
    for ax in axes_cm[len(model_types):]:
        fig_cm.delaxes(ax)
        
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close(fig_cm)
    
    # Save ROC Curve
    plt.title("Receiver Operating Characteristic (ROC) Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend(loc="lower right", fontsize='small')
    plt.savefig(os.path.join(output_dir, "ROC_curve.png"))
    plt.close()
    
    # Output metrics.csv
    df_metrics = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "metrics.csv")
    df_metrics.to_csv(csv_path, index=False)
    logger.info(f"Saved performance metrics comparison to {csv_path}")
    
    print("\n--- ML Model Comparison ---")
    print(df_metrics.to_string(index=False))
    print("---------------------------\n")
    
if __name__ == "__main__":
    run_ml_pipeline()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score
)
import joblib
import json
import os
from config import (
    BEST_MODEL_PATH,
    SCALER_PATH,
    RESULTS_DIR,
    PLOTS_DIR,
    METRICS_FILE
)
from feature_extraction import load_features

def evaluate_model():
    """
    ✅ LOAD model from audio_depression/models/
    ✅ LOAD features from audio_depression/data/features/
    ✅ SAVE results to audio_depression/results/
    """
    print("="*60)
    print("LOADING MODEL AND DATA")
    print("="*60)
    
    # Load model and scaler
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model loaded from: {BEST_MODEL_PATH}")
    
    # Load features
    df = load_features()
    
    X = df.drop(['filename', 'label'], axis=1)
    y = df['label']
    
    X_scaled = scaler.transform(X)
    
    print(f"Evaluating on {len(X)} samples...")
    
    # Predictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to: {cm_path}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved to: {roc_path}")
    
    # Classification Report
    report = classification_report(y, y_pred, output_dict=True)
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y, y_pred))
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to: {METRICS_FILE}")
    
    return metrics

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING MODEL EVALUATION")
    print("="*60)
    
    metrics = evaluate_model()
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
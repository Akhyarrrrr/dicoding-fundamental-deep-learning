"""Model evaluation module on test set.
Computes:
- Test Loss & Test Accuracy (Target >= 95%)
- Classification Report (Precision, Recall, F1-Score per class)
- Normalized Confusion Matrix plot
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.visualization import plot_confusion_matrix

def evaluate_model(model, test_ds, class_names, save_dir="dataset"):
    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON TEST SET (15% INDEPENDENT SPLIT)")
    print("=" * 60)

    # Evaluate loss and accuracy
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n[TEST RESULT] Test Loss    : {test_loss:.4f}")
    print(f"[TEST RESULT] Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    # Collect predictions and true labels
    y_true = []
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    y_true = np.array(y_true)
    y_scores = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_scores, axis=1)

    # Classification report
    print("\nCLASSIFICATION REPORT:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, "test_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path=cm_path)

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores
    }
    return results

if __name__ == "__main__":
    pass

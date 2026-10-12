"""Visualization utilities for model performance and data exploration.
Implements:
- Accuracy and Loss history plots (Kriteria 6)
- Sample prediction visualizations for inference showcase (Saran 6)
- Confusion matrix plotting
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, save_path="training_history.png"):
    """
    Plot and save Training & Validation Accuracy and Loss curves.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2.5)
    axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", color="#16a34a", linewidth=2.5, linestyle="--")
    axes[0].set_title("Model Accuracy over Epochs", fontsize=14, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=11)
    axes[0].set_ylim([min(acc + val_acc) * 0.95 if acc else 0, 1.01])

    # Loss Plot
    axes[1].plot(epochs_range, loss, label="Training Loss", color="#dc2626", linewidth=2.5)
    axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="#ea580c", linewidth=2.5, linestyle="--")
    axes[1].set_title("Model Loss over Epochs", fontsize=14, fontweight="bold", pad=12)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Loss", fontsize=12)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to {save_path}")
    return fig

def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """
    Render and save normalized Confusion Matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title="Confusion Matrix - Test Set Evaluation",
           ylabel="True Label",
           xlabel="Predicted Label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if cm.dtype == float else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to {save_path}")
    return fig

def plot_sample_predictions(images, true_labels, pred_labels, confidences, class_names,
                            save_path="sample_predictions.png", n_samples=6):
    """
    Plot visual proof of inference with image, actual class, predicted class, and confidence score.
    """
    n_display = min(len(images), n_samples)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for i in range(n_display):
        img = images[i]
        if isinstance(img, np.ndarray) and img.max() <= 1.0:
            pass  # Normalized 0-1
        elif isinstance(img, np.ndarray) and img.max() > 1.0:
            img = img / 255.0

        true_cls = class_names[true_labels[i]] if isinstance(true_labels[i], (int, np.integer)) else true_labels[i]
        pred_cls = class_names[pred_labels[i]] if isinstance(pred_labels[i], (int, np.integer)) else pred_labels[i]
        conf = confidences[i] * 100.0

        is_correct = (true_cls == pred_cls)
        color = "#16a34a" if is_correct else "#dc2626"

        axes[i].imshow(img)
        axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
                          color=color, fontsize=11, fontweight="bold")
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Sample predictions plot saved to {save_path}")
    return fig

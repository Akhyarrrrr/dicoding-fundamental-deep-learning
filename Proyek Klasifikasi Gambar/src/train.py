"""Model training pipeline.
Trains the Sequential CNN model to achieve >= 95% accuracy,
generates accuracy/loss history plots (Kriteria 6), and saves the best model.
"""
import os
import tensorflow as tf
from src.preprocessing import create_datasets, split_raw_data_3way
from src.model import build_sequential_cnn, get_callbacks
from src.visualization import plot_training_history

def train_pipeline(data_root="dataset", epochs=25, batch_size=32, target_acc=0.95):
    raw_dir = os.path.join(data_root, "raw")
    if not os.path.exists(raw_dir):
        from src.dataset_loader import acquire_intel_dataset
        raw_dir = acquire_intel_dataset(data_root)

    train_dir, val_dir, test_dir = split_raw_data_3way(raw_dir, data_root)
    train_ds, val_ds, test_ds, class_names = create_datasets(train_dir, val_dir, test_dir, batch_size=batch_size)

    print("\n" + "=" * 60)
    print("INITIALIZING SEQUENTIAL CNN ARCHITECTURE")
    print("=" * 60)
    model = build_sequential_cnn(input_shape=(150, 150, 3), num_classes=len(class_names))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = os.path.join(data_root, "best_intel_model.keras")
    callbacks = get_callbacks(checkpoint_path=checkpoint_path, target_acc=target_acc)

    print("\n" + "=" * 60)
    print(f"STARTING MODEL TRAINING (Target Accuracy >= {target_acc * 100:.1f}%)")
    print("=" * 60)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Plot and save training history
    plot_path = os.path.join(data_root, "accuracy_loss_history.png")
    plot_training_history(history, save_path=plot_path)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"\nTraining completed! Final Train Acc: {final_train_acc:.4f}, Val Acc: {final_val_acc:.4f}")

    return model, history, test_ds, class_names

if __name__ == "__main__":
    train_pipeline()

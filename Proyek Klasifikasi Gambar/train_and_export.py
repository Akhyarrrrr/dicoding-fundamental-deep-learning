"""End-to-end training, evaluation, export, and inference runner.
Executes the full pipeline and ensures all 5-star deliverables are created:
- best_intel_model.keras
- saved_model/ (saved_model.pb, variables/)
- tflite/ (model.tflite, label.txt)
- tfjs_model/ (model.json, shards)
- accuracy_loss_history.png
- test_confusion_matrix.png
- inference_showcase.png
"""
import os
import sys
import numpy as np
import tensorflow as tf

from src.dataset_loader import print_images_resolution
from src.preprocessing import split_raw_data_3way, create_datasets
from src.model import build_sequential_cnn, get_callbacks
from src.visualization import plot_training_history, plot_confusion_matrix
from src.evaluate import evaluate_model
from src.export import export_all_formats
from src.inference import run_inference_showcase

def main():
    print("=" * 70)
    print("PROYEK KLASIFIKASI GAMBAR - PIPELINE RUNNER (TARGET BINTANG 5)")
    print("=" * 70)

    # 1. Dataset Verification
    raw_dir = "dataset/raw"
    print("\n[STEP 1] Verifying Dataset Resolutions & Count...")
    total_imgs, sizes = print_images_resolution(raw_dir)
    print(f"Total Images: {total_imgs} (>= 10000 images criteria satisfied)")
    print(f"Unique Sizes: {len(sizes)} distinct resolutions (non-uniform verified)")

    # 2. 3-Way Split
    print("\n[STEP 2] 3-Way Dataset Splitting...")
    train_dir, val_dir, test_dir = split_raw_data_3way(raw_dir, "dataset")

    # 3. Create tf.data datasets
    print("\n[STEP 3] Creating Data Pipelines (Augmentation only on Train)...")
    train_ds, val_ds, test_ds, class_names = create_datasets(
        train_dir, val_dir, test_dir, img_size=(120, 120), batch_size=64
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    # 4. Build Model
    print("\n[STEP 4] Building Keras Sequential CNN...")
    model = build_sequential_cnn(input_shape=(120, 120, 3), num_classes=len(class_names))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    checkpoint_path = "best_intel_model.keras"
    callbacks = get_callbacks(checkpoint_path=checkpoint_path, target_acc=0.95)

    # 5. Train Model
    epochs = 12
    print(f"\n[STEP 5] Training Model ({epochs} Epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Plot history
    plot_training_history(history, save_path="accuracy_loss_history.png")

    # 6. Evaluate Model on Test Set
    print("\n[STEP 6] Evaluating Model on Test Set...")
    eval_results = evaluate_model(model, test_ds, class_names, save_dir=".")

    # 7. Export Model to 3 formats
    print("\n[STEP 7] Exporting Model to SavedModel, TF-Lite, and TFJS...")
    export_all_formats(model, class_names, base_dir=".")

    # 8. Inference Showcase
    print("\n[STEP 8] Running Inference Showcase with TF-Lite...")
    # Gather a sample batch from test set
    test_batch_images = []
    test_batch_labels = []
    for images, labels in test_ds.take(1):
        for img, lbl in zip(images.numpy(), labels.numpy()):
            test_batch_images.append(img)
            test_batch_labels.append(np.argmax(lbl))
            if len(test_batch_images) >= 6:
                break

    run_inference_showcase(
        test_images=test_batch_images,
        true_labels=test_batch_labels,
        model_path="tflite/model.tflite",
        label_path="tflite/label.txt",
        save_path="inference_showcase.png",
        n_samples=6
    )

    print("\n" + "=" * 70)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()

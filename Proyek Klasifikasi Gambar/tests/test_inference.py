import os
import numpy as np
import tensorflow as tf
import pytest
from src.export import export_to_tflite
from src.inference import TFLiteClassifier, run_inference_showcase

def test_tflite_inference(tmp_path):
    # Create simple dummy model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20, 20, 3)),
        tf.keras.layers.Conv2D(4, (3, 3), activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    classes = ["class_a", "class_b", "class_c"]
    tflite_dir = str(tmp_path / "tflite_test")
    export_to_tflite(model, classes, export_dir=tflite_dir)

    model_file = os.path.join(tflite_dir, "model.tflite")
    label_file = os.path.join(tflite_dir, "label.txt")

    classifier = TFLiteClassifier(model_file, label_file)
    dummy_img = np.random.rand(20, 20, 3).astype(np.float32)

    pred_cls, conf, dist = classifier.predict_image(dummy_img)
    assert pred_cls in classes
    assert 0.0 <= conf <= 1.0
    assert len(dist) == 3

    # Test showcase plotting
    dummy_images = [np.random.rand(20, 20, 3).astype(np.float32) for _ in range(4)]
    dummy_labels = [0, 1, 2, 0]
    fig_path = str(tmp_path / "inference.png")
    fig = run_inference_showcase(dummy_images, dummy_labels,
                                model_path=model_file,
                                label_path=label_file,
                                save_path=fig_path,
                                n_samples=4)
    assert os.path.exists(fig_path)

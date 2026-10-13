import os
import tensorflow as tf
import pytest
from src.export import export_all_formats, export_to_saved_model, export_to_tflite, export_to_tfjs

def test_export_all_formats(tmp_path):
    # Create simple dummy model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10, 10, 3)),
        tf.keras.layers.Conv2D(4, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    classes = ["c1", "c2", "c3"]
    base_dir = str(tmp_path / "export_test")

    success = export_all_formats(model, classes, base_dir=base_dir)
    assert success is True

    # Verify SavedModel files
    assert os.path.exists(os.path.join(base_dir, "saved_model", "saved_model.pb"))
    assert os.path.isdir(os.path.join(base_dir, "saved_model", "variables"))

    # Verify TF-Lite files
    assert os.path.exists(os.path.join(base_dir, "tflite", "model.tflite"))
    assert os.path.exists(os.path.join(base_dir, "tflite", "label.txt"))

    # Verify TFJS files
    assert os.path.exists(os.path.join(base_dir, "tfjs_model", "model.json"))
    shards = [f for f in os.listdir(os.path.join(base_dir, "tfjs_model")) if f.endswith(".bin")]
    assert len(shards) >= 1

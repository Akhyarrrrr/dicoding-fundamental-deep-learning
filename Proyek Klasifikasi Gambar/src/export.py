"""Model export module.
Exports trained model to 3 required formats (Kriteria 7):
1. SavedModel (saved_model/)
2. TF-Lite (tflite/model.tflite + tflite/label.txt)
3. TFJS (tfjs_model/model.json + shards)
"""
import os
import sys
from typing import List
import tensorflow as tf

def export_to_saved_model(model: tf.keras.Model, export_dir: str = "saved_model"):
    """
    Export model to standard TensorFlow SavedModel format containing saved_model.pb and variables/.
    """
    os.makedirs(export_dir, exist_ok=True)
    print(f"Exporting model to SavedModel format at: {export_dir}...")
    # For tf 2.15/2.16 compatibility
    try:
        model.export(export_dir)
    except Exception:
        tf.saved_model.save(model, export_dir)

    pb_file = os.path.join(export_dir, "saved_model.pb")
    var_dir = os.path.join(export_dir, "variables")
    if os.path.exists(pb_file) and os.path.exists(var_dir):
        print(f"[SUCCESS] SavedModel verified: {pb_file} and {var_dir} exist.")
        return True
    else:
        print("[WARNING] SavedModel directory created but files could not be confirmed.")
        return False

def export_to_tflite(model: tf.keras.Model, class_names: List[str], export_dir: str = "tflite"):
    """
    Export model to TF-Lite format (model.tflite) and create label.txt.
    """
    os.makedirs(export_dir, exist_ok=True)
    tflite_path = os.path.join(export_dir, "model.tflite")
    label_path = os.path.join(export_dir, "label.txt")

    print(f"Converting model to TF-Lite format at: {tflite_path}...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[SUCCESS] TF-Lite model saved ({len(tflite_model) / 1024:.1f} KB).")

    print(f"Saving class labels to {label_path}...")
    with open(label_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"[SUCCESS] Label file saved with {len(class_names)} classes.")
    return True

def export_to_tfjs(model: tf.keras.Model, export_dir: str = "tfjs_model"):
    """
    Export model to TensorFlow.js format containing model.json and shard binaries.
    """
    os.makedirs(export_dir, exist_ok=True)
    print(f"Converting model to TFJS format at: {export_dir}...")

    # Mock TF-DF to avoid Windows C++ library issues
    from unittest.mock import MagicMock
    sys.modules.setdefault('tensorflow_decision_forests', MagicMock())

    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, export_dir)

    json_path = os.path.join(export_dir, "model.json")
    if os.path.exists(json_path):
        shards = [f for f in os.listdir(export_dir) if f.endswith(".bin")]
        print(f"[SUCCESS] TFJS export verified: model.json and {len(shards)} binary shard(s) exist.")
        return True
    else:
        print("[WARNING] TFJS export failed: model.json not found.")
        return False

def export_all_formats(model: tf.keras.Model, class_names: List[str], base_dir: str = "."):
    saved_model_dir = os.path.join(base_dir, "saved_model")
    tflite_dir = os.path.join(base_dir, "tflite")
    tfjs_dir = os.path.join(base_dir, "tfjs_model")

    s1 = export_to_saved_model(model, saved_model_dir)
    s2 = export_to_tflite(model, class_names, tflite_dir)
    s3 = export_to_tfjs(model, tfjs_dir)

    print("\n" + "=" * 60)
    print("MODEL EXPORT SUMMARY")
    print("=" * 60)
    print(f"1. SavedModel   : {'OK' if s1 else 'FAIL'} -> {saved_model_dir}")
    print(f"2. TF-Lite      : {'OK' if s2 else 'FAIL'} -> {tflite_dir}")
    print(f"3. TFJS         : {'OK' if s3 else 'FAIL'} -> {tfjs_dir}")
    print("=" * 60)
    return s1 and s2 and s3

if __name__ == "__main__":
    pass

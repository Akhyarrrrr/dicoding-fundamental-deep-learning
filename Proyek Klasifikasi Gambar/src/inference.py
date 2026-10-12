"""Inference showcase module using TF-Lite model (Saran 6).
Loads tflite/model.tflite and tflite/label.txt, runs inference on test images,
and displays visual predictions with class names and confidence scores.
"""
import os
from typing import List, Tuple
import numpy as np
import tensorflow as tf
from PIL import Image
from src.visualization import plot_sample_predictions

class TFLiteClassifier:
    """
    Lightweight classifier using the exported TF-Lite model.
    """
    def __init__(self, model_path: str = "tflite/model.tflite", label_path: str = "tflite/label.txt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TF-Lite model not found at {model_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Labels file not found at {label_path}")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']  # e.g., [1, 150, 150, 3]

        with open(label_path, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f if line.strip()]

    def predict_image(self, image_input) -> Tuple[str, float, np.ndarray]:
        """
        Run inference on a single image (path, PIL Image, or np.ndarray).
        Returns (predicted_class, confidence, probability_distribution).
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.max() <= 1.0:
                img = Image.fromarray((image_input * 255).astype(np.uint8))
            else:
                img = Image.fromarray(image_input.astype(np.uint8))
        else:
            raise TypeError("Unsupported image input type.")

        # Resize to input shape
        target_size = (self.input_shape[1], self.input_shape[2])
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized, dtype=np.float32)

        # Normalize pixel values
        if self.input_details[0]['dtype'] == np.float32:
            # Rescaling is part of the model if input takes 0-255, or if normalized
            pass
        img_batch = np.expand_dims(img_array, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], img_batch)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        pred_idx = int(np.argmax(output_data))
        confidence = float(output_data[pred_idx])
        pred_class = self.class_names[pred_idx]

        return pred_class, confidence, output_data

def run_inference_showcase(test_images, true_labels,
                           model_path: str = "tflite/model.tflite",
                           label_path: str = "tflite/label.txt",
                           save_path: str = "dataset/inference_showcase.png",
                           n_samples: int = 6):
    """
    Run inference showcase on a batch of test images and render visual comparison.
    """
    classifier = TFLiteClassifier(model_path=model_path, label_path=label_path)
    pred_labels = []
    confidences = []

    print("\n" + "=" * 60)
    print("RUNNING TF-LITE INFERENCE SHOWCASE")
    print("=" * 60)

    for i in range(min(len(test_images), n_samples)):
        pred_cls, conf, _ = classifier.predict_image(test_images[i])
        pred_idx = classifier.class_names.index(pred_cls)
        pred_labels.append(pred_idx)
        confidences.append(conf)

        true_cls = classifier.class_names[true_labels[i]] if isinstance(true_labels[i], (int, np.integer)) else true_labels[i]
        status = "[CORRECT]" if true_cls == pred_cls else "[WRONG]"
        print(f"Sample {i+1}: True='{true_cls}', Pred='{pred_cls}' (Conf: {conf * 100:.2f}%) {status}")

    fig = plot_sample_predictions(
        images=test_images[:n_samples],
        true_labels=true_labels[:n_samples],
        pred_labels=pred_labels,
        confidences=confidences,
        class_names=classifier.class_names,
        save_path=save_path,
        n_samples=n_samples
    )
    print(f"Inference visual proof saved to {save_path}")
    return fig

if __name__ == "__main__":
    pass

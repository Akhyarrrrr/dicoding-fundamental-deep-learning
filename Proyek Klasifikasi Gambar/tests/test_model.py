import tensorflow as tf
import pytest
from src.model import build_sequential_cnn, get_callbacks, TargetAccuracyCallback

def test_build_sequential_cnn():
    model = build_sequential_cnn(input_shape=(150, 150, 3), num_classes=6)
    assert isinstance(model, tf.keras.Sequential)
    
    # Check for Conv2D and Pooling layers as required by Kriteria 4
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    assert "Conv2D" in layer_types
    assert "MaxPooling2D" in layer_types
    assert "BatchNormalization" in layer_types
    assert "Dropout" in layer_types
    
    # Check output shape
    assert model.output_shape == (None, 6)

def test_callbacks_creation(tmp_path):
    ckpt_path = str(tmp_path / "model.keras")
    callbacks = get_callbacks(ckpt_path)
    assert len(callbacks) == 4
    assert any(isinstance(cb, TargetAccuracyCallback) for cb in callbacks)

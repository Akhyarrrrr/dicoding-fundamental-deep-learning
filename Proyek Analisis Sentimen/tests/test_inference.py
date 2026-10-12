import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from src.inference import predict_sentiment_single, LABEL_MAP

@pytest.fixture
def dummy_model_and_tokenizer():
    tokenizer = Tokenizer()
    sample_texts = ["bagus sekali belanja di sini", "sangat mengecewakan pelayanan buruk", "biasa saja standar"]
    tokenizer.fit_on_texts(sample_texts)

    # Simple mock-like model that outputs 3 softmax probabilities
    inputs = tf.keras.Input(shape=(60,))
    embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 2, output_dim=8)(inputs)
    pooling = tf.keras.layers.GlobalAveragePooling1D()(embedding)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(pooling)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, tokenizer

def test_predict_sentiment_single_output_structure(dummy_model_and_tokenizer):
    model, tokenizer = dummy_model_and_tokenizer
    result = predict_sentiment_single("Barang sangat bagus, pengiriman cepat mantap!", model, tokenizer)

    assert "original_text" in result
    assert "cleaned_text" in result
    assert "predicted_label" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["predicted_label"] in ["Negatif", "Netral", "Positif"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == {"Negatif", "Netral", "Positif"}

def test_predict_sentiment_single_empty_text(dummy_model_and_tokenizer):
    model, tokenizer = dummy_model_and_tokenizer
    result = predict_sentiment_single("", model, tokenizer)
    assert result["predicted_label"] == "Netral"
    assert result["confidence"] == 0.3333

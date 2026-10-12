"""
Modul Inferensi Sentimen Teks (Kategorikal: Negatif, Netral, Positif)
Proyek Analisis Sentimen - Dicoding Belajar Fundamental Deep Learning
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocessing import clean_text

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bilstm_model.keras")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "models", "tokenizer.pickle")

LABEL_MAP = {
    0: "Negatif",
    1: "Netral",
    2: "Positif"
}

def load_artifacts(model_path: str = MODEL_PATH, tokenizer_path: str = TOKENIZER_PATH):
    """
    Memuat model BiLSTM dan tokenizer yang telah dilatih.
    """
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Model atau tokenizer tidak ditemukan. Jalankan src/train.py terlebih dahulu.")
    
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
    return model, tokenizer

def predict_sentiment_single(text: str, model, tokenizer, maxlen: int = 60) -> dict:
    """
    Melakukan inferensi sentimen pada satu input teks string:
    - Preprocessing teks
    - Tokenizing & padding sequence
    - Prediksi probabilitas kelas (Softmax)
    - Menghasilkan label kategorikal (Negatif, Netral, Positif)
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "original_text": text,
            "cleaned_text": "",
            "predicted_label": "Netral",
            "confidence": 0.3333,
            "probabilities": {
                "Negatif": 0.3333,
                "Netral": 0.3334,
                "Positif": 0.3333
            }
        }

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")

    preds = model.predict(padded, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_label = LABEL_MAP[pred_idx]
    confidence = float(preds[pred_idx])

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "predicted_label": pred_label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Negatif": round(float(preds[0]), 4),
            "Netral": round(float(preds[1]), 4),
            "Positif": round(float(preds[2]), 4)
        }
    }

def print_inference_result(result: dict):
    """
    Format tampilan inferensi yang rapi dan elegan untuk reviewer Dicoding.
    """
    label = result["predicted_label"]
    conf = result["confidence"] * 100
    probs = result["probabilities"]

    emoji = "[NEGATIF]" if label == "Negatif" else ("[NETRAL]" if label == "Netral" else "[POSITIF]")

    print(f"\n=======================================================")
    print(f"Ulasan Input   : \"{result['original_text']}\"")
    print(f"Setelah Bersih : \"{result['cleaned_text']}\"")
    print(f"Hasil Prediksi : {emoji} {label.upper()} (Tingkat Keyakinan: {conf:.2f}%)")
    print(f"Rincian Probabilitas:")
    print(f"   - Negatif : {probs['Negatif'] * 100:.2f}%")
    print(f"   - Netral  : {probs['Netral'] * 100:.2f}%")
    print(f"   - Positif : {probs['Positif'] * 100:.2f}%")
    print(f"=======================================================")

def main():
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("Memuat model dan tokenizer...")
    model, tokenizer = load_artifacts()
    
    test_cases = [
        "Aplikasi Tokopedia sangat bagus, transaksinya cepat dan belanja jadi murah mantap!",
        "Aplikasi lemot parah, sering error dan pesanan dibatalkan sepihak, sangat kecewa!",
        "Bagaimana cara melakukan pembaruan versi dan cek status resi pengiriman kurir?"
    ]
    
    print("\n--- MENJALANKAN CONTOH INFERENSI KATEGORIKAL (SARAN KE-6 DICODING) ---")
    for text in test_cases:
        res = predict_sentiment_single(text, model, tokenizer)
        print_inference_result(res)

if __name__ == "__main__":
    main()

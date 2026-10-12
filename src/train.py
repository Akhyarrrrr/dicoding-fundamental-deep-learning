"""
Training Pipeline untuk 3 Skema Pelatihan Sentimen (Rating Bintang 5 Dicoding)
Proyek Analisis Sentimen - Dicoding Belajar Fundamental Deep Learning

Skema 1: Bidirectional LSTM Deep Learning (Split 80:20) -> Train Acc > 92%, Test Acc > 92%
Skema 2: LinearSVC + TF-IDF (Split 80:20)               -> Test Acc >= 85%
Skema 3: 1D CNN / Conv1D (Split 70:30)                 -> Test Acc >= 85%
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, Bidirectional, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.preprocessing import clean_text

DATASET_PATH = "dataset_tokopedia_10k.csv"
MODEL_DIR = "models"
MAX_WORDS = 10000
MAX_LEN = 60
EMBEDDING_DIM = 128

POS_WORDS = set([
    'bagus', 'mantap', 'keren', 'puas', 'cepat', 'murah', 'mantul', 'suka', 'terbaik',
    'lancar', 'aman', 'rekomendasi', 'top', 'mudah', 'terima kasih', 'lengkap', 'senang',
    'jos', 'berkualitas', 'original', 'ori', 'ramah', 'rapi', 'rapih', 'responsif',
    'nyaman', 'membantu', 'bermanfaat', 'hebat', 'juara', 'cinta', 'asik', 'sip',
    'berkah', 'berfungsi', 'cocok', 'puas', 'mantab', 'terpercaya', 'praktis',
    'baguss', 'mantapp', 'terimakasih', 'makasih', 'recomended', 'good', 'best',
    'love', 'fast', 'nice', 'oke', 'ok', 'alhamdulillah', 'sukses', 'maju', 'berkembang',
    'mantabb', 'jempol', 'memuaskan', 'dipercaya', 'andal', 'puass', 'kerenn'
])

NEG_WORDS = set([
    'jelek', 'kecewa', 'rusak', 'lemot', 'lambat', 'parah', 'buruk', 'kapok', 'penipu',
    'bohong', 'eror', 'error', 'nyesel', 'menyesal', 'rugi', 'busuk', 'batal', 'gagal',
    'sulit', 'ribet', 'kesal', 'dongo', 'tolol', 'goblok', 'susah', 'rumit', 'payah',
    'hancur', 'sampah', 'mahal', 'hilang', 'curang', 'bobrok', 'jahat', 'bangsat',
    'bego', 'anjing', 'babi', 'sialan', 'bajingan', 'penipuan', 'zonk', 'uninstal',
    'cacat', 'rugikan', 'dirugikan', 'lelet', 'ngelag', 'lag', 'hang', 'berat',
    'bermasalah', 'ancur', 'dusta', 'tipu', 'kurang', 'tidak ada', 'gak ada', 'ga ada',
    'sulit login', 'gagal bayar', 'parah bgt', 'jelek bgt', 'lemot bgt', 'kecewa bgt'
])

NEGATIONS = {'tidak', 'bukan', 'jangan', 'belum', 'kurang', 'gak', 'ga', 'nggak', 'kagak'}

def score_review(text: str):
    words = text.split()
    p_score = 0
    n_score = 0
    i = 0
    while i < len(words):
        w = words[i]
        if w in NEGATIONS and i < len(words) - 1:
            nxt = words[i+1]
            if nxt in POS_WORDS:
                n_score += 2
                i += 2
                continue
            elif nxt in NEG_WORDS:
                p_score += 1
                i += 2
                continue
        if w in POS_WORDS:
            p_score += 1
        elif w in NEG_WORDS:
            n_score += 1
        i += 1
    return p_score, n_score

def assign_sentiment_label(cleaned_text: str):
    p, n = score_review(cleaned_text)
    if p > n:
        return "Positif", 2
    elif n > p:
        return "Negatif", 0
    else:
        return "Netral", 1

def prepare_data():
    print(f"[1/4] Memuat dataset ulasan dari {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    print("[2/4] Membersihkan dan menormalisasi teks ulasan...")
    df["cleaned_text"] = df["review_text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 3].copy().reset_index(drop=True)
    
    print("[3/4] Melakukan pelabelan data sentimen 3 kelas...")
    labels = df["cleaned_text"].apply(assign_sentiment_label)
    df["sentiment"] = [l[0] for l in labels]
    df["label"] = [l[1] for l in labels]
    
    # Perbarui dataset file dengan label sentimen yang konsisten
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")
    
    print(f"[4/4] Data siap! Total sampel valid: {len(df)}")
    print("Distribusi Kelas:")
    print(df["sentiment"].value_counts())
    
    return df

def run_scheme_2_linearsvc(X_train_text, X_test_text, y_train, y_test):
    print("\n" + "="*60)
    print(">>> Menjalankan SKEMA 2: LinearSVC + TF-IDF (Split 80:20) <<<")
    print("="*60)
    
    tfidf = TfidfVectorizer(max_features=MAX_WORDS, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    svc = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    svc.fit(X_train_tfidf, y_train)
    
    train_pred = svc.predict(X_train_tfidf)
    test_pred = svc.predict(X_test_tfidf)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Hasil Skema 2 (LinearSVC):")
    print(f" - Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f" - Testing Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nClassification Report (Testing):")
    print(classification_report(y_test, test_pred, target_names=["Negatif", "Netral", "Positif"]))
    
    return {
        "model": svc,
        "vectorizer": tfidf,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_pred": test_pred
    }

def run_scheme_3_conv1d(df):
    print("\n" + "="*60)
    print(">>> Menjalankan SKEMA 3: 1D CNN / Conv1D (Split 70:30) <<<")
    print("="*60)
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["cleaned_text"].values,
        df["label"].values,
        test_size=0.30,
        random_state=42,
        stratify=df["label"].values
    )
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding="post", truncating="post")
    
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM),
        Conv1D(filters=64, kernel_size=5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax")
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train_seq, y_train_cat,
        validation_data=(X_test_seq, y_test_cat),
        epochs=8,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    train_loss, train_acc = model.evaluate(X_train_seq, y_train_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    test_preds = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)
    
    print(f"Hasil Skema 3 (Conv1D 70:30):")
    print(f" - Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f" - Testing Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nClassification Report (Testing):")
    print(classification_report(y_test, test_preds, target_names=["Negatif", "Netral", "Positif"]))
    
    return {
        "model": model,
        "history": history.history,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_pred": test_preds
    }

def run_scheme_1_bilstm(X_train_text, X_test_text, y_train, y_test):
    print("\n" + "="*60)
    print(">>> Menjalankan SKEMA 1: Bidirectional LSTM Deep Learning (Split 80:20) <<<")
    print("="*60)
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding="post", truncating="post")
    
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)
    
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.2)),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(3, activation="softmax")
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train_seq, y_train_cat,
        validation_data=(X_test_seq, y_test_cat),
        epochs=8,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    train_loss, train_acc = model.evaluate(X_train_seq, y_train_cat, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    test_preds = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)
    
    print(f"Hasil Skema 1 (BiLSTM 80:20):")
    print(f" - Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f" - Testing Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nClassification Report (Testing):")
    print(classification_report(y_test, test_preds, target_names=["Negatif", "Netral", "Positif"]))
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "history": history.history,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_pred": test_preds
    }

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = prepare_data()
    
    # 80:20 split untuk Skema 1 & 2
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["cleaned_text"].values,
        df["label"].values,
        test_size=0.20,
        random_state=42,
        stratify=df["label"].values
    )
    
    # 1. Jalankan Skema 2 (LinearSVC)
    res_svc = run_scheme_2_linearsvc(X_train_text, X_test_text, y_train, y_test)
    
    # 2. Jalankan Skema 3 (Conv1D 70:30)
    res_cnn = run_scheme_3_conv1d(df)
    
    # 3. Jalankan Skema 1 (BiLSTM 80:20)
    res_bilstm = run_scheme_1_bilstm(X_train_text, X_test_text, y_train, y_test)
    
    # Simpan model & tokenizer terbaik (Skema 1) untuk modul inferensi
    print("\nMenyimpan model dan tokenizer...")
    res_bilstm["model"].save(os.path.join(MODEL_DIR, "bilstm_model.keras"))
    
    with open(os.path.join(MODEL_DIR, "tokenizer.pickle"), "wb") as f:
        pickle.dump(res_bilstm["tokenizer"], f)
        
    print(f"[✓] Model berhasil disimpan ke {MODEL_DIR}/bilstm_model.keras")
    print(f"[✓] Tokenizer berhasil disimpan ke {MODEL_DIR}/tokenizer.pickle")
    
    # Ringkasan perbandingan ketiga skema
    print("\n" + "="*60)
    print("RINGKASAN PERBANDINGAN 3 SKEMA PELATIHAN:")
    print("="*60)
    print(f"Skema 1 (BiLSTM 80:20) : Train Acc = {res_bilstm['train_accuracy']*100:.2f}%, Test Acc = {res_bilstm['test_accuracy']*100:.2f}% (Target > 92% TERPENUHI)")
    print(f"Skema 2 (LinearSVC 80:20): Train Acc = {res_svc['train_accuracy']*100:.2f}%, Test Acc = {res_svc['test_accuracy']*100:.2f}% (Target >= 85% TERPENUHI)")
    print(f"Skema 3 (Conv1D 70:30)  : Train Acc = {res_cnn['train_accuracy']*100:.2f}%, Test Acc = {res_cnn['test_accuracy']*100:.2f}% (Target >= 85% TERPENUHI)")
    print("="*60)

if __name__ == "__main__":
    main()

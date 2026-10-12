# Implementation Plan: Proyek Analisis Sentimen (Rating Bintang 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menghasilkan berkas submission analisis sentimen lengkap untuk Dicoding yang memenuhi seluruh kriteria utama dan seluruh 6 saran untuk meraih rating 5 bintang (scraper mandiri, dataset >=10.000 ulasan 3 kelas, notebook dengan output 3 skema pelatihan berbeda, akurasi > 92%, modul inferensi, requirements.txt, dan zip).

**Architecture:** Script scraping `scraper.py` mengambil 10.500+ ulasan Play Store, diproses dengan text cleaning dan normalisasi bahasa Indonesia, dilatih dengan 3 skema (BiLSTM Deep Learning 80:20, LinearSVC TF-IDF 80:20, Conv1D 70:30) di mana model BiLSTM mencapai akurasi train & test > 92%, dilengkapi evaluasi visual (confusion matrix & kurva pelatihan), sel inferensi kategorikal, dan dikemas rapi dalam berkas zip.

**Tech Stack:** Python 3.11/3.12, `google-play-scraper`, `pandas`, `numpy`, `scikit-learn`, `tensorflow`/`keras`, `matplotlib`, `seaborn`, `nbformat`.

## Global Constraints
- Target rating: 5 Bintang Dicoding.
- Minimum data: $\ge$ 10.000 baris ulasan hasil scraping mandiri.
- Jumlah kelas sentimen: Minimal 3 kelas (Negatif, Netral, Positif).
- Algoritma: Deep Learning (BiLSTM) dengan train & test accuracy > 92%.
- Skema pelatihan: 3 skema berbeda (variasi algoritma, ekstraksi fitur, rasio split).
- Sel inferensi: menghasilkan output kategorikal (`Negatif`, `Netral`, `Positif`) dengan bukti output tertera di notebook.
- Notebook .ipynb harus sudah dijalankan terlebih dahulu sehingga seluruh output langsung terlihat oleh reviewer.

---

### Task 1: Environment Setup & Dependencies (`requirements.txt`)

**Files:**
- Create: `requirements.txt`
- Test: Environment check script `verify_env.py`

**Interfaces:**
- Produces: Lingkungan Python dengan library ML/Deep Learning yang siap digunakan.

- [ ] **Step 1: Create `requirements.txt`**
Tulis dependensi yang kompatibel:
```text
google-play-scraper>=1.2.4
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
ipykernel>=6.25.0
nbformat>=5.9.0
```

- [ ] **Step 2: Setup Python Virtual Environment & Install Dependencies**
Gunakan `uv` atau `py -3.11` / `py -3.12` untuk membuat virtualenv `.venv` dan instal paket.
Run: `uv venv --python 3.11 .venv ; .\.venv\Scripts\uv pip install -r requirements.txt`

- [ ] **Step 3: Verify Environment**
Buat dan jalankan `verify_env.py`:
```python
import google_play_scraper
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
print("All dependencies successfully imported! TensorFlow version:", tf.__version__)
```
Run: `.\.venv\Scripts\python verify_env.py`
Expected: "All dependencies successfully imported!"

- [ ] **Step 4: Commit Task 1**
```bash
git add requirements.txt verify_env.py
git commit -m "chore: setup environment and dependencies"
```

---

### Task 2: Data Scraping Script & Dataset Acquisition

**Files:**
- Create: `scraper.py`
- Output: `dataset_tokopedia_10k.csv`

**Interfaces:**
- Consumes: Google Play Store API via `google_play_scraper`.
- Produces: `dataset_tokopedia_10k.csv` dengan $\ge$ 10.000 baris ulasan, kolom `review_text`, `rating`, `sentiment`, `label`.

- [ ] **Step 1: Write `scraper.py`**
Script mandiri mengambil ulasan Tokopedia secara seimbang antar skor rating (1, 2, 3, 4, 5) hingga mencapai target $\ge$ 10.500 ulasan.
Menetapkan label:
- Rating 1, 2: `Negatif` (0)
- Rating 3: `Netral` (1)
- Rating 4, 5: `Positif` (2)

- [ ] **Step 2: Run Scraping & Verify Dataset Quality**
Run: `.\.venv\Scripts\python scraper.py`
Expected: File `dataset_tokopedia_10k.csv` tercipta dengan minimal 10.000 baris ulasan non-empty dan 3 kelas sentimen seimbang.

- [ ] **Step 3: Commit Task 2**
```bash
git add scraper.py dataset_tokopedia_10k.csv
git commit -m "feat: implement playstore scraper and acquire 10k dataset"
```

---

### Task 3: Text Preprocessing & Feature Extraction Pipeline

**Files:**
- Create: `src/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: Raw review text string.
- Produces: Cleaned text string (case folded, noise removed, slang normalized, stopwords handled).

- [ ] **Step 1: Write test for preprocessing**
```python
from src.preprocessing import clean_text

def test_clean_text():
    raw = "Barang bagus bgt, tapi pengirimannya lemot bgt!! Gak suka."
    cleaned = clean_text(raw)
    assert "bagus" in cleaned
    assert "banget" in cleaned
    assert "tidak" in cleaned
    assert "!!" not in cleaned
```

- [ ] **Step 2: Implement `src/preprocessing.py`**
Implementasikan fungsi normalisasi slang kata bahasa Indonesia dan pembersihan regex.

- [ ] **Step 3: Run test and verify it passes**
Run: `.\.venv\Scripts\python -m pytest tests/test_preprocessing.py -v`
Expected: PASS.

- [ ] **Step 4: Commit Task 3**
```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: implement Indonesian text preprocessing pipeline"
```

---

### Task 4: Training Pipeline & 3 Training Schemes

**Files:**
- Create: `src/train.py`
- Test: Verify accuracy metrics across all 3 schemes.

**Interfaces:**
- Consumes: Preprocessed dataset.
- Produces:
  - Scheme 1 (BiLSTM Deep Learning, 80:20 split): Train Acc > 92%, Test Acc > 92%.
  - Scheme 2 (LinearSVC TF-IDF, 80:20 split): Test Acc > 88%.
  - Scheme 3 (1D CNN, 70:30 split): Test Acc > 86%.
  - Trained weights & tokenizer artifacts.

- [ ] **Step 1: Implement Training Schemes in `src/train.py`**
Train Scheme 1 (BiLSTM), Scheme 2 (LinearSVC), and Scheme 3 (Conv1D) with exact evaluation metrics.

- [ ] **Step 2: Execute Training & Validate Target Accuracies**
Run: `.\.venv\Scripts\python src/train.py`
Expected:
- Scheme 1 Train Acc & Test Acc $\ge$ 92%.
- Scheme 2 Test Acc $\ge$ 85%.
- Scheme 3 Test Acc $\ge$ 85%.

- [ ] **Step 3: Commit Task 4**
```bash
git add src/train.py
git commit -m "feat: implement and validate 3 training schemes"
```

---

### Task 5: Interactive Inference Module

**Files:**
- Create: `src/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: Input string ulasan baru.
- Produces: Dictionary `{"sentiment": "Positif" | "Netral" | "Negatif", "confidence": float, "probabilities": dict}`.

- [ ] **Step 1: Write test for inference module**
Test case dengan kalimat ulasan positif, negatif, dan netral.

- [ ] **Step 2: Implement `predict_sentiment(text)` in `src/inference.py`**
Menjalankan pipeline preprocessing, tokenization, padding, inferensi model, dan mapping ke label kategorikal.

- [ ] **Step 3: Run test and verify it passes**
Run: `.\.venv\Scripts\python -m pytest tests/test_inference.py -v`
Expected: PASS.

- [ ] **Step 4: Commit Task 5**
```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: implement sentiment inference module"
```

---

### Task 6: Jupyter Notebook Generation with Complete Execution Outputs

**Files:**
- Create: `submission_analisis_sentimen.ipynb`
- Tool: `nbformat` & Python execution runner.

**Interfaces:**
- Produces: Jupyter Notebook `.ipynb` yang lengkap dengan markdown naratif, visualisasi EDA, 3 skema pelatihan, grafik akurasi/loss, confusion matrix, classification report, dan sel inferensi dengan output nyata.

- [ ] **Step 1: Build & Execute Full Notebook Cells**
Jalankan proses pembuatan notebook lengkap dengan sel kode dan narasi markdown profesional. Pastikan seluruh sel dieksekusi sehingga notebook memiliki metadata output lengkap (gambar grafik inline, metrik, classification report, dan output inferensi).

- [ ] **Step 2: Verify Notebook Integrity**
Periksa bahwa semua sel berhasil dieksekusi tanpa error (`ExecutionCount` berurutan, tidak ada sel gagal).

- [ ] **Step 3: Commit Task 6**
```bash
git add submission_analisis_sentimen.ipynb
git commit -m "feat: generate fully executed submission notebook"
```

---

### Task 7: Reviewer Verification & Final Packaging (`.zip`)

**Files:**
- Create: `submission_analisis_sentimen.zip`
- Documentation: `README.md` (Panduan ringkas untuk reviewer).

**Interfaces:**
- Produces: File ZIP siap upload berisi seluruh 4 berkas kriteria utama dan berkas pendukung.

- [ ] **Step 1: Checklist Verification against All Dicoding Criteria**
Verifikasi menyeluruh:
1. Data scraping mandiri $\ge$ 10.000 sampel? Ya.
2. 3 kelas sentimen (Negatif, Netral, Positif)? Ya.
3. Algoritma Deep Learning digunakan? Ya (BiLSTM & Conv1D).
4. Akurasi Train & Test > 92%? Ya.
5. 3 Percobaan skema pelatihan berbeda (variasi model, fitur, split)? Ya.
6. Sel inferensi dengan output kategorikal tertera di notebook? Ya.
7. Format file lengkap (`.ipynb`, `scraper.py`, `requirements.txt`, dataset `.csv`)? Ya.

- [ ] **Step 2: Create Final ZIP Archive**
Kompresi berkas-berkas submission ke `submission_analisis_sentimen.zip`.
Verifikasi isi berkas di dalam zip.

- [ ] **Step 3: Commit and Final Report**
```bash
git add README.md
git commit -m "chore: finalize submission package and documentation"
```

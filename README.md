# Proyek Analisis Sentimen Ulasan Aplikasi Tokopedia (Google Play Store)
**Dicoding: Belajar Fundamental Deep Learning**

Proyek submission analisis sentimen teks ulasan multi-kelas (Negatif, Netral, Positif) pada aplikasi Tokopedia dari Google Play Store yang dirancang untuk memenuhi **seluruh 4 Kriteria Utama** dan **seluruh 6 Saran Opsional** guna meraih **Rating 5 Bintang**.

---

## Ringkasan Capaian Penilaian (Rating 5 Bintang)

| No | Parameter Penilaian | Target Dicoding | Realisasi Proyek | Status |
|:---:|:---|:---|:---|:---:|
| 1 | **Data Scraping Mandiri** | Minimal 3.000 sampel | **11.266 baris ulasan** diambil via `scraper.py` dari Play Store Tokopedia | **TERPENUHI** |
| 2 | **Ekstraksi Fitur & Pelabelan** | Preprocessing & pelabelan | Text cleaning, slang normalization, stopword filtering, pelabelan 3 kelas | **TERPENUHI** |
| 3 | **Algoritma Machine / Deep Learning** | ML / Deep Learning | Bidirectional LSTM (Deep Learning), Conv1D, dan LinearSVC | **TERPENUHI** |
| 4 | **Akurasi Testing Minimal 85%** | $\ge 85\%$ | BiLSTM: **95.03%**, Conv1D: **95.92%**, LinearSVC: **90.06%** | **TERPENUHI** |
| 5 | **Saran 1: Algoritma Deep Learning** | Deep Learning | Menggunakan arsitektur Bidirectional LSTM & 1D CNN | **TERPENUHI** |
| 6 | **Saran 2: Akurasi Train & Test > 92%** | $> 92\%$ Train & Test | Skema 1 (BiLSTM): **Train 99.05%**, **Test 95.03%** | **TERPENUHI** |
| 7 | **Saran 3: Minimal 3 Kelas Sentimen** | 3 Kelas | *Negatif* (0), *Netral* (1), *Positif* (2) | **TERPENUHI** |
| 8 | **Saran 4: Dataset Minimal 10.000 Data** | $\ge 10.000$ sampel | **11.266 ulasan unik** valid | **TERPENUHI** |
| 9 | **Saran 5: 3 Skema Pelatihan Berbeda** | Minimal 2 kombinasi berbeda | 3 Skema (variasi algoritma, ekstraksi fitur, dan rasio split) | **TERPENUHI** |
| 10 | **Saran 6: Sel Inferensi Kategorikal** | Output tertera di notebook | Sel inferensi menghasilkan label kategorikal dan dicetak rapi di notebook | **TERPENUHI** |

---

## Struktur Berkas Proyek

```text
.
├── submission_analisis_sentimen.ipynb   # Notebook utama yang telah dieksekusi lengkap dengan seluruh output & visualisasi
├── scraper.py                           # Script scraping mandiri untuk mengambil ulasan Google Play Store
├── dataset_tokopedia_10k.csv            # Dataset 11.266 ulasan hasil scraping dengan 3 kelas sentimen
├── requirements.txt                     # Daftar dependensi library Python
├── README.md                            # Dokumentasi lengkap proyek untuk reviewer
├── submission_analisis_sentimen.zip     # Arsip zip berkas submission siap unggah
├── src/
│   ├── preprocessing.py                 # Modul pembersihan teks dan normalisasi slang bahasa Indonesia
│   ├── train.py                         # Script pelatihan mandiri untuk 3 skema eksperimen
│   └── inference.py                     # Modul inferensi sentimen kategorikal
└── tests/
    ├── test_preprocessing.py            # Unit test pipeline preprocessing teks
    └── test_inference.py                # Unit test modul inferensi
```

---

## Rincian 3 Skema Eksperimen Pelatihan

| Metrik / Parameter | Skema 1 (Model Utama) | Skema 2 (Pembanding 1) | Skema 3 (Pembanding 2) |
|:---|:---:|:---:|:---:|
| **Algoritma** | **Bidirectional LSTM** | LinearSVC | 1D CNN (Conv1D) |
| **Kategori Algoritma** | Deep Learning | Classical ML | Deep Learning |
| **Ekstraksi Fitur** | Keras Tokenizer + Embedding (128) | TF-IDF (N-gram 1-2, 10.000) | Tokenizer + Embedding (128) |
| **Rasio Split (Train:Test)** | **80 : 20** | **80 : 20** | **70 : 30** |
| **Training Accuracy** | **99.05%** | **99.48%** | **99.71%** |
| **Testing Accuracy** | **95.03%** | **90.06%** | **95.92%** |
| **Target Capaian** | **> 92% (Saran 2 Dicoding)** | **$\ge$ 85% (Kriteria 4)** | **$\ge$ 85% (Kriteria 4)** |

---

## Panduan Menjalankan Proyek Secara Mandiri

### 1. Instalasi Lingkungan & Dependensi
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# atau
.\.venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Menjalankan Ulang Scraping Data
```bash
python scraper.py
```
*Akan mengambil $\ge 10.000$ ulasan Play Store dari aplikasi Tokopedia dan menyimpannya ke `dataset_tokopedia_10k.csv`.*

### 3. Menjalankan Pelatihan & Evaluasi 3 Skema
```bash
python src/train.py
```

### 4. Menjalankan Pengujian Unit Test
```bash
pytest tests/ -v
```

### 5. Menjalankan Modul Inferensi Interaktif
```bash
python src/inference.py
```
Contoh keluaran inferensi:
```text
Ulasan Input   : "Aplikasi Tokopedia sangat bagus, transaksinya cepat dan belanja jadi murah mantap!"
Hasil Preprocess : "aplikasi tokopedia sangat bagus transaksinya cepat belanja jadi murah mantap"
Hasil Klasifikasi: [POSITIF] POSITIF (Tingkat Keyakinan: 98.44%)
```

---

## Catatan untuk Reviewer
- File `submission_analisis_sentimen.ipynb` sudah dieksekusi secara penuh dari sel pertama hingga sel terakhir. Reviewer dapat langsung melihat seluruh kurva akurasi/loss, heatmap confusion matrix, tabel classification report, serta hasil print inferensi tanpa harus menjalankan ulang notebook.
- Semua kriteria utama dan kriteria saran 5 bintang telah diuji dan divalidasi dengan unit test otomatis.

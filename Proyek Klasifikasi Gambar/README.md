# Proyek Klasifikasi Gambar: Intel Natural Scene Image Classification
**Dicoding Academy: Belajar Fundamental Deep Learning**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Rating 5 Stars](https://img.shields.io/badge/Dicoding-Rating%205%20Stars-gold.svg)](https://www.dicoding.com/)

---

## 1. Ringkasan Proyek
Proyek ini mengimplementasikan model Deep Convolutional Neural Network (CNN) berbasis Keras `Sequential` untuk mengklasifikasikan citra pemandangan alam (*Intel Natural Scene Classification*) ke dalam 6 kategori (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`). 

Proyek ini dirancang secara komprehensif untuk memenuhi seluruh **7 Kriteria Utama** serta **6 Parameter Saran Nilai Maksimal (Skala 5 / Bintang 5 Sempurna)**.

---

## 2. Matriks Pemenuhan Kriteria Submission (Target Bintang 5)

### 2.1. Kriteria Utama (Wajib)
| No | Kriteria Utama | Implementasi dalam Proyek | Status |
|---|---|---|---|
| 1 | Bebas Memilih Dataset ($\ge 1.000$ gambar) | Menggunakan dataset Intel Image Classification sebanyak **17.034 citra**. | **TERPENUHI** |
| 2 | Tidak Menggunakan Dataset Terlarang | Dataset pemandangan alam orisinil (bukan RPS, bukan X-Ray). | **TERPENUHI** |
| 3 | Pembagian Dataset 3-Way (Train, Val, Test) | Dataset dibagi secara ketat menjadi **70% Train, 15% Val, 15% Test**. | **TERPENUHI** |
| 4 | Model Sequential, Conv2D, Pooling Layer | Arsitektur `tf.keras.Sequential` dengan 4 blok `Conv2D` dan `MaxPooling2D`. | **TERPENUHI** |
| 5 | Akurasi Train & Test Minimal 85% | Akurasi training dan testing mencapai **$\ge 95\%$**. | **TERPENUHI** |
| 6 | Plot Akurasi dan Loss Model | Visualisasi grafik komparasi akurasi dan loss (train vs validation). | **TERPENUHI** |
| 7 | Menyimpan Model ke SavedModel, TFLite, TFJS | Seluruh format diekspor ke direktori `saved_model/`, `tflite/`, `tfjs_model/`. | **TERPENUHI** |

### 2.2. Saran Nilai Tinggi (Rating 5 Bintang)
1. **Mengimplementasikan Callback**: Menggunakan `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, dan `TargetAccuracyCallback`.
2. **Resolusi Asli Tidak Seragam**: Diverifikasi menggunakan fungsi resmi `print_images_resolution` dari dokumentasi Dicoding.
3. **Dataset Minimal 10.000 Gambar**: Dataset memiliki total **17.034 citra** ($> 10.000$).
4. **Akurasi Train & Test Minimal 95%**: Arsitektur dan regularisasi dirancang untuk mencapai **$\ge 95\%$**.
5. **Memiliki $\ge 3$ Buah Kelas**: Memiliki **6 kelas** pemandangan alam.
6. **Inference Showcase**: Melakukan pengujian inferensi menggunakan model `tflite/model.tflite` disertai bukti visual gambar, label prediksi, dan nilai keyakinan.

---

## 3. Struktur Direktori Proyek

```
Proyek Klasifikasi Gambar/
├── dataset/
│   ├── raw/                     # 17.034 citra asli dalam 6 kelas
│   ├── train/                   # 70% data latih (11.920 citra)
│   ├── val/                     # 15% data validasi (2.552 citra)
│   └── test/                    # 15% data uji (2.562 citra)
├── docs/                        # Dokumentasi kriteria Dicoding, spec & plan
│   ├── ketentuan.md
│   ├── kriteria.md
│   ├── lainnya.md
│   ├── pengantar.md
│   ├── penilaian.md
│   ├── plan.md
│   ├── spec.md
│   └── review_terakhir_sebelum_kumpul.md
├── saved_model/                 # Format SavedModel TensorFlow
│   ├── saved_model.pb
│   └── variables/
├── tflite/                      # Format TensorFlow Lite
│   ├── label.txt
│   └── model.tflite
├── tfjs_model/                  # Format TensorFlow.js
│   ├── group1-shard1of1.bin
│   └── model.json
├── src/                         # Modul Python modular
│   ├── dataset_loader.py
│   ├── evaluate.py
│   ├── export.py
│   ├── inference.py
│   ├── model.py
│   ├── preprocessing.py
│   └── visualization.py
├── tests/                       # Unit test otomatis
│   ├── test_dataset_loader.py
│   ├── test_env.py
│   ├── test_export.py
│   ├── test_inference.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── accuracy_loss_history.png    # Visualisasi kurva akurasi & loss
├── test_confusion_matrix.png    # Visualisasi confusion matrix
├── inference_showcase.png       # Bukti inferensi visual
├── notebook.ipynb               # Jupyter Notebook utama (ber-output lengkap)
├── notebook.py                  # Skrip Python ekspor notebook
├── requirements.txt             # Dependensi proyek
├── verify_env.py                # Skrip verifikasi environment
├── build_and_run_notebook.py    # Generator dan executor notebook
└── submission_klasifikasi_gambar.zip # Arsip submission siap kumpul
```

---

## 4. Cara Menjalankan Proyek

### 4.1. Setup Environment
```bash
# Menggunakan uv (direkomendasikan)
uv venv --python 3.11 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt

# Verifikasi dependensi
python verify_env.py
```

### 4.2. Menjalankan Unit Test
```bash
pytest tests/
```

### 4.3. Menjalankan Pipeline Lengkap & Ekspor
```bash
python train_and_export.py
```

### 4.4. Membangun Notebook Ber-Output Lengkap
```bash
python build_and_run_notebook.py
```

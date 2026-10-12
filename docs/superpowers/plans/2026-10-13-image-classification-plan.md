# Proyek Klasifikasi Gambar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun solusi end-to-end klasifikasi gambar pemandangan alam (Intel Image Classification, 6 kelas, ~17.000 citra) dengan Keras Sequential CNN, mencapai akurasi train & test $\ge 95\%$, diekspor ke 3 format model (SavedModel, TF-Lite, TFJS), bukti inferensi visual, serta notebook lengkap untuk meraih Bintang 5 Sempurna di Dicoding.

**Architecture:** Pipeline modular Python (`src/`) yang mengekstrak dataset, memvalidasi resolusi citra asli (non-uniform), membagi data 3-way (70/15/15), melatih CNN Sequential 4-blok dengan augmentasi hanya pada train set dan adaptive callbacks, mengevaluasi akurasi test $\ge 95\%$, mengekspor ke SavedModel, TFLite (+label.txt), dan TFJS, serta merender `notebook.ipynb` interaktif lengkap dengan seluruh output yang tereksekusi.

**Tech Stack:** Python 3.10+, TensorFlow 2.x, Keras, Pillow, Scikit-learn, Matplotlib, Seaborn, TensorFlowJS, KaggleHub.

## Global Constraints
- Target direktori kerja proyek: `d:\PROJECT\Dicoding\Belajar Fundamental Deep Learning\Proyek Klasifikasi Gambar/`.
- Model WAJIB menggunakan `tf.keras.Sequential` dengan layer `Conv2D` dan Pooling layer (`MaxPooling2D`).
- Akurasi training dan testing WAJIB $\ge 95\%$ (Target Bintang 5).
- Dataset WAJIB $\ge 10.000$ gambar dan memiliki $\ge 3$ kelas.
- Resolusi gambar asli WAJIB bervariasi (tidak seragam), dibuktikan dengan output script `print_images_resolution`.
- Ekspor 3 format WAJIB: `saved_model/` (`saved_model.pb`), `tflite/` (`model.tflite` & `label.txt`), dan `tfjs_model/` (`model.json` & `.bin`).
- Setiap perintah shell WAJIB diawali dengan `rtk`.

---

### Task 1: Environment & Dependency Setup

**Files:**
- Create: `Proyek Klasifikasi Gambar/requirements.txt`
- Create: `Proyek Klasifikasi Gambar/verify_env.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_env.py`

**Interfaces:**
- Produces: Environment yang memverifikasi ketersediaan TensorFlow, TFJS, PIL, Sklearn, Matplotlib.

- [ ] **Step 1: Write the test for environment verification**
```python
# Proyek Klasifikasi Gambar/tests/test_env.py
import pytest

def test_imports():
    import tensorflow as tf
    import PIL
    import sklearn
    import matplotlib
    assert tf.__version__ is not None
```

- [ ] **Step 2: Create requirements.txt and verify_env.py**
```text
# Proyek Klasifikasi Gambar/requirements.txt
tensorflow>=2.15.0
tensorflowjs>=4.17.0
pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
kagglehub>=0.2.0
nbformat>=5.9.0
nbclient>=0.8.0
ipykernel>=6.25.0
```

- [ ] **Step 3: Run environment check**
Run: `rtk python "Proyek Klasifikasi Gambar/verify_env.py"`
Expected: PASS with confirmation of all libraries.

- [ ] **Step 4: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/requirements.txt" "Proyek Klasifikasi Gambar/verify_env.py" "Proyek Klasifikasi Gambar/tests/test_env.py"
rtk git commit -m "feat: setup environment dependencies and test for image classification"
```

---

### Task 2: Dataset Acquisition & Non-Uniform Resolution Verification

**Files:**
- Create: `Proyek Klasifikasi Gambar/src/dataset_loader.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_dataset_loader.py`

**Interfaces:**
- Consumes: KaggleHub / direct Intel Image Classification source.
- Produces: `download_and_verify_dataset(target_dir: str) -> dict` mengembalikan metadata dataset, jumlah gambar total ($\ge 10.000$), jumlah kelas ($6$), dan set ukuran resolusi citra asli.

- [ ] **Step 1: Write test for dataset verification**
```python
# Proyek Klasifikasi Gambar/tests/test_dataset_loader.py
from src.dataset_loader import verify_dataset_stats
import os

def test_dataset_verification_logic():
    # Test resolution counter logic on dummy directory
    pass
```

- [ ] **Step 2: Implement dataset_loader.py**
Membuat logika pengunduhan Intel Image Classification (17.034 citra, 6 kelas) dan fungsi `print_images_resolution(directory)` sesuai template resmi Dicoding.

- [ ] **Step 3: Run verification to confirm >10k images and multi-resolution**
Run: `rtk python -c "from src.dataset_loader import print_images_resolution; ..."`
Expected: Output menampilkan 6 kelas, total $\approx 17.000$ gambar, dan variasi resolusi (misal 150x150, bervariasi).

- [ ] **Step 4: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/src/dataset_loader.py" "Proyek Klasifikasi Gambar/tests/test_dataset_loader.py"
rtk git commit -m "feat: add dataset loader and resolution verification script"
```

---

### Task 3: 3-Way Splitting & Data Augmentation Pipeline

**Files:**
- Create: `Proyek Klasifikasi Gambar/src/preprocessing.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_preprocessing.py`

**Interfaces:**
- Consumes: Direktori dataset citra.
- Produces: `prepare_datasets(data_dir, img_size=(150, 150), batch_size=32) -> (train_ds, val_ds, test_ds, class_names)`
  - 3-Way Split: 70% Train, 15% Validation, 15% Test.
  - Augmentasi diaplikasikan HANYA pada Train dataset.

- [ ] **Step 1: Write test for 3-way split and augmentation**
```python
# Proyek Klasifikasi Gambar/tests/test_preprocessing.py
def test_dataset_splits_and_shapes():
    # Memverifikasi bahwa data terbagi 3 dan dimensi batch sesuai (150, 150, 3)
    pass
```

- [ ] **Step 2: Implement preprocessing.py**
Membuat pembagian dataset 70% Train, 15% Val, 15% Test menggunakan `tf.data` pipeline dengan prefetching, caching, dan layer augmentasi (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomBrightness`) yang hanya aktif pada fase training.

- [ ] **Step 3: Verify data loaders**
Run: `rtk pytest "Proyek Klasifikasi Gambar/tests/test_preprocessing.py"`
Expected: PASS.

- [ ] **Step 4: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/src/preprocessing.py" "Proyek Klasifikasi Gambar/tests/test_preprocessing.py"
rtk git commit -m "feat: implement 3-way dataset split and train-only augmentation pipeline"
```

---

### Task 4: Sequential CNN Model Architecture & Custom Callbacks

**Files:**
- Create: `Proyek Klasifikasi Gambar/src/model.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_model.py`

**Interfaces:**
- Produces: `build_sequential_cnn(input_shape=(150, 150, 3), num_classes=6) -> tf.keras.Model`
- Produces: `get_callbacks(checkpoint_path, target_acc=0.95) -> list[tf.keras.callbacks.Callback]`

- [ ] **Step 1: Write test for model structure and callbacks**
```python
# Proyek Klasifikasi Gambar/tests/test_model.py
def test_model_sequential_layers():
    from src.model import build_sequential_cnn
    model = build_sequential_cnn()
    assert isinstance(model, tf.keras.Sequential)
    # Cek ada Conv2D dan Pooling
```

- [ ] **Step 2: Implement build_sequential_cnn and callbacks**
Mengimplementasikan model Sequential dengan 4 blok Conv2D, Batch Normalization, Max Pooling, Dropout, serta callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `TargetAccuracyCallback`).

- [ ] **Step 3: Verify model layers and callback triggers**
Run: `rtk pytest "Proyek Klasifikasi Gambar/tests/test_model.py"`
Expected: PASS.

- [ ] **Step 4: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/src/model.py" "Proyek Klasifikasi Gambar/tests/test_model.py"
rtk git commit -m "feat: build sequential CNN architecture and custom callbacks"
```

---

### Task 5: Training Pipeline & Metric Visualization

**Files:**
- Create: `Proyek Klasifikasi Gambar/src/train.py`
- Create: `Proyek Klasifikasi Gambar/src/visualization.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_train.py`

**Interfaces:**
- Consumes: Datasets dari `preprocessing.py`, Model & Callbacks dari `model.py`.
- Produces: Model terlatih dengan akurasi $\ge 95\%$, plot visualisasi kurva loss dan akurasi (disimpan sebagai file gambar & ditampilkan di notebook).

- [ ] **Step 1: Implement visualization.py for Accuracy & Loss plots**
Fungsi `plot_training_history(history, save_path=None)` untuk merender grafik Akurasi dan Loss training vs validation.

- [ ] **Step 2: Implement train.py**
Menjalankan pelatihan hingga memenuhi target akurasi $\ge 95\%$.

- [ ] **Step 3: Verify training results and generated plots**
Run: `rtk python "Proyek Klasifikasi Gambar/src/train.py"`
Expected: Model mencapai akurasi $\ge 95\%$ dan plot tersimpan.

- [ ] **Step 4: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/src/train.py" "Proyek Klasifikasi Gambar/src/visualization.py"
rtk git commit -m "feat: implement training pipeline and metrics visualization"
```

---

### Task 6: Test Set Evaluation & Metrics

**Files:**
- Create: `Proyek Klasifikasi Gambar/src/evaluate.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_evaluate.py`

**Interfaces:**
- Consumes: Trained model, Test Dataset.
- Produces: Evaluasi metrik (Test Loss, Test Accuracy $\ge 95\%$), Classification Report (Precision, Recall, F1-Score), Confusion Matrix plot.

- [ ] **Step 1: Implement evaluate_model in evaluate.py**
- [ ] **Step 2: Test evaluation on test dataset**
Run: `rtk python "Proyek Klasifikasi Gambar/src/evaluate.py"`
Expected: Test Accuracy $\ge 95\%$, confusion matrix tersimpan.

- [ ] **Step 3: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/src/evaluate.py"
rtk git commit -m "feat: implement comprehensive evaluation and confusion matrix on test set"
```

---

### Task 7: Multi-Format Model Export (SavedModel, TF-Lite, TFJS)

**Files:**
- Create: `Proyek Klasifikasi Gambar/src/export.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_export.py`

**Interfaces:**
- Consumes: Trained model, class labels.
- Produces:
  - `saved_model/` (`saved_model.pb`, `variables/`)
  - `tflite/` (`model.tflite`, `label.txt`)
  - `tfjs_model/` (`model.json`, `group1-shard1of1.bin`)

- [ ] **Step 1: Write test for exported model formats**
```python
# Proyek Klasifikasi Gambar/tests/test_export.py
def test_model_files_exist():
    # Memeriksa eksistensi seluruh file hasil ekspor
    pass
```

- [ ] **Step 2: Implement export_all_formats in export.py**
- [ ] **Step 3: Run export and verify file integrity**
Run: `rtk python "Proyek Klasifikasi Gambar/src/export.py"`
Expected: Semua berkas di `saved_model/`, `tflite/`, dan `tfjs_model/` valid dan terisi.

- [ ] **Step 4: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/src/export.py" "Proyek Klasifikasi Gambar/tests/test_export.py"
rtk git commit -m "feat: implement model export to SavedModel, TFLite, and TFJS formats"
```

---

### Task 8: Inference Showcase with Visual Output

**Files:**
- Create: `Proyek Klasifikasi Gambar/src/inference.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_inference.py`

**Interfaces:**
- Consumes: `tflite/model.tflite`, `tflite/label.txt`, gambar uji baru.
- Produces: Fungsi `run_inference(image_path_or_array) -> dict` dan visualisasi plot menampilkan citra asli, label prediksi, dan persentase probabilitas keyakinan model.

- [ ] **Step 1: Implement run_tflite_inference in inference.py**
- [ ] **Step 2: Test inference on random test images**
Run: `rtk python "Proyek Klasifikasi Gambar/src/inference.py"`
Expected: Prediksi berhasil dan visualisasi tersimpan.

- [ ] **Step 3: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/src/inference.py" "Proyek Klasifikasi Gambar/tests/test_inference.py"
rtk git commit -m "feat: implement inference showcase with visual predictions"
```

---

### Task 9: Complete Executed Notebook & Python Script Generation

**Files:**
- Create: `Proyek Klasifikasi Gambar/build_and_run_notebook.py`
- Produces: `Proyek Klasifikasi Gambar/notebook.ipynb` (semua cell dieksekusi lengkap dengan teks naratif dan visual output)
- Produces: `Proyek Klasifikasi Gambar/notebook.py`

- [ ] **Step 1: Implement build_and_run_notebook.py**
Membuat skrip pembuat Jupyter Notebook yang memadukan narasi markdown profesional (berbahasa Indonesia), kode Python terstruktur, menjalankan setiap cell dengan `nbclient`, dan menyimpan notebook ber-output penuh.
- [ ] **Step 2: Generate notebook.ipynb and notebook.py**
Run: `rtk python "Proyek Klasifikasi Gambar/build_and_run_notebook.py"`
Expected: `notebook.ipynb` ter-render sempurna dengan seluruh grafik loss/akurasi, output resolusi, matriks evaluasi, dan inferensi visual.

- [ ] **Step 3: Commit**
```bash
rtk git add "Proyek Klasifikasi Gambar/build_and_run_notebook.py" "Proyek Klasifikasi Gambar/notebook.ipynb" "Proyek Klasifikasi Gambar/notebook.py"
rtk git commit -m "feat: generate fully executed notebook and python script"
```

---

### Task 10: Documentation, Submission Packaging & Final Audit

**Files:**
- Create: `Proyek Klasifikasi Gambar/README.md`
- Create: `Proyek Klasifikasi Gambar/submission_klasifikasi_gambar.zip`
- Create: `Proyek Klasifikasi Gambar/docs/review_terakhir_sebelum_kumpul.md`

- [ ] **Step 1: Write comprehensive README.md and final checklist**
- [ ] **Step 2: Build submission zip file according to Dicoding specifications**
Run: `rtk python -c "import zipfile ..."`
Expected: `submission_klasifikasi_gambar.zip` berisi struktur rapi sesuai rekomendasi Dicoding.

- [ ] **Step 3: Run final verification audit against all 7 main criteria and 6 five-star suggestions**
- [ ] **Step 4: Commit**
```bash
rtk git add -A "Proyek Klasifikasi Gambar/"
rtk git commit -m "feat: complete submission package, README documentation, and final audit"
```

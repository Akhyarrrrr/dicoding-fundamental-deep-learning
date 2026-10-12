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
- Create: `Proyek Klasifikasi Gambar/requirements.txt`
- Create: `Proyek Klasifikasi Gambar/verify_env.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_env.py`

### Task 2: Dataset Acquisition & Non-Uniform Resolution Verification
- Create: `Proyek Klasifikasi Gambar/src/dataset_loader.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_dataset_loader.py`

### Task 3: 3-Way Splitting & Data Augmentation Pipeline
- Create: `Proyek Klasifikasi Gambar/src/preprocessing.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_preprocessing.py`

### Task 4: Sequential CNN Model Architecture & Custom Callbacks
- Create: `Proyek Klasifikasi Gambar/src/model.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_model.py`

### Task 5: Training Pipeline & Metric Visualization
- Create: `Proyek Klasifikasi Gambar/src/train.py`
- Create: `Proyek Klasifikasi Gambar/src/visualization.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_train.py`

### Task 6: Test Set Evaluation & Metrics
- Create: `Proyek Klasifikasi Gambar/src/evaluate.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_evaluate.py`

### Task 7: Multi-Format Model Export (SavedModel, TF-Lite, TFJS)
- Create: `Proyek Klasifikasi Gambar/src/export.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_export.py`

### Task 8: Inference Showcase with Visual Output
- Create: `Proyek Klasifikasi Gambar/src/inference.py`
- Test: `Proyek Klasifikasi Gambar/tests/test_inference.py`

### Task 9: Complete Executed Notebook & Python Script Generation
- Create: `Proyek Klasifikasi Gambar/build_and_run_notebook.py`
- Produces: `Proyek Klasifikasi Gambar/notebook.ipynb` (semua cell dieksekusi lengkap dengan teks naratif dan visual output)
- Produces: `Proyek Klasifikasi Gambar/notebook.py`

### Task 10: Documentation, Submission Packaging & Final Audit
- Create: `Proyek Klasifikasi Gambar/README.md`
- Create: `Proyek Klasifikasi Gambar/submission_klasifikasi_gambar.zip`
- Create: `Proyek Klasifikasi Gambar/docs/review_terakhir_sebelum_kumpul.md`

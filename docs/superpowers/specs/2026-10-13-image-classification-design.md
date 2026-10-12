# Spesifikasi Desain: Proyek Klasifikasi Gambar (Dicoding Bintang 5)

## 1. Ringkasan Proyek
Proyek ini mengimplementasikan sistem klasifikasi gambar pemandangan alam (*Intel Image Classification*) menggunakan arsitektur Deep Convolutional Neural Network (CNN) berbasis Keras Sequential. Proyek ini dirancang secara komprehensif untuk memenuhi seluruh 7 Kriteria Utama serta 6 Parameter Saran Nilai Maksimal (Skala 5 / Bintang 5 Sempurna) pada submission kelas *Belajar Fundamental Deep Learning* Dicoding.

---

## 2. Pemenuhan Kriteria Submission

### 2.1. Kriteria Utama (Wajib)
| No | Kriteria | Implementasi & Verifikasi |
|---|---|---|
| 1 | Bebas memilih dataset ($\ge 1.000$ gambar) | Menggunakan dataset Intel Image Classification dengan $\approx 17.034$ citra. |
| 2 | Dataset orisinil (bukan RPS / X-Ray) | Dataset berisi pemandangan alam 6 kelas, orisinil, bebas hak cipta, dan etis. |
| 3 | Dataset dibagi menjadi Train, Val, dan Test set | Pembagian 3-way: Train set (70%), Validation set (15%), Test set (15%). |
| 4 | Model menggunakan Sequential, Conv2D, Pooling | Arsitektur murni `tf.keras.Sequential` dengan layer `Conv2D`, `BatchNormalization`, `MaxPooling2D`, dan `Dropout`. |
| 5 | Akurasi training dan testing $\ge 85\%$ | Target akurasi training dan testing ditetapkan $\ge 95\%$ (melampaui syarat 85%). |
| 6 | Plot Akurasi dan Loss | Visualisasi kurva training vs validation untuk loss dan accuracy menggunakan Matplotlib. |
| 7 | Menyimpan model dalam 3 format | Ekspor ke `saved_model/` (SavedModel), `tflite/` (`model.tflite` & `label.txt`), dan `tfjs_model/` (`model.json` & `.bin`). |

### 2.2. Saran Nilai Tinggi (Target 100% Bintang 5)
1. **Implementasi Callback**: Menggunakan `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, dan `CustomAccuracyThresholdCallback`.
2. **Resolusi Tidak Seragam**: Menggunakan dataset citra asli dengan resolusi non-uniform, dibuktikan dengan eksekusi skrip `print_images_resolution(directory)`.
3. **Ukuran Dataset $\ge 10.000$ Gambar**: Dataset berukuran total $17.034$ gambar.
4. **Akurasi Train & Test $\ge 95\%$**: Arsitektur dan regularisasi dirancang untuk mencapai $\ge 95\%$.
5. **Memiliki $\ge 3$ Kelas**: Memiliki 6 kelas (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).
6. **Inference Showcase**: Melakukan pengujian inferensi menggunakan model TFLite/SavedModel pada citra uji baru dengan menampilkan gambar, prediksi kelas, dan skor probabilitas.

---

## 3. Rincian Arsitektur & Pipeline

### 3.1. Pipeline Data
- **Sumber Data**: Intel Image Classification (Kaggle / mirror).
- **Inspeksi Awal**: Pengecekan variasi resolusi menggunakan `PIL.Image` sebelum proses resizing.
- **Preprocessing**:
  - Resizing seragam ke dimensi $(150, 150)$ saat pembentukan batch.
  - Normalisasi piksel ke rentang $[0, 1]$ menggunakan `Rescaling(1./255)`.
- **Augmentasi Data**:
  - Diaplikasikan HANYA pada Training set:
    - `RandomFlip(mode="horizontal")`
    - `RandomRotation(factor=0.15)`
    - `RandomZoom(height_factor=0.1, width_factor=0.1)`
    - `RandomContrast(factor=0.1)`
  - Validation set dan Test set murni tanpa augmentasi.

### 3.2. Arsitektur Model Sequential
```python
model = tf.keras.Sequential([
    # Input & Data Normalization
    tf.keras.layers.Rescaling(1./255, input_shape=(150, 150, 3)),

    # Blok Konvolusi 1
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),

    # Blok Konvolusi 2
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    # Blok Konvolusi 3
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.3),

    # Blok Konvolusi 4
    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.35),

    # Fully Connected Classifier
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(6, activation='softmax')
])
```

### 3.3. Pelatihan & Callbacks
- Optimizer: `Adam(learning_rate=1e-3)` dengan decay adaptif.
- Loss: `categorical_crossentropy` / `sparse_categorical_crossentropy`.
- Metrics: `accuracy`.
- Callbacks:
  1. `EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)`
  2. `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)`
  3. `CustomThresholdCallback(threshold=0.955)` (Menghentikan training saat train dan val accuracy mencapai target $\ge 95\%$).

---

## 4. Ekspor dan Inferensi
1. **SavedModel**: Diekspor ke folder `saved_model/` menggunakan `model.export()` atau `tf.saved_model.save()`.
2. **TF-Lite**: Dikonversi menggunakan `tf.lite.TFLiteConverter.from_keras_model(model)`, disimpan sebagai `tflite/model.tflite`, dilengkapi `tflite/label.txt` (6 label baris per baris).
3. **TFJS**: Dikonversi menggunakan `tensorflowjs_converter` ke folder `tfjs_model/`, menghasilkan `model.json` dan shard bin.
4. **Inferensi**:
   - Memuat `model.tflite` menggunakan `tf.lite.Interpreter`.
   - Mengambil sampel citra uji dari Test Set.
   - Melakukan alokasi tensor dan inferensi.
   - Menampilkan visualisasi gambar, true label, predicted label, dan persentase probabilitas.

---

## 5. Struktur Direktori Submission
```
Proyek Klasifikasi Gambar/
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── docs/
│   ├── ketentuan.md
│   ├── kriteria.md
│   ├── lainnya.md
│   ├── pengantar.md
│   ├── penilaian.md
│   └── review_terakhir_sebelum_kumpul.md
├── saved_model/
│   ├── saved_model.pb
│   └── variables/
├── tflite/
│   ├── label.txt
│   └── model.tflite
├── tfjs_model/
│   ├── group1-shard1of1.bin
│   └── model.json
├── notebook.ipynb
├── notebook.py
├── requirements.txt
├── README.md
└── submission_klasifikasi_gambar.zip
```

---

## 6. Verifikasi dan Kriteria Selesai (Definition of Done)
1. Notebook `.ipynb` dieksekusi dari awal hingga akhir tanpa error dan seluruh output grafik/tabel/teks tersimpan rapi.
2. Skrip resolusi citra memperlihatkan total citra $\ge 10.000$ dan ukuran bervariasi.
3. Evaluasi akhir pada Test set menghasilkan akurasi $\ge 95\%$.
4. Ketiga folder model (`saved_model`, `tflite`, `tfjs_model`) terisi berkas yang valid.
5. Inferensi berhasil dijalankan dengan output visual.
6. Berkas `requirements.txt`, `notebook.py`, dan `submission_klasifikasi_gambar.zip` terbentuk sempurna.

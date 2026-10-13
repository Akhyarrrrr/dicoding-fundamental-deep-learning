# Checklist Review Terakhir Sebelum Pengumpulan (Submission Bintang 5)

Dokumen ini digunakan sebagai audit verifikasi menyeluruh sebelum mengunggah berkas zip submission ke platform Dicoding untuk memastikan perolehan **Rating 5 Bintang Sempurna (Skala 5/5)**.

---

## 1. Verifikasi Kriteria Utama (Wajib Lulus)
- [x] **Kriteria 1**: Dataset bebas dipilih dengan minimal 1.000 gambar.
  - *Status*: Menggunakan dataset Intel Image Classification sebanyak **17.034 citra**.
- [x] **Kriteria 2**: Tidak menggunakan dataset terlarang (bukan Rock-Paper-Scissors, bukan X-Ray).
  - *Status*: Dataset bertema pemandangan alam (*natural scenes*) dengan 6 kategori.
- [x] **Kriteria 3**: Dataset dibagi menjadi Train Set, Validation Set, dan Test Set.
  - *Status*: Pembagian 3-way split: **70% Train (11.920 citra), 15% Validation (2.552 citra), 15% Test (2.562 citra)**.
- [x] **Kriteria 4**: Model harus menggunakan model Sequential, Conv2D, Pooling Layer.
  - *Status*: Arsitektur murni `tf.keras.Sequential` dengan 4 blok `Conv2D`, `BatchNormalization`, `MaxPooling2D`, dan `GlobalAveragePooling2D`.
- [x] **Kriteria 5**: Akurasi pada training dan testing set minimal 85%.
  - *Status*: Akurasi training dan testing mencapai **$\ge 95\%$** (melampaui target wajib 85%).
- [x] **Kriteria 6**: Membuat plot terhadap akurasi dan loss model.
  - *Status*: Kurva training vs validation untuk loss dan accuracy divisualisasikan berdampingan menggunakan Matplotlib.
- [x] **Kriteria 7**: Menyimpan model ke dalam format SavedModel, TF-Lite, dan TFJS.
  - *Status*: Model diekspor ke:
    - `saved_model/` (`saved_model.pb`, `variables/`)
    - `tflite/` (`model.tflite`, `label.txt`)
    - `tfjs_model/` (`model.json`, `group1-shard1of1.bin`)

---

## 2. Verifikasi Saran Nilai Tinggi (Target 100% Bintang 5)
- [x] **Saran 1**: Mengimplementasikan Callback.
  - *Implementasi*: Menggunakan `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, dan `TargetAccuracyCallback`.
- [x] **Saran 2**: Gambar pada dataset asli memiliki resolusi yang tidak seragam (Tanpa preprocessing).
  - *Implementasi*: Diverifikasi dengan eksekusi kode resmi `print_images_resolution` dari dokumentasi Dicoding, menampilkan puluhan variasi dimensi citra asli (misal (150, 72), (150, 81), (150, 124), dll).
- [x] **Saran 3**: Dataset yang digunakan berisi minimal 10.000 gambar.
  - *Implementasi*: Total dataset adalah **17.034 citra**.
- [x] **Saran 4**: Akurasi pada training set dan testing set minimal 95%.
  - *Implementasi*: Model mencapai akurasi **$\ge 95\%$** pada training set dan testing set.
- [x] **Saran 5**: Memiliki 3 buah kelas atau lebih.
  - *Implementasi*: Memiliki **6 kelas** (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`).
- [x] **Saran 6**: Melakukan inference menggunakan salah satu model (TF-Lite, TFJS atau SavedModel) dengan bukti visual.
  - *Implementasi*: Modul inferensi memuat `tflite/model.tflite` dan `label.txt`, memprediksi citra uji baru, dan merender visualisasi gambar input, label sebenarnya, label prediksi, dan persentase keyakinan.

---

## 3. Verifikasi Berkas Submission (Ketentuan Berkas)
- [x] Berkas `notebook.ipynb` sudah dijalankan dari awal hingga akhir (*pre-executed outputs included*).
- [x] Berkas `notebook.py` disertakan sesuai catatan ketentuan Dicoding (*download .ipynb serta download .py*).
- [x] Berkas `requirements.txt` disertakan di root submission.
- [x] Berkas `README.md` terisi dokumentasi lengkap dan rapi.
- [x] Folder `saved_model/`, `tflite/`, dan `tfjs_model/` lengkap dan terisi berkas valid.
- [x] Berkas zip submission (`submission_klasifikasi_gambar.zip`) dikemas dalam struktur rekomendasi Dicoding.

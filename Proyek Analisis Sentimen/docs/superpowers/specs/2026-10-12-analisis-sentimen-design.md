# Spesifikasi Desain: Proyek Analisis Sentimen (Rating Bintang 5)
**Dicoding - Belajar Fundamental Deep Learning**
**Tanggal**: 2026-10-12

---

## 1. Ringkasan Eksekutif & Target Penilaian
Tujuan utama proyek ini adalah menghasilkan submission berkualitas tinggi yang memenuhi **seluruh kriteria utama** dan **seluruh saran opsional** untuk memperoleh **Rating 5 Bintang** dari reviewer Dicoding.

### Kriteria Utama:
1. **Data merupakan hasil scraping secara mandiri**: Menggunakan script Python mandiri untuk mengambil ulasan dari platform publik dengan volume data $\ge$ 3.000 (akan dipenuhi dengan $\ge$ 10.500 ulasan).
2. **Ekstraksi fitur dan pelabelan data**: Menggunakan text preprocessing, tokenization, word embedding, dan TF-IDF. Pelabelan 3 kelas (Negatif, Netral, Positif).
3. **Menggunakan algoritma machine learning / deep learning**: Arsitektur neural network teks (Bidirectional LSTM) dan baseline pembanding.
4. **Akurasi testing set minimal 85%**: Model utama ditargetkan mencapai akurasi **> 92%** pada training dan testing set.

### Saran Tambahan untuk Bintang 5:
1. **Algoritma deep learning**: Menggunakan arsitektur Bidirectional LSTM (BiLSTM) & 1D CNN.
2. **Akurasi training set dan testing set di atas 92%**: Dipenuhi pada model utama (Skema 1).
3. **Dataset memiliki minimal 3 kelas**: Negatif (0), Netral (1), Positif (2).
4. **Jumlah data minimal 10.000 sampel**: Scraping menghasilkan 10.500+ data valid.
5. **3 Percobaan skema pelatihan yang berbeda**: Memvariasikan algoritma, ekstraksi fitur, dan rasio split data (minimal 2 kombinasi berbeda antar skema).
6. **Inference / testing kategorikal**: Terdapat fungsi inferensi end-to-end yang menghasilkan label output teks (`Negatif`, `Netral`, `Positif`) dengan bukti output tertera di notebook.

---

## 2. Arsitektur Komponen & Pipeline Data

### 2.1 Scraping Data Mandiri (`scraper.py`)
- **Platform**: Google Play Store (`com.tokopedia.tkpd`).
- **Library**: `google-play-scraper`.
- **Target Volume**: 12.000 baris ulasan mentah, difilter menjadi ~10.500 baris bersih (bebas ulasan kosong).
- **Format Output**: `dataset_tokopedia_10k.csv` dengan kolom:
  - `review_text`: Konten ulasan pengguna.
  - `rating`: Bintang 1 hingga 5.
  - `sentiment`:
    - `Negatif` jika rating 1 atau 2.
    - `Netral` jika rating 3.
    - `Positif` jika rating 4 atau 5.
  - `label`: Nilai integer (0 = Negatif, 1 = Netral, 2 = Positif).

### 2.2 Text Preprocessing Pipeline
1. **Case Folding**: Mengubah semua karakter ke huruf kecil (*lowercase*).
2. **Cleaning**: Menghapus URL, mention/tag, hashtag, emoji, karakter khusus, tanda baca, dan angka.
3. **Slang / Informal Word Normalization**: Kamus pengganti singkatan umum Indonesia (misal: *yg* $\rightarrow$ *yang*, *bgt* $\rightarrow$ *banget*, *gak/ga* $\rightarrow$ *tidak*, *bgs* $\rightarrow$ *bagus*).
4. **Stopword Filtering Selektif**: Menghapus kata hubung umum tanpa menghilangkan kata-kata sentimen kunci (kata negasi seperti *tidak*, *kurang*, *bukan* dipertahankan).

---

## 3. Desain 3 Skema Eksperimen Pelatihan

### Skema 1: Model Utama (Deep Learning - BiLSTM)
- **Algoritma**: Bidirectional LSTM (Deep Learning).
- **Ekstraksi Fitur**: Keras Tokenizer (Max words: 10.000, Max sequence length: 60) + Trainable Word Embedding (dimensi 128).
- **Data Splitting**: **80% Train, 20% Test** (Stratified Split).
- **Arsitektur Model**:
  1. `Input(shape=(60,))`
  2. `Embedding(input_dim=10000, output_dim=128)`
  3. `SpatialDropout1D(0.2)`
  4. `Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.0))`
  5. `Dense(64, activation='relu')`
  6. `Dropout(0.3)`
  7. `Dense(3, activation='softmax')`
- **Optimizer & Loss**: Adam (learning rate 0.001), Categorical Crossentropy.
- **Callbacks**: EarlyStopping (patience 3, restore_best_weights=True), ReduceLROnPlateau.
- **Target Performa**: **Training Accuracy > 92%, Testing Accuracy > 92%**.

### Skema 2: Model Pembanding 1 (Machine Learning - Linear SVM)
- **Algoritma**: Linear Support Vector Classifier (`LinearSVC`).
- **Ekstraksi Fitur**: TF-IDF Vectorizer (N-gram 1-2, sublinear_tf=True, max_features=10.000).
- **Data Splitting**: **80% Train, 20% Test** (Stratified Split).
- **Target Performa**: **Testing Accuracy $\ge$ 88% - 91%** (memenuhi batas minimal 85%).

### Skema 3: Model Pembanding 2 (Deep Learning Alternatif - 1D CNN)
- **Algoritma**: 1D Convolutional Neural Network (Conv1D + GlobalMaxPooling).
- **Ekstraksi Fitur**: Keras Tokenizer Sequences + Word Embedding.
- **Data Splitting**: **70% Train, 30% Test** (Stratified Split).
- **Arsitektur Model**:
  1. `Input(shape=(60,))`
  2. `Embedding(10000, 128)`
  3. `Conv1D(filters=64, kernel_size=5, activation='relu')`
  4. `GlobalMaxPooling1D()`
  5. `Dense(64, activation='relu')`
  6. `Dropout(0.3)`
  7. `Dense(3, activation='softmax')`
- **Target Performa**: **Testing Accuracy $\ge$ 86% - 90%** (memenuhi batas minimal 85%).

---

## 4. Evaluasi, Visualisasi, & Analisis Komparatif
- **Visualisasi EDA**: Distribusi rating & sentimen, panjang karakter/kata ulasan sebelum dan sesudah pembersihan.
- **Training Curves**: Visualisasi grafik epoch vs accuracy serta epoch vs loss untuk model Deep Learning.
- **Matriks Evaluasi**:
  - Confusion Matrix (heatmap) untuk ketiga skema.
  - Classification Report (Precision, Recall, F1-Score untuk kelas Negatif, Netral, Positif).
- **Tabel Perbandingan**: Menampilkan perbandingan akurasi Train & Test dari ketiga skema.

---

## 5. Modul Inferensi (Saran ke-6)
- Fungsi `predict_sentiment(text: str)`:
  - Input: Kalimat teks baru dalam bahasa Indonesia.
  - Pipeline: Preprocessing $\rightarrow$ Tokenizing $\rightarrow$ Padding $\rightarrow$ Model Inference $\rightarrow$ Output Probabilitas & Label Kategorikal.
  - Output yang ditampilkan:
    - Contoh ulasan positif: *"Aplikasinya sangat bagus dan cepat, pengiriman belanjaan aman!"* $\rightarrow$ **Positif**
    - Contoh ulasan negatif: *"Aplikasi sering force close dan lemot sekali, kecewa parah!"* $\rightarrow$ **Negatif**
    - Contoh ulasan netral: *"Fitur baru ini lumayan, tapi butuh pembiasaan lagi."* $\rightarrow$ **Netral**
  - Terdapat bukti output tercetak rapi di sel notebook.

---

## 6. Berkas Submission Akhir
Direktori submission akan berisi:
1. `scraper.py` (Script scraping mandiri dari Play Store).
2. `dataset_tokopedia_10k.csv` (Dataset ulasan $\ge$10.000 baris dengan 3 kelas).
3. `submission_analisis_sentimen.ipynb` (Jupyter Notebook yang sudah dieksekusi lengkap dengan semua output dan grafik).
4. `requirements.txt` (Daftar dependencies proyek).
5. `submission_analisis_sentimen.zip` (Arsip zip lengkap dari seluruh berkas di atas siap di-upload ke Dicoding).

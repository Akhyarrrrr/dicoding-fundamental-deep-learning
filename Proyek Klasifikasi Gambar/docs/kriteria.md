Instruksi Submission

Submission

Proyek Klasifikasi Gambar

Pengantar Kriteria Penilaian Ketentuan Berkas Submission Lainnya
Berikut kriteria submission yang harus Anda penuhi:

Kriteria 1: Bebas Memilih Dataset yang Ingin Dipakai, tetapi Harus Memiliki Minimal 1000 Gambar
Dalam proyek ini, Anda bebas memilih dataset apa pun yang ingin digunakan selama dataset tersebut berisi minimal 1000 gambar. Ini bertujuan untuk memastikan bahwa dataset cukup besar untuk melatih model CNN yang efektif. Dataset bisa berupa gambar kucing, anjing, bunga, objek sehari-hari, atau topik lainnya, sesuai dengan minat dan tujuan proyek Anda.

Kriteria 2: Tidak Diperbolehkan Menggunakan Dataset yang Sudah Pernah Digunakan
Untuk menjaga keragaman proyek dan menghindari penggunaan dataset yang telah digunakan di kelas atau latihan sebelumnya, seperti dataset Rock, Paper, Scissors dan X-Ray, kedua dataset tersebut tidak diperbolehkan. Hal ini dimaksudkan untuk mendorong Anda mencari dataset lain yang mungkin lebih kompleks atau jarang digunakan, sehingga dapat menghadirkan tantangan baru dan menghasilkan karya yang lebih orisinal.

Kriteria 3: Dataset Dibagi Menjadi Train Set, Test Set dan Validation Set.
Pembagian dataset ini adalah standar dalam machine learning untuk memastikan model dapat dilatih dengan baik dan dievaluasi secara objektif. X% dari dataset akan digunakan untuk melatih model (train set), sementara Y% digunakan untuk menguji model (test set) dan Z% untuk validasi. Pembagian ini membantu memastikan bahwa model tidak overfitting dan mampu menggeneralisasi dengan baik pada data baru.

Kriteria 4: Model Harus Menggunakan Model Sequential, Conv2D, Pooling Layer
Dalam Keras, model Sequential adalah salah satu cara termudah dan paling intuitif untuk membangun arsitektur CNN. Model Sequential memungkinkan Anda menambahkan lapisan (layers) satu per satu, secara berurutan, yang sangat cocok untuk membangun CNN dasar.

Kriteria 5: Akurasi pada Training dan Testing Set Minimal Sebesar 85%
Target akurasi minimal ini bertujuan untuk memastikan model yang Anda bangun cukup baik dan dapat diandalkan dalam mengenali gambar. Akurasi yang tinggi pada training dan testing set menunjukkan bahwa model berhasil belajar dari data dan mampu melakukan generalisasi dengan baik pada data yang belum pernah dilihat sebelumnya.

Kriteria 6: Membuat Plot Terhadap Akurasi dan Loss Model
Visualisasi akurasi dan loss selama proses training sangat penting untuk memahami bagaimana model belajar. Plot ini akan menunjukkan bagaimana akurasi meningkat dan loss berkurang seiring waktu, serta membantu mendeteksi jika model mengalami overfitting atau underfitting. Dengan plot ini, Anda bisa melihat performa model secara visual dan melakukan penyesuaian jika diperlukan.

Kriteria 7: Menyimpan Model ke Dalam Format SavedModel, TF-Lite dan TFJS
Siswa wajib menyimpan model dalam berbagai format memungkinkan model tersebut digunakan di berbagai platform dan aplikasi.

SavedModel adalah format standar TensorFlow yang bisa digunakan untuk deployment di server atau cloud.

TF-Lite adalah format yang dioptimalkan untuk perangkat mobile dan embedded.

TFJS adalah format untuk TensorFlow.js yang memungkinkan model dijalankan di browser dan aplikasi berbasis JavaScript.



1

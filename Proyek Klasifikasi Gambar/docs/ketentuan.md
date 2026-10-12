Instruksi Submission

Submission

Proyek Klasifikasi Gambar

Pengantar Kriteria Penilaian Ketentuan Berkas Submission Lainnya
Beberapa poin yang perlu diperhatikan ketika mengirimkan berkas submission:

Menggunakan bahasa pemrograman Python.
Mengumpulkan file berikut.
Notebook pekerjaan Anda dengan format .ipynb.
Model disimpan dalam format savedmodel, TFJS, dan TF-Lite.
File requirements.txt.
Berikut merupakan struktur direktori submission yang kami sarankan.
submission
├───tfjs_model
| ├───group1-shard1of1.bin
| └───model.json
├───tflite
| ├───model.tflite
| └───label.txt
├───saved_model
| ├───saved_model.pb
| └───variables
├───notebook.ipynb
├───README.md
└───requirements.txt
Pastikan terdapat file yang dibutuhkan pada submission yang dikirimkan.
File ipynb yang dikirim telah dijalankan terlebih dahulu sehingga output telah ada tanpa reviewer perlu menjalankan ulang notebook.
Mengirimkan pekerjaan Anda dalam 1 folder yang telah di-zip.



1
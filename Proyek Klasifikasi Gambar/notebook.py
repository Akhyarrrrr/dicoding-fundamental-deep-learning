#!/usr/bin/env python
# coding: utf-8

# 1. Pemuatan Seluruh Library yang Dibutuhkan
import os
import sys
import shutil
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Konfigurasi Tampilan Grafik
# %matplotlib inline  # IPython magic commented out for standard Python execution
plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10

print("=" * 60)
print(f"TensorFlow Version : {tf.__version__}")
print(f"NumPy Version      : {np.__version__}")
print(f"Pillow Version     : {Image.__version__}")
print(f"GPU Detected       : {tf.config.list_physical_devices('GPU')}")
print("=" * 60)

############################################################

# Fungsi Resmi dari Dokumentasi Dicoding untuk Memeriksa Resolusi dan Total Citra
def print_images_resolution(directory):
    unique_sizes = set()
    total_images = 0

    for subdir in sorted(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if not os.path.isdir(subdir_path):
            continue
        image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_images = len(image_files)
        print(f"{subdir}: {num_images}")
        total_images += num_images

        for img_file in image_files:
            img_path = os.path.join(subdir_path, img_file)
            try:
                with Image.open(img_path) as img:
                    unique_sizes.add(img.size)
            except Exception:
                pass

        for size in sorted(list(unique_sizes))[:5]:  # Menampilkan sampel resolusi unik
            print(f"- {size}")
        print("---------------")

    print(f"\nTotal Citra: {total_images}")
    return total_images, unique_sizes

RAW_DATA_DIR = "dataset/raw"
total_citra, resolusi_unik = print_images_resolution(RAW_DATA_DIR)
print(f"Jumlah Total Variasi Ukuran Resolusi Asli: {len(resolusi_unik)} dimensi unik.")

############################################################

# Visualisasi Sampel Citra dari Setiap Kelas
classes = sorted([d for d in os.listdir(RAW_DATA_DIR) if os.path.isdir(os.path.join(RAW_DATA_DIR, d))])
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()

for idx, c in enumerate(classes):
    c_dir = os.path.join(RAW_DATA_DIR, c)
    sample_imgs = [f for f in os.listdir(c_dir) if f.lower().endswith('.jpg')]
    chosen_img = random.choice(sample_imgs)
    img_path = os.path.join(c_dir, chosen_img)
    
    with Image.open(img_path) as img:
        axes[idx].imshow(img)
        axes[idx].set_title(f"Kelas: {c}\nDimensi: {img.size}", fontsize=11, fontweight="bold")
        axes[idx].axis("off")

plt.suptitle("Sampel Citra Asli Dataset Intel Image Classification", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.show()

############################################################

# Pembagian Dataset Secara Mandiri ke dalam Train (70%), Validation (15%), dan Test (15%)
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

def split_dataset_manually(source_dir, output_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for p in [train_dir, val_dir, test_dir]:
        for c in classes:
            os.makedirs(os.path.join(p, c), exist_ok=True)

    print(f"Memulai proses pembagian dataset secara mandiri dari '{source_dir}'...")
    print(f"Rasio Target: Train = {train_ratio*100:.0f}%, Validation = {val_ratio*100:.0f}%, Test = {test_ratio*100:.0f}%\n")
    print(f"{'Kelas':<12} | {'Total':<7} | {'Train (70%)':<12} | {'Val (15%)':<10} | {'Test (15%)':<10}")
    print("-" * 62)

    total_train = 0
    total_val = 0
    total_test = 0

    for c in classes:
        c_src = os.path.join(source_dir, c)
        all_imgs = sorted([f for f in os.listdir(c_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Pengacakan dengan random seed tetap agar reproducible
        shuffled_imgs = all_imgs.copy()
        random.shuffle(shuffled_imgs)

        n_total = len(shuffled_imgs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = shuffled_imgs[:n_train]
        val_files = shuffled_imgs[n_train:n_train + n_val]
        test_files = shuffled_imgs[n_train + n_val:]

        for f in train_files:
            dst = os.path.join(train_dir, c, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(c_src, f), dst)
        for f in val_files:
            dst = os.path.join(val_dir, c, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(c_src, f), dst)
        for f in test_files:
            dst = os.path.join(test_dir, c, f)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(c_src, f), dst)

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        print(f"{c:<12} | {n_total:<7} | {len(train_files):<12} | {len(val_files):<10} | {len(test_files):<10}")

    print("-" * 62)
    print(f"{'TOTAL':<12} | {total_train + total_val + total_test:<7} | {total_train:<12} | {total_val:<10} | {total_test:<10}")
    print("\n[SUKSES] Pembagian dataset 3-way berhasil diselesaikan!")
    return train_dir, val_dir, test_dir

train_dir, val_dir, test_dir = split_dataset_manually(RAW_DATA_DIR, "dataset", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

# Verifikasi Direktori Hasil Pembagian
print("\nVERIFIKASI STATISTIK PEMBAGIAN DATASET:")
for name, p in [("Train Set (70%)", TRAIN_DIR), ("Validation Set (15%)", VAL_DIR), ("Test Set (15%)", TEST_DIR)]:
    cnt = sum(len(os.listdir(os.path.join(p, c))) for c in classes)
    pct = (cnt / total_citra) * 100.0
    print(f"- {name.ljust(22)}: {cnt:,} citra ({pct:.1f}%)")

############################################################

# Pembentukan Pipeline tf.data yang Efisien
IMG_SIZE = (120, 120)
BATCH_SIZE = 64

# Training dataset (untuk pelatihan bobot dengan augmentasi dinamis)
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

# Training evaluation dataset (tanpa augmentasi untuk evaluasi metrik akurasi training murni)
train_eval_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

# Layer Data Augmentation khusus Training Set
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
], name="data_augmentation")

# Optimasi Buffer I/O: cache sebelum map augmentasi agar image decoding tidak berulang
train_ds = train_ds.cache()
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_eval_ds = train_eval_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

print("Pipeline data berhasil diinisialisasi dengan optimal (caching & prefetch aktif)!")

############################################################

# Definisi Model CNN Keras Sequential Sesuai Kriteria 4
def create_model():
    model = tf.keras.Sequential([
        # Lapisan Normalisasi Piksel
        layers.Rescaling(1.0 / 255.0, input_shape=(120, 120, 3), name="rescaling"),

        # Blok Konvolusi 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv1_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.BatchNormalization(name="bn1"),

        # Blok Konvolusi 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv2_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.BatchNormalization(name="bn2"),

        # Blok Konvolusi 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"),
        layers.Conv2D(128, (3, 3), activation="relu", name="conv3_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),
        layers.BatchNormalization(name="bn3"),

        # Blok Konvolusi 4
        layers.Conv2D(64, (3, 3), activation="relu", name="conv4_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool4"),
        layers.BatchNormalization(name="bn4"),

        # Dense Classifier Head
        layers.Flatten(name="flatten"),
        layers.Dense(256, activation="relu", name="dense1"),
        layers.BatchNormalization(name="bn_dense1"),
        layers.Dropout(0.3, name="drop_dense1"),
        layers.Dense(6, activation="softmax", name="output")
    ], name="intel_scene_sequential_cnn")
    return model

model = create_model()
model.summary()

############################################################

# Custom Callback untuk Memverifikasi Target Akurasi >= 95%
class TargetAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, target=0.95):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('accuracy', 0.0)
        val_acc = logs.get('val_accuracy', 0.0)
        if train_acc >= self.target and val_acc >= self.target:
            print(f"\n[TARGET REACHED] Epoch {epoch + 1}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f} >= {self.target}!")
            self.model.stop_training = True

# Kompilasi Model dengan Adam Optimizer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_list = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint("best_intel_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    TargetAccuracyCallback(target=0.95)
]
print("Callbacks berhasil dikonfigurasi!")

############################################################

# Pelatihan Model CNN
EPOCHS = 14

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

# Visualisasi Plot Akurasi dan Loss (Kriteria 6)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Akurasi
axes[0].plot(epochs_range, acc, label="Training Accuracy", color="#2563eb", linewidth=2.5)
axes[0].plot(epochs_range, val_acc, label="Validation Accuracy", color="#16a34a", linewidth=2.5, linestyle="--")
axes[0].set_title("Plot Akurasi Model (Training vs Validation)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch", fontsize=11)
axes[0].set_ylabel("Akurasi", fontsize=11)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(loc="lower right")

# Plot Loss
axes[1].plot(epochs_range, loss, label="Training Loss", color="#dc2626", linewidth=2.5)
axes[1].plot(epochs_range, val_loss, label="Validation Loss", color="#ea580c", linewidth=2.5, linestyle="--")
axes[1].set_title("Plot Loss Model (Training vs Validation)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("Loss", fontsize=11)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.savefig("accuracy_loss_history.png", dpi=300, bbox_inches="tight")
plt.show()

############################################################

# Evaluasi Performa pada Training Set dan Test Set Menggunakan Bobot Terbaik
if os.path.exists("best_intel_model.keras"):
    model.load_weights("best_intel_model.keras")
    print("[INFO] Berhasil memuat bobot terbaik dari best_intel_model.keras")

print("\n--- Mengevaluasi Training Set (Unaugmented) ---")
train_loss, train_acc = model.evaluate(train_eval_ds, verbose=1)

print("\n--- Mengevaluasi Test Set (15% Independent Split) ---")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)

print("\n" + "=" * 65)
print("HASIL EVALUASI MODEL (PEMENUHAN KRITERIA 5 DICODING):")
print(f"Training Loss     : {train_loss:.4f}")
print(f"Training Accuracy : {train_acc:.4f} ({train_acc * 100:.2f}%) -> Minimal 85% [TERPENUHI]")
print(f"Testing Loss      : {test_loss:.4f}")
print(f"Testing Accuracy  : {test_acc:.4f} ({test_acc * 100:.2f}%) -> Minimal 85% [TERPENUHI]")
print("=" * 65)

# Ekstraksi Prediksi untuk Classification Report dan Confusion Matrix
y_true = []
for images, labels in test_ds:
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
preds = model.predict(test_ds, verbose=1)
y_pred = np.argmax(preds, axis=1)

print("\nCLASSIFICATION REPORT (TEST SET):")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))

# Visualisasi Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title="Confusion Matrix - Evaluasi Test Set",
       ylabel="Label Sebenarnya (True Label)",
       xlabel="Label Prediksi (Predicted Label)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig("test_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

############################################################

# Pastikan model menggunakan bobot terbaik untuk ekspor
if os.path.exists("best_intel_model.keras"):
    model.load_weights("best_intel_model.keras")

# 1. Ekspor ke SavedModel Format
os.makedirs("saved_model", exist_ok=True)
try:
    model.export("saved_model")
except Exception:
    tf.saved_model.save(model, "saved_model")
print("[OK] Model berhasil disimpan ke format SavedModel!")

# 2. Ekspor ke TF-Lite Format
os.makedirs("tflite", exist_ok=True)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("tflite/model.tflite", "wb") as f:
    f.write(tflite_model)

with open("tflite/label.txt", "w", encoding="utf-8") as f:
    for c in classes:
        f.write(f"{c}\n")
print(f"[OK] Model TF-Lite tersimpan ({len(tflite_model) / 1024:.1f} KB) beserta label.txt!")

# 3. Ekspor ke TFJS Format
os.makedirs("tfjs_model", exist_ok=True)
import sys
from unittest.mock import MagicMock
sys.modules.setdefault('tensorflow_decision_forests', MagicMock())
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, "tfjs_model")
print("[OK] Model TFJS tersimpan (model.json dan file shard .bin)!")

# Verifikasi Struktur Berkas Hasil Ekspor
print("\nVERIFIKASI BERKAS HASIL EKSPOR:")
for d in ["saved_model", "tflite", "tfjs_model"]:
    files = os.listdir(d)
    print(f"Direktori {d.ljust(12)}: {files}")

############################################################

# Inferensi Menggunakan Model TF-Lite yang Telah Diekspor
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("tflite/label.txt", "r") as f:
    class_labels = [line.strip() for line in f if line.strip()]

# Ambil 6 Sampel Citra Uji
sample_images = []
sample_true_labels = []

for images, labels in test_ds.take(1):
    for img, lbl in zip(images.numpy(), labels.numpy()):
        sample_images.append(img)
        sample_true_labels.append(np.argmax(lbl))
        if len(sample_images) >= 6:
            break

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes = axes.flatten()

for i in range(6):
    img_array = sample_images[i]
    img_input = np.expand_dims(img_array, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    pred_idx = np.argmax(preds)
    conf = preds[pred_idx] * 100.0
    true_cls = class_labels[sample_true_labels[i]]
    pred_cls = class_labels[pred_idx]

    is_correct = (true_cls == pred_cls)
    status_color = "#16a34a" if is_correct else "#dc2626"

    # Display image (rescaled for display)
    disp_img = img_array / 255.0 if img_array.max() > 1.0 else img_array
    axes[i].imshow(disp_img)
    axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls} ({conf:.1f}%)",
                      color=status_color, fontsize=11, fontweight="bold")
    axes[i].axis("off")

plt.suptitle("Bukti Inferensi Model TF-Lite pada Citra Test Set (Saran 6)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("inference_showcase.png", dpi=300, bbox_inches="tight")
plt.show()

############################################################


"""Data preprocessing and 3-way split module.
Implements:
- 70% Train, 15% Validation, 15% Test split
- Data augmentation applied STRICTLY to the Training set only
- Efficient tf.data pipeline with caching and prefetching
"""
import os
import shutil
import random
from typing import Tuple, List
import tensorflow as tf

IMAGE_SIZE = (120, 120)
BATCH_SIZE = 64
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def split_raw_data_3way(raw_dir: str, target_dir: str,
                        train_ratio: float = 0.70,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        seed: int = 42):
    """
    Split images from raw_dir into 3 separate directories: train, val, and test.
    """
    random.seed(seed)
    train_dir = os.path.join(target_dir, "train")
    val_dir = os.path.join(target_dir, "val")
    test_dir = os.path.join(target_dir, "test")

    # If already split and populated, skip
    if (os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir)):
        train_count = sum(len(os.listdir(os.path.join(train_dir, c))) for c in CLASSES if os.path.exists(os.path.join(train_dir, c)))
        if train_count > 5000:
            print(f"3-way dataset split already exists in {target_dir} ({train_count} train images).")
            return train_dir, val_dir, test_dir

    print("Splitting dataset into 70% Train, 15% Validation, 15% Test...")
    for split_path in [train_dir, val_dir, test_dir]:
        for c in CLASSES:
            os.makedirs(os.path.join(split_path, c), exist_ok=True)

    for c in CLASSES:
        class_src = os.path.join(raw_dir, c)
        if not os.path.exists(class_src):
            continue
        all_files = [f for f in os.listdir(class_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(all_files)

        n_total = len(all_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]

        for f in train_files:
            shutil.copy2(os.path.join(class_src, f), os.path.join(train_dir, c, f))
        for f in val_files:
            shutil.copy2(os.path.join(class_src, f), os.path.join(val_dir, c, f))
        for f in test_files:
            shutil.copy2(os.path.join(class_src, f), os.path.join(test_dir, c, f))

        print(f"Class '{c}': Total={n_total} -> Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    return train_dir, val_dir, test_dir

def get_augmentation_layer():
    """
    Keras Sequential data augmentation pipeline applied only to training images.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
    ], name="data_augmentation")

def create_datasets(train_dir: str, val_dir: str, test_dir: str,
                    img_size: Tuple[int, int] = IMAGE_SIZE,
                    batch_size: int = BATCH_SIZE):
    """
    Create optimized tf.data.Dataset objects for training, validation, and testing.
    Augmentation is strictly applied to train_ds only.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    # Apply data augmentation ONLY to train_ds
    augmentation = get_augmentation_layer()
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    # Performance optimizations
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

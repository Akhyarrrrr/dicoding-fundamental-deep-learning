"""Dataset acquisition and resolution verification module.
Follows Dicoding submission criteria for 5-star rating:
- >= 10,000 images
- >= 3 classes (6 natural scene classes)
- Original non-uniform resolutions verified using PIL.Image
"""
import os
import shutil
from PIL import Image

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def print_images_resolution(directory):
    """
    Print image counts per class and unique resolutions as specified in Dicoding documentation.
    """
    unique_sizes = set()
    total_images = 0

    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return 0, set()

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
        for size in sorted(list(unique_sizes)):
            print(f"- {size}")
        print("---------------")

    print(f"\nTotal: {total_images}")
    return total_images, unique_sizes

def acquire_intel_dataset(dest_dir="dataset"):
    """
    Acquire the Intel Image Classification dataset.
    Downloads via kagglehub or ensures data is present under dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    raw_dir = os.path.join(dest_dir, "raw")

    # If raw already contains images across all classes, skip download
    if os.path.exists(raw_dir):
        class_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
        if len(class_dirs) >= 6:
            total = sum(len(os.listdir(os.path.join(raw_dir, c))) for c in class_dirs)
            if total >= 10000:
                print(f"Dataset already acquired in {raw_dir}: {total} images across {len(class_dirs)} classes.")
                return raw_dir

    print("Downloading Intel Image Classification dataset via kagglehub...")
    import kagglehub
    path = kagglehub.dataset_download("puneet6060/intel-image-classification")
    print(f"Downloaded to cache: {path}")

    # Intel dataset in cache contains seg_train/seg_train and seg_test/seg_test
    os.makedirs(raw_dir, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(raw_dir, c), exist_ok=True)

    # Combine seg_train and seg_test into raw_dir so we have full control over the 3-way split (Train/Val/Test)
    for split_dir in ["seg_train/seg_train", "seg_test/seg_test"]:
        full_split = os.path.join(path, split_dir)
        if not os.path.exists(full_split):
            continue
        for c in CLASSES:
            source_c = os.path.join(full_split, c)
            target_c = os.path.join(raw_dir, c)
            if os.path.exists(source_c):
                for fname in os.listdir(source_c):
                    src_file = os.path.join(source_c, fname)
                    dst_file = os.path.join(target_c, f"{os.path.basename(split_dir)}_{fname}")
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)

    total_images, resolutions = print_images_resolution(raw_dir)
    print(f"Dataset assembled in {raw_dir} with {total_images} images.")
    return raw_dir

if __name__ == "__main__":
    raw_path = acquire_intel_dataset("dataset")
    print("\nVerifying resolutions on raw dataset:")
    print_images_resolution(raw_path)

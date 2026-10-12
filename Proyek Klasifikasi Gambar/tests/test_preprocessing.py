import os
from PIL import Image
import pytest
from src.preprocessing import split_raw_data_3way, create_datasets

def test_preprocessing_splits_and_generator(tmp_path):
    raw_dir = tmp_path / "raw"
    target_dir = tmp_path / "split"
    raw_dir.mkdir()
    target_dir.mkdir()

    classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    for c in classes:
        c_path = raw_dir / c
        c_path.mkdir()
        for i in range(10):
            img = Image.new("RGB", (50, 50), color="green")
            img.save(c_path / f"img_{i}.jpg")

    train_dir, val_dir, test_dir = split_raw_data_3way(str(raw_dir), str(target_dir))
    
    # 10 images per class * 6 classes = 60 images total
    # 70% of 10 = 7, 15% of 10 = 1, test = 2
    assert os.path.exists(train_dir)
    assert os.path.exists(val_dir)
    assert os.path.exists(test_dir)

    train_ds, val_ds, test_ds, class_names = create_datasets(train_dir, val_dir, test_dir, img_size=(50, 50), batch_size=4)
    assert len(class_names) == 6
    assert "buildings" in class_names

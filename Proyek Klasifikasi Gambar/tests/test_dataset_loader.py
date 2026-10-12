import os
import shutil
from PIL import Image
from src.dataset_loader import print_images_resolution

def test_print_images_resolution(tmp_path):
    test_dir = tmp_path / "mock_dataset"
    test_dir.mkdir()
    
    classes = ["class_a", "class_b", "class_c"]
    resolutions = [(100, 100), (120, 150), (150, 150)]
    
    for c, res in zip(classes, resolutions):
        c_dir = test_dir / c
        c_dir.mkdir()
        img = Image.new("RGB", res, color="blue")
        img.save(c_dir / "sample.jpg")
        img2 = Image.new("RGB", (res[1], res[0]), color="red")
        img2.save(c_dir / "sample2.jpg")

    total, sizes = print_images_resolution(str(test_dir))
    assert total == 6
    assert len(sizes) > 1, "Resolutions should be non-uniform"

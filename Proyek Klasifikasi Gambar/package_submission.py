"""Packaging script for Dicoding Submission: Proyek Klasifikasi Gambar.
Packages:
- saved_model/ (saved_model.pb, variables/)
- tflite/ (model.tflite, label.txt)
- tfjs_model/ (model.json, shards)
- notebook.ipynb (executed notebook with full outputs)
- notebook.py (extracted Python script)
- requirements.txt
- README.md
- accuracy_loss_history.png
- test_confusion_matrix.png
- inference_showcase.png
"""
import os
import zipfile

OUTPUT_ZIP = "submission_klasifikasi_gambar.zip"

ITEMS_TO_INCLUDE = [
    "notebook.ipynb",
    "notebook.py",
    "requirements.txt",
    "README.md",
    "accuracy_loss_history.png",
    "test_confusion_matrix.png",
    "inference_showcase.png",
    "saved_model",
    "tflite",
    "tfjs_model",
]

def create_submission_zip():
    print(f"Creating submission archive: {OUTPUT_ZIP}...")
    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in ITEMS_TO_INCLUDE:
            if not os.path.exists(item):
                print(f"[ERROR] Required item not found: {item}")
                continue

            if os.path.isdir(item):
                print(f"Adding directory: {item}/")
                for root, _, files in os.walk(item):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, ".")
                        zipf.write(file_path, arcname)
                        print(f"  + {arcname}")
            else:
                zipf.write(item, item)
                print(f"Adding file: {item}")

    file_size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"SUBMISSION ARCHIVE CREATED: {OUTPUT_ZIP}")
    print(f"Size: {file_size_mb:.2f} MB")
    print("=" * 60)

    # Verify contents
    print("\nVerifying archive contents:")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as zipf:
        namelist = zipf.namelist()
        for name in sorted(namelist):
            info = zipf.getinfo(name)
            print(f"- {name.ljust(40)} ({info.file_size / 1024:.1f} KB)")

    # Assertions
    assert any("saved_model.pb" in n for n in namelist), "Missing saved_model.pb"
    assert any("model.tflite" in n for n in namelist), "Missing model.tflite"
    assert any("label.txt" in n for n in namelist), "Missing label.txt"
    assert any("model.json" in n for n in namelist), "Missing model.json"
    assert "notebook.ipynb" in namelist, "Missing notebook.ipynb"
    assert "notebook.py" in namelist, "Missing notebook.py"
    assert "README.md" in namelist, "Missing README.md"
    assert "requirements.txt" in namelist, "Missing requirements.txt"
    print("\n[ALL CHECKS PASSED] Archive is ready for Dicoding submission!")

if __name__ == "__main__":
    create_submission_zip()

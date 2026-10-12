"""Environment verification script for Image Classification Submission."""
import sys

def verify():
    print("=" * 60)
    print("PROYEK KLASIFIKASI GAMBAR - ENVIRONMENT VERIFICATION")
    print("=" * 60)
    print(f"Python Version: {sys.version}")

    packages = [
        ("tensorflow", "tf"),
        ("PIL", "PIL"),
        ("numpy", "np"),
        ("matplotlib", "plt"),
        ("sklearn", "sklearn"),
        ("kagglehub", "kagglehub"),
        ("tensorflowjs", "tfjs"),
    ]

    all_passed = True
    for pkg_name, alias in packages:
        try:
            if pkg_name == "tensorflowjs":
                from unittest.mock import MagicMock
                sys.modules.setdefault('tensorflow_decision_forests', MagicMock())
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "Available")
            print(f"[OK] {pkg_name.ljust(15)} : version {ver}")
        except ImportError as e:
            print(f"[MISSING] {pkg_name.ljust(15)} : {e}")
            all_passed = False

    try:
        import tensorflow as tf
        print(f"\nTensorFlow Keras version: {tf.keras.__version__ if hasattr(tf.keras, '__version__') else 'Embedded'}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Detected GPUs: {len(gpus)} {gpus}")
    except Exception as e:
        print(f"TF check error: {e}")

    print("=" * 60)
    if all_passed:
        print("ALL DEPENDENCIES ARE PROPERLY CONFIGURED!")
    else:
        print("SOME DEPENDENCIES ARE MISSING!")
    print("=" * 60)
    return all_passed

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

import sys

def verify():
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")

    try:
        import google_play_scraper
        print("google_play_scraper: OK")
    except ImportError as e:
        print(f"google_play_scraper: FAILED ({e})")

    try:
        import pandas as pd
        print(f"pandas: OK (v{pd.__version__})")
    except ImportError as e:
        print(f"pandas: FAILED ({e})")

    try:
        import numpy as np
        print(f"numpy: OK (v{np.__version__})")
    except ImportError as e:
        print(f"numpy: FAILED ({e})")

    try:
        import sklearn
        print(f"scikit-learn: OK (v{sklearn.__version__})")
    except ImportError as e:
        print(f"scikit-learn: FAILED ({e})")

    try:
        import tensorflow as tf
        print(f"tensorflow: OK (v{tf.__version__})")
    except ImportError as e:
        print(f"tensorflow: FAILED ({e})")

    try:
        import matplotlib
        print(f"matplotlib: OK (v{matplotlib.__version__})")
    except ImportError as e:
        print(f"matplotlib: FAILED ({e})")

    try:
        import seaborn
        print(f"seaborn: OK (v{seaborn.__version__})")
    except ImportError as e:
        print(f"seaborn: FAILED ({e})")

    try:
        import nbformat
        print(f"nbformat: OK (v{nbformat.__version__})")
    except ImportError as e:
        print(f"nbformat: FAILED ({e})")

if __name__ == "__main__":
    verify()

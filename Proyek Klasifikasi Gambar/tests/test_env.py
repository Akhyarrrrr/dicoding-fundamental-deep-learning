import pytest

def test_imports():
    import tensorflow as tf
    import PIL
    import numpy as np
    import matplotlib
    import sklearn

    assert tf.__version__ is not None
    assert np.__version__ is not None
    assert PIL.__version__ is not None

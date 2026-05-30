import os
import tempfile
import numpy as np

from utils import get_file_hash, calculate_mape


def test_get_file_hash_length_and_stability():
    # create temp file
    fd, path = tempfile.mkstemp()
    os.close(fd)
    try:
        h = get_file_hash(path)
        assert isinstance(h, str) and len(h) == 32
    finally:
        os.remove(path)


def test_calculate_mape():
    y_true = [100, 200, 300]
    y_pred = [110, 190, 310]
    mape = calculate_mape(y_true, y_pred)
    # manual calc
    expected = np.mean([10/100, 10/200, 10/300]) * 100
    assert abs(mape - expected) < 1e-6

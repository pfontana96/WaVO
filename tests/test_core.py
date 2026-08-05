import numpy as np

import wavo


def test_version():
    assert wavo.__version__ == "0.1.0"


def test_hann_window_shape_and_range():
    w = wavo.hann_window_2d(32, 64)
    assert w.shape == (32, 64)
    assert w.dtype == np.float64
    assert w.min() >= 0.0
    assert w.max() <= 1.0


def test_hann_window_matches_numpy():
    rows, cols = 16, 24
    w = wavo.hann_window_2d(rows, cols)
    # Periodic Hann: numpy's symmetric hanning over n+1 points, last dropped.
    ref = np.outer(np.hanning(rows + 1)[:-1], np.hanning(cols + 1)[:-1])
    np.testing.assert_allclose(w, ref, atol=1e-12)

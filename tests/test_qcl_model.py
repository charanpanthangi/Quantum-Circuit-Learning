"""Tests for the PennyLane-based QCL model."""

import numpy as np
from app.qcl_model import qcl_predict


def test_qcl_predict_range():
    x = np.linspace(0, 1, 5)
    weights = np.zeros((2, 3))
    preds = qcl_predict(x, weights)
    assert preds.shape == x.shape
    # Expectation values from PauliZ lie in [-1, 1]
    assert np.all(preds <= 1.0) and np.all(preds >= -1.0)

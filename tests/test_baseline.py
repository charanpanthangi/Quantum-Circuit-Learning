"""Tests for the classical baseline regressor."""

import numpy as np
from app.dataset import generate_function_data, split_data
from app.classical_baseline import train_classical_regressor, evaluate_classical_regressor


def test_baseline_predictions_shape():
    x, y = generate_function_data(n_samples=40, noise=0.0)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y, test_size=0.25, val_size=0.25)

    model = train_classical_regressor(x_train, y_train)
    mse, r2, preds = evaluate_classical_regressor(model, x_test, y_test)
    assert preds.shape == x_test.shape
    assert mse >= 0
    assert -np.inf < r2 <= 1

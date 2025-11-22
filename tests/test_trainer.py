"""Tests for the training loop and evaluation metrics."""

import numpy as np
from app.dataset import generate_function_data, split_data
from app.trainer import train_qcl, evaluate_qcl


def test_training_reduces_loss():
    x, y = generate_function_data(n_samples=30, noise=0.0)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y, test_size=0.2, val_size=0.2)

    weights, history = train_qcl(x_train, y_train, x_val, y_val, n_epochs=10, lr=0.2, n_layers=1)

    # Ensure loss values were recorded and decrease overall
    assert len(history["train_loss"]) == 10
    assert history["train_loss"][-1] <= history["train_loss"][0]

    mse, r2, preds = evaluate_qcl(x_test, y_test, weights)
    assert preds.shape == x_test.shape
    assert mse >= 0
    assert -np.inf < r2 <= 1

"""
Classical regression baseline to compare against the quantum model.
The baseline gives a familiar reference point for performance.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


def train_classical_regressor(x_train: np.ndarray, y_train: np.ndarray) -> MLPRegressor:
    """Train a small multi-layer perceptron on the 1D regression task."""

    # Reshape to 2D because scikit-learn expects inputs with columns.
    x_train_reshaped = x_train.reshape(-1, 1)

    # A tiny network with tanh activation works well for smooth functions.
    model = MLPRegressor(hidden_layer_sizes=(16, 16), activation="tanh", max_iter=2000)
    model.fit(x_train_reshaped, y_train)
    return model


def evaluate_classical_regressor(model: MLPRegressor, x: np.ndarray, y: np.ndarray):
    """Evaluate the classical model and return predictions plus metrics."""

    preds = model.predict(x.reshape(-1, 1))
    mse = mean_squared_error(y, preds)
    total_var = np.sum((y - np.mean(y)) ** 2)
    residuals = np.sum((y - preds) ** 2)
    r2 = 1 - residuals / total_var if total_var > 0 else float("nan")
    return mse, r2, preds

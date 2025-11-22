"""
Training helpers for the quantum circuit learning model.
This module handles gradient descent, loss tracking, and evaluation utilities.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

from .qcl_model import qnode, qcl_predict


def mean_squared_error(y_true, y_pred):
    """Compute mean squared error in plain words and math-friendly code."""

    return np.mean((y_true - y_pred) ** 2)


def train_qcl(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 100,
    lr: float = 0.1,
    n_layers: int = 2,
):
    """Train the QCL model with gradient descent.

    Args:
        x_train: Training inputs.
        y_train: Training targets.
        x_val: Validation inputs to monitor generalization.
        y_val: Validation targets.
        n_epochs: Number of training epochs.
        lr: Learning rate for the optimizer.
        n_layers: How many rotation layers to use in the circuit.

    Returns:
        Tuple of ``(weights, history)`` where weights are the learned parameters
        and history stores loss curves for plotting.
    """

    # PennyLane has its own NumPy wrapper that keeps track of gradients.
    x_train_t = qml.numpy.array(x_train, requires_grad=False)
    y_train_t = qml.numpy.array(y_train, requires_grad=False)

    # Initialize trainable weights. Shape (n_layers, 3) for RX, RY, RZ.
    weights = qml.numpy.random.normal(scale=0.1, size=(n_layers, 3), requires_grad=True)

    # Simple gradient descent optimizer from PennyLane.
    opt = qml.GradientDescentOptimizer(stepsize=lr)

    # Prepare a vectorized version of the QNode for fast batch evaluation.
    train_qnode = qml.map(qnode, x_train_t)

    history = {"train_loss": [], "val_loss": []}

    for _ in range(n_epochs):
        # Define loss function inside the loop so ``weights`` closes over it.
        def loss_fn(current_weights):
            preds = train_qnode(current_weights)
            return qml.numpy.mean((preds - y_train_t) ** 2)

        # One optimization step updates the weights using the computed gradients.
        weights = opt.step(loss_fn, weights)

        # Track training and validation loss for visualization.
        train_loss = float(loss_fn(weights))
        val_preds = qcl_predict(x_val, weights)
        val_loss = float(mean_squared_error(y_val, val_preds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return weights, history


def evaluate_qcl(x_test: np.ndarray, y_test: np.ndarray, weights: np.ndarray):
    """Evaluate the trained QCL model on held-out data."""

    preds = qcl_predict(x_test, weights)
    mse = mean_squared_error(y_test, preds)

    # R^2 score: 1 - (residual sum / total sum). Useful for regression quality.
    total_var = np.sum((y_test - np.mean(y_test)) ** 2)
    residuals = np.sum((y_test - preds) ** 2)
    r2 = 1 - residuals / total_var if total_var > 0 else float("nan")

    return mse, r2, preds

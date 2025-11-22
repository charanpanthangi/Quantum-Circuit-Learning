"""
Simple dataset utilities for learning a smooth function with quantum circuits.
The module shows how to create training data for regression using a sine wave.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def generate_function_data(n_samples: int = 100, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate pairs ``(x, y)`` that follow a smooth function.

    The regression task aims to learn a continuous mapping from ``x`` to ``y``.
    Here we choose ``y = sin(2πx)`` because the curve is smooth yet nonlinear,
    which makes the learning problem interesting but still beginner friendly.

    Args:
        n_samples: Number of samples to draw uniformly from ``[0, 1]``.
        noise: Standard deviation of optional Gaussian noise added to ``y``.

    Returns:
        A tuple ``(x, y)`` where both arrays have shape ``(n_samples,)``.
    """

    # Sample inputs uniformly between 0 and 1 to cover one full sine period.
    x = np.random.rand(n_samples)

    # Compute the target outputs with a simple sine function.
    y = np.sin(2 * np.pi * x)

    # Optionally add small random noise to show the model handling imperfections.
    if noise > 0:
        y = y + np.random.normal(scale=noise, size=n_samples)

    return x, y


def split_data(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.2):
    """Split the dataset into train, validation, and test parts.

    The split helps us evaluate how well the model generalizes to unseen data.

    Args:
        x: Input features.
        y: Target values.
        test_size: Fraction of data reserved for the test set.
        val_size: Fraction of data reserved for the validation set.

    Returns:
        Tuple of ``(x_train, x_val, x_test, y_train, y_val, y_test)``.
    """

    # First split off the test set from the full dataset.
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # Split the remaining data into training and validation sets.
    val_fraction = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=val_fraction, random_state=42
    )

    return x_train, x_val, x_test, y_train, y_val, y_test

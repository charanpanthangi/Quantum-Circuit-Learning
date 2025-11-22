"""Unit tests for the dataset helpers."""

import numpy as np
from app.dataset import generate_function_data, split_data


def test_generate_function_data_shapes():
    x, y = generate_function_data(n_samples=50, noise=0.01)
    assert x.shape == (50,)
    assert y.shape == (50,)
    assert np.all(x >= 0) and np.all(x <= 1)


def test_split_data_sizes():
    x, y = generate_function_data(n_samples=100)
    splits = split_data(x, y, test_size=0.2, val_size=0.2)
    x_train, x_val, x_test, *_ = splits
    # 60% train, 20% val, 20% test
    assert len(x_train) == 60
    assert len(x_val) == 20
    assert len(x_test) == 20

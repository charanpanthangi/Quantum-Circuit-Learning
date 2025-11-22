"""
Command-line entry point for running the Quantum Circuit Learning demo.
This script trains the quantum model, compares it with a classical regressor,
prints metrics, and saves SVG plots.
"""

from __future__ import annotations

import argparse
import os
import numpy as np

from .dataset import generate_function_data, split_data
from .trainer import train_qcl, evaluate_qcl
from .classical_baseline import train_classical_regressor, evaluate_classical_regressor
from .plots import plot_qcl_fit, plot_classical_fit, plot_training_loss


def parse_args():
    """Parse command-line arguments for experiment settings."""

    parser = argparse.ArgumentParser(description="Quantum Circuit Learning demo")
    parser.add_argument("--n-samples", type=int, default=120, help="Number of data points to generate")
    parser.add_argument("--noise", type=float, default=0.0, help="Standard deviation of Gaussian noise")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs for the quantum model")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for gradient descent")
    parser.add_argument("--layers", type=int, default=2, help="Number of trainable rotation layers")
    return parser.parse_args()


def main():
    """Run the full QCL workflow from data generation to evaluation."""

    args = parse_args()
    np.random.seed(0)

    # Step 1: Create the dataset for the regression task.
    x, y = generate_function_data(n_samples=args.n_samples, noise=args.noise)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y)

    # Step 2: Train the quantum model.
    weights, history = train_qcl(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        n_epochs=args.epochs,
        lr=args.lr,
        n_layers=args.layers,
    )

    # Step 3: Train a classical baseline for comparison.
    classical_model = train_classical_regressor(x_train, y_train)

    # Step 4: Evaluate both models on the test set.
    qcl_mse, qcl_r2, qcl_preds = evaluate_qcl(x_test, y_test, weights)
    classical_mse, classical_r2, classical_preds = evaluate_classical_regressor(classical_model, x_test, y_test)

    # Report metrics in a simple, friendly way.
    print("Quantum model - MSE: {:.4f}, R^2: {:.4f}".format(qcl_mse, qcl_r2))
    print("Classical model - MSE: {:.4f}, R^2: {:.4f}".format(classical_mse, classical_r2))

    # Step 5: Save SVG plots to the examples folder.
    os.makedirs("examples", exist_ok=True)
    plot_qcl_fit(x_test, y_test, qcl_preds, "examples/qcl_function_fit.svg")
    plot_training_loss(history, "examples/qcl_training_loss.svg")
    plot_classical_fit(x_test, y_test, classical_preds, "examples/classical_function_fit.svg")


if __name__ == "__main__":
    main()

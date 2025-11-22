"""
Plotting helpers that save lightweight SVG figures instead of binary images.
The SVG format stays crisp in GitHub previews and keeps the repository small.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


# Ensure Matplotlib creates vector graphics.
plt.rcParams["figure.dpi"] = 120


def plot_qcl_fit(x, y_true, y_pred, output_path):
    """Plot the quantum model fit and save as an SVG."""

    plt.figure()
    plt.scatter(x, y_true, label="Target data", color="tab:blue", s=10)
    plt.plot(x, y_pred, label="QCL prediction", color="tab:orange")
    plt.title("Quantum Circuit Learning fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_classical_fit(x, y_true, y_pred, output_path):
    """Plot the classical baseline fit as an SVG."""

    plt.figure()
    plt.scatter(x, y_true, label="Target data", color="tab:blue", s=10)
    plt.plot(x, y_pred, label="Classical prediction", color="tab:green")
    plt.title("Classical baseline fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_training_loss(history, output_path):
    """Plot training and validation loss curves as an SVG."""

    plt.figure()
    plt.plot(history.get("train_loss", []), label="Train loss", color="tab:orange")
    plt.plot(history.get("val_loss", []), label="Validation loss", color="tab:red")
    plt.title("QCL training loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()

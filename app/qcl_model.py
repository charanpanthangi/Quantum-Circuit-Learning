"""
Quantum Circuit Learning model built with PennyLane.
This module defines the device, parameterized circuit, and prediction helper.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np


def create_qcl_device(n_qubits: int = 1) -> qml.Device:
    """Create a PennyLane device used to run the quantum circuit.

    Args:
        n_qubits: Number of qubits to simulate. One qubit is enough for the
            small regression demo but more can be used to increase capacity.

    Returns:
        A PennyLane device ready for building a QNode.
    """

    # ``default.qubit`` is a built-in simulator that runs on a classical CPU.
    return qml.device("default.qubit", wires=n_qubits)


def qcl_circuit(x: float, weights: qml.numpy.tensor) -> qml.numpy.tensor:
    """Parameterized quantum circuit that encodes ``x`` and outputs an expectation.

    The circuit uses two steps:
    1. Encode the classical input ``x`` into rotations on the first qubit.
    2. Apply trainable rotations and entangling gates so the circuit can bend
       the measured expectation value toward the target ``y``.

    Args:
        x: Single input value from the training data.
        weights: Trainable angles for the rotation gates. Shape ``(n_layers, 3)``.

    Returns:
        Expectation value of the Pauli-Z measurement on the first wire.
    """

    # Encode the input with angle rotations. Multiplying ``x`` by ``pi`` spreads
    # values nicely around the Bloch sphere.
    qml.RX(np.pi * x, wires=0)
    qml.RY(2 * np.pi * x, wires=0)

    # Apply each layer of trainable rotations. The angles will be updated during
    # training so the circuit output matches the target function.
    for layer in weights:
        qml.RX(layer[0], wires=0)
        qml.RY(layer[1], wires=0)
        qml.RZ(layer[2], wires=0)

    # Measure the expectation value of Z. The value lies in [-1, 1].
    return qml.expval(qml.PauliZ(0))


# Create a device and bind it to a QNode. Using a global device keeps things simple.
device = create_qcl_device()


@qml.qnode(device)
def qnode(x: float, weights):
    """QNode wrapping the circuit to make it differentiable.

    Args:
        x: Input value.
        weights: Trainable angles.

    Returns:
        Expectation value from the circuit for this input.
    """

    return qcl_circuit(x, weights)


def qcl_predict(x_batch: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Predict outputs for a batch of ``x`` values using the QCL model.

    Args:
        x_batch: Array of input values.
        weights: Current trainable parameters.

    Returns:
        Array of predictions with the same shape as ``x_batch``.
    """

    # Vectorize the QNode so that it runs efficiently over many inputs.
    qnode_vectorized = qml.map(qnode, qml.numpy.array(x_batch))
    predictions = qnode_vectorized(weights)
    return np.array(predictions, dtype=float)

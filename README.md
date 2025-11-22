# Quantum Circuit Learning (QCL) — Function Regression with PQCs

Beginner-friendly example showing how a small parameterized quantum circuit can learn
a smooth function such as `y = sin(2πx)`. The project also includes a simple
classical regressor for comparison, SVG-only plots, and a notebook tutorial.

## Why Quantum Circuit Learning?

Quantum Circuit Learning (QCL) uses trainable rotations in a quantum circuit to
approximate functions, similar to how tiny neural networks work. Inputs are
encoded into quantum states, rotated with learnable angles, and measured to
produce predictions. Because the expectation value of a measurement is
continuous, the circuit can learn smooth targets like a sine wave.

## Repository contents

```
app/
  dataset.py            # generate and split the regression data
  qcl_model.py          # PennyLane quantum circuit and prediction helper
  trainer.py            # gradient descent training loop for QCL
  classical_baseline.py # small MLP baseline
  plots.py              # SVG-only plotting utilities
  main.py               # CLI to run the full demo
notebooks/
  qcl_regression_demo.ipynb   # tutorial using SVG outputs
examples/
  qcl_function_fit.svg        # quantum model fit on test data
  qcl_training_loss.svg       # loss curve during training
  classical_function_fit.svg  # classical baseline fit
```

## Running the demo

1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Run the CLI with default settings:
   ```bash
   python app/main.py
   ```
   You can adjust the dataset size or training length:
   ```bash
   python app/main.py --n-samples 150 --epochs 80 --noise 0.02
   ```
3. After running, the following SVG plots will appear in `examples/`:
   - `qcl_function_fit.svg`
   - `qcl_training_loss.svg`
   - `classical_function_fit.svg`

## Why SVG-only visuals?

GitHub previews SVG files directly in the browser, so you can zoom without
losing clarity and avoid unsupported binary previews. All plots in this
repository are saved as SVG to keep the repo light-weight and friendly to
version control. The notebook also configures Matplotlib to use SVG with:
```
%config InlineBackend.figure_formats = ['svg']
```

## Classical vs quantum comparison

The project trains two models on the same data:
- **Quantum model:** a PennyLane QNode with input encoding rotations and
  trainable RX/RY/RZ layers.
- **Classical baseline:** a small scikit-learn MLPRegressor with tanh
  activation.

Both models are evaluated on a held-out test set with mean squared error and
R² score to show how the quantum approach stacks up against a familiar
classical method.

## Testing

Run the lightweight test suite with:
```bash
pytest
```
Tests cover data generation, the quantum model’s output range, the training
loop, and the classical baseline.

## Docker usage

Build and run the project in Docker:
```bash
docker build -t qcl-demo .
docker run --rm qcl-demo --help
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for
details.

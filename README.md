# Perceptron

Perceptron implementations from scratch in NumPy — from the classic Rosenblatt unit-step version to a sigmoid variant served as a REST API with Flask.

## What's here

| File | Purpose |
|---|---|
| `initial.py` | Classic perceptron with a unit-step activation — the textbook starting point |
| `perceptron_api.py` | Sigmoid perceptron trained on the Excel dataset and exposed as a Flask prediction API |
| `percptronrend/perceptron_api.py` | Deployment copy of the API (e.g. for Render), served with gunicorn |
| `excel (spd).py` | Utility for working with `synthetic_perceptron_dataset.xlsx` |
| `excel (testset).py` | Utility for the `testset.xlsm` evaluation set |
| `synthetic_perceptron_dataset.xlsx` | Synthetic training data |
| `testset.xlsm` | Held-out test set |
| `fib.py` | Small Fibonacci exercise (unrelated warm-up) |

## Setup

```bash
pip install -r requirements.txt
```

## Run the API

```bash
python perceptron_api.py
```

The Flask app trains the perceptron on the Excel dataset at startup (features are min-max scaled) and exposes a JSON prediction endpoint. For production-style serving:

```bash
gunicorn perceptron_api:app
```

## How the perceptron works

Weights start at zero and update per-sample:

```
w ← w + lr · (y − ŷ) · x
b ← b + lr · (y − ŷ)
```

`initial.py` uses a hard unit-step activation (`ŷ = 1 if w·x + b > 0`); `perceptron_api.py` swaps in a sigmoid so the error signal is graded rather than binary, with a 0.5 threshold at predict time.

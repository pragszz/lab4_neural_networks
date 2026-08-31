# Lab 4 — Neural Networks (Day 1: Regression)

A small MLP regressor built in PyTorch on the California Housing dataset, compared
against a hand-derived NumPy gradient descent (linear model) on the same data.

## Files
- `stage_a_regression.py` — data loading/preprocessing (`load_data`) and gradient descent (`gradient_descent_numpy`)
- `train.py` — MLP builder (`build_mlp`) and training loop (`train_full_batch`)
- `day1_regression_starter.ipynb` — runs everything and plots the loss curves

## Setup
```
pip install torch scikit-learn matplotlib numpy
```

## Run
Open the notebook and run the cells top to bottom.

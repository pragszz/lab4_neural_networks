from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

def load_data(data, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_train = torch.tensor(np.asarray(y_train), dtype=torch.float32).reshape(-1,1)
    y_test  = torch.tensor(np.asarray(y_test),  dtype=torch.float32).reshape(-1,1)

    return X_train, y_train, X_test, y_test

def gradient_descent_numpy(X, y, X_test, y_test, lr=0.1, epochs=200):
    X = X.numpy()
    y = y.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()
    n, d = X.shape

    # Make sure y has shape (n, 1)
    y = y.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    w = np.zeros((d, 1))
    b = 0.0
    losses = []

    for _ in range(epochs):
        # Forward pass
        pred = X @ w + b
        err = pred - y

        # Training loss
        loss = np.mean(err ** 2)
        losses.append(loss)

        # Gradients
        grad_w = (2.0 / n) * (X.T @ err)
        grad_b = (2.0 / n) * np.sum(err)

        # Update
        w -= lr * grad_w
        b -= lr * grad_b

    # Test MSE — use X_test and y_test
    test_pred = X_test @ w + b
    test_mse = np.mean((test_pred - y_test) ** 2)

    return w, b, losses, test_mse
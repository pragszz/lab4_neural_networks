import torch
import torch.nn as nn
import torch.optim as optim

def build_mlp(X_train):
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 32),
        nn.ReLU(),
        nn.Linear(32, 1))
    return model


def train_full_batch(X_train, y_train, model, lr=0.01, epochs=100):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        predictions = model(X_train)
        loss = loss_fn(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses


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

def train_full_loader(model,train_loader,loss_fn,optimizer,epochs,device="cpu",val_loader=None,patience=None):

    train_losses = []
    val_accuracies = []

    best_state = None
    best_val_accuracy = 0
    patience_counter = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_samples = 0

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(X)

            loss = loss_fn(pred, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

        average_loss = total_loss / total_samples
        train_losses.append(average_loss)

        if val_loader is not None:

            model.eval()

            correct = 0
            total = 0

            with torch.no_grad():

                for X, y in val_loader:

                    X = X.to(device)
                    y = y.to(device)

                    pred = model(X)

                    predictions = torch.argmax(pred, dim=1)

                    correct += (predictions == y).sum().item()
                    total += y.size(0)

            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)

            # Early stopping
            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy
                patience_counter = 0

                best_state = {
                    k: v.clone()
                    for k, v in model.state_dict().items()
                }

            else:
                patience_counter += 1

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {average_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            if patience is not None and patience_counter >= patience:
                print("Early stopping")
                break

        else:

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {average_loss:.4f}"
            )
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_accuracies


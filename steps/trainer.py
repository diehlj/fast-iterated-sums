from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.helpers import calculate_topk_accuracy


@dataclass(frozen=True)
class Metrics:
    train_loss: float
    train_acc: float
    val_loss: float
    top1_val_acc: float
    top5_val_acc: float


def train_epoch(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Metrics:
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    top1_val_acc = 0.0
    top5_val_acc = 0.0

    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == labels).sum().item()

    scheduler.step()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)

    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            top1_val_acc += calculate_topk_accuracy(
                y_pred=outputs, y_true=labels, topk=1
            )
            top5_val_acc += calculate_topk_accuracy(
                y_pred=outputs, y_true=labels, topk=5
            )

    val_loss /= len(validation_loader)
    top1_val_acc /= len(validation_loader.dataset)
    top5_val_acc /= len(validation_loader.dataset)

    return Metrics(
        train_loss=train_loss,
        train_acc=train_acc,
        val_loss=val_loss,
        top1_val_acc=top1_val_acc,
        top5_val_acc=top5_val_acc,
    )


def train_autoencoder(
    ae: nn.Module,
    fe: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_epochs: int,
    device: str,
) -> tuple[list[float], list[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        ae.train()
        train_loss_sum = 0

        for data, _ in train_loader:
            data = data.to(device)

            with torch.no_grad():
                features = fe(data)

            output = ae(features)
            train_loss = criterion(output, features)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss_sum += train_loss.item()

        train_loss_avg = train_loss_sum / len(train_loader)
        train_losses.append(train_loss_avg)

        ae.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            for data, _ in validation_loader:
                data = data.to(device)
                features = fe(data)

                output = ae(features)

                val_loss = criterion(output, features)
                val_loss_sum += val_loss.item()

            val_loss_avg = val_loss_sum / len(validation_loader)
            validation_losses.append(val_loss_avg)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], train loss: {train_loss_avg:.4f}, validation loss: {val_loss_avg:.4f}"
        )

    return train_losses, validation_losses

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

@dataclass(slots=True)
class EpochResult:
    loss: float
    accuracy: float
    sample_count: int


@dataclass(slots=True)
class BatchResult:
    loss: float
    accuracy: float
    sample_count: int
    output_shape: tuple[int, ...]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loss_function() -> nn.Module:
    return nn.CrossEntropyLoss()


def create_optimizer(model: nn.Module, learning_rate: float) -> Optimizer:
    return Adam(model.parameters(), lr=learning_rate)


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(outputs, dim=1)
    correct_predictions = int((predictions == labels).sum().item())
    total_predictions = int(labels.size(0))
    return correct_predictions / max(total_predictions, 1)


def train_one_batch(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> BatchResult:
    model.to(device)
    model.train()

    images = images.to(device)
    labels = labels.to(device)

    # 1. Clear old gradients.
    optimizer.zero_grad()

    # 2. Run a forward pass through the model.
    outputs = model(images)

    # 3. Compare the predictions with the true labels.
    loss = criterion(outputs, labels)

    # 4. Backpropagate the error and update the weights.
    loss.backward()
    optimizer.step()

    return BatchResult(
        loss=float(loss.item()),
        accuracy=calculate_accuracy(outputs, labels),
        sample_count=int(labels.size(0)),
        output_shape=tuple(outputs.shape),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.to(device)
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_correct += int((torch.argmax(outputs, dim=1) == labels).sum().item())
        total_samples += batch_size

    return EpochResult(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
        sample_count=total_samples,
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_correct += int((torch.argmax(outputs, dim=1) == labels).sum().item())
            total_samples += batch_size

    return EpochResult(
        loss=total_loss / max(total_samples, 1),
        accuracy=total_correct / max(total_samples, 1),
        sample_count=total_samples,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    class_to_index: dict[str, int],
    epoch: int,
    metrics: EpochResult,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "class_to_index": class_to_index,
            "metrics": {
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "sample_count": metrics.sample_count,
            },
            "class_names": class_names_from_mapping(class_to_index),
        },
        checkpoint_path,
    )


def class_names_from_mapping(class_to_index: dict[str, int]) -> list[str]:
    return [
        class_name
        for class_name, _ in sorted(class_to_index.items(), key=lambda item: item[1])
    ]

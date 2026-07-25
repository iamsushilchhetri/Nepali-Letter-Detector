from __future__ import annotations

import torch
from torch import nn


class NepaliLetterCNN(nn.Module):
    """Custom CNN written from scratch for Nepali handwritten character images."""

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2 for classification.")

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout_features = nn.Dropout(p=0.25)
        self.output_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout_classifier = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.bn1(self.conv1(images)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_features(x)
        x = self.output_pool(x)

        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.dropout_classifier(x)
        return self.fc2(x)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

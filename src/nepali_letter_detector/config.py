from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
}


@dataclass(slots=True)
class DataConfig:
    project_root: Path
    data_dir: Path
    image_size: int = 32
    batch_size: int = 16
    num_workers: int = 0
    grayscale: bool = True
    normalize_mean: tuple[float, ...] = (0.5,)
    normalize_std: tuple[float, ...] = (0.5,)

    @classmethod
    def from_project_root(cls, project_root: Path) -> "DataConfig":
        return cls(project_root=project_root, data_dir=project_root / "data")

    def split_dir(self, split_name: str) -> Path:
        return self.data_dir / split_name

    @property
    def input_channels(self) -> int:
        return 1 if self.grayscale else 3


@dataclass(slots=True)
class TrainingConfig:
    project_root: Path
    models_dir: Path
    epochs: int = 10
    learning_rate: float = 1e-3
    random_seed: int = 42
    device_preference: str = "auto"
    dry_run_batch_size: int = 4
    dry_run_num_classes: int = 5

    @classmethod
    def from_project_root(cls, project_root: Path) -> "TrainingConfig":
        return cls(project_root=project_root, models_dir=project_root / "models")

    @property
    def checkpoint_dir(self) -> Path:
        return self.models_dir / "checkpoints"

    @property
    def resolved_device(self) -> torch.device:
        if self.device_preference == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device_preference)

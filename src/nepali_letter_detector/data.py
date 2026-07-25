from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DataConfig, IMAGE_EXTENSIONS


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(slots=True)
class ImageRecord:
    image_path: Path
    class_name: str
    class_index: int


@dataclass(slots=True)
class SplitSummary:
    split_name: str
    class_names: tuple[str, ...]
    image_count: int
    class_counts: dict[str, int]

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    @property
    def has_images(self) -> bool:
        return self.image_count > 0


@dataclass(slots=True)
class PreparedData:
    class_to_index: dict[str, int]
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader | None
    summaries: dict[str, SplitSummary]


def _visible_subdirectories(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ],
        key=lambda path: path.name.lower(),
    )


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.as_posix().lower(),
    )


def summarize_split(split_dir: Path, split_name: str) -> SplitSummary:
    class_counts: Counter[str] = Counter()
    for class_dir in _visible_subdirectories(split_dir):
        class_counts[class_dir.name] = len(_image_files(class_dir))

    return SplitSummary(
        split_name=split_name,
        class_names=tuple(sorted(class_counts)),
        image_count=sum(class_counts.values()),
        class_counts=dict(sorted(class_counts.items())),
    )


def summarize_dataset(data_config: DataConfig) -> dict[str, SplitSummary]:
    return {
        split_name: summarize_split(data_config.split_dir(split_name), split_name)
        for split_name in SPLIT_NAMES
    }


def expected_dataset_layout(data_config: DataConfig) -> str:
    data_dir = data_config.data_dir
    return "\n".join(
        [
            f"{data_dir.name}/",
            "|-- train/",
            "|   |-- <class_name>/",
            "|   |   `-- image_001.png",
            "|-- validation/",
            "|   |-- <class_name>/",
            "|   |   `-- image_001.png",
            "`-- test/",
            "    |-- <class_name>/",
            "    `-- image_001.png",
        ]
    )


def dataset_is_ready_for_training(summaries: dict[str, SplitSummary]) -> bool:
    train_summary = summaries["train"]
    validation_summary = summaries["validation"]
    return train_summary.has_images and validation_summary.has_images


def build_class_to_index(train_dir: Path) -> dict[str, int]:
    class_directories = _visible_subdirectories(train_dir)
    return {class_dir.name: index for index, class_dir in enumerate(class_directories)}


def build_image_transform(data_config: DataConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((data_config.image_size, data_config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config.normalize_mean,
                std=data_config.normalize_std,
            ),
        ]
    )


class ImageFolderClassificationDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        class_to_index: dict[str, int],
        data_config: DataConfig,
    ) -> None:
        self.split_dir = split_dir
        self.class_to_index = class_to_index
        self.data_config = data_config
        self.transform = build_image_transform(data_config)
        self.image_mode = "L" if data_config.grayscale else "RGB"
        self.records = self._collect_records()

        if not self.records:
            raise ValueError(
                f"No labeled images were found in '{split_dir}'. "
                "Add class folders with real images before training."
            )

    def _collect_records(self) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        for class_dir in _visible_subdirectories(self.split_dir):
            if class_dir.name not in self.class_to_index:
                raise ValueError(
                    f"Unknown class folder '{class_dir.name}' in '{self.split_dir}'. "
                    "Validation and test labels must match the training labels."
                )

            for image_path in _image_files(class_dir):
                records.append(
                    ImageRecord(
                        image_path=image_path,
                        class_name=class_dir.name,
                        class_index=self.class_to_index[class_dir.name],
                    )
                )

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            image = image.convert(self.image_mode)
            tensor = self.transform(image)
        return tensor, record.class_index


def prepare_dataloaders(data_config: DataConfig) -> PreparedData:
    summaries = summarize_dataset(data_config)
    if not summaries["train"].has_images:
        raise ValueError(
            "The training split does not contain any labeled images yet."
        )
    if not summaries["validation"].has_images:
        raise ValueError(
            "The validation split does not contain any labeled images yet."
        )

    class_to_index = build_class_to_index(data_config.split_dir("train"))
    if not class_to_index:
        raise ValueError(
            "No class folders were found in the training split. "
            "Create one folder per Nepali letter."
        )

    train_dataset = ImageFolderClassificationDataset(
        split_dir=data_config.split_dir("train"),
        class_to_index=class_to_index,
        data_config=data_config,
    )
    validation_dataset = ImageFolderClassificationDataset(
        split_dir=data_config.split_dir("validation"),
        class_to_index=class_to_index,
        data_config=data_config,
    )

    test_loader: DataLoader | None = None
    if summaries["test"].has_images:
        test_dataset = ImageFolderClassificationDataset(
            split_dir=data_config.split_dir("test"),
            class_to_index=class_to_index,
            data_config=data_config,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=data_config.batch_size,
            shuffle=False,
            num_workers=data_config.num_workers,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
    )

    return PreparedData(
        class_to_index=class_to_index,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        summaries=summaries,
    )

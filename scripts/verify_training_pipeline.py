from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nepali_letter_detector.config import DataConfig, TrainingConfig
from nepali_letter_detector.data import (
    dataset_is_ready_for_training,
    expected_dataset_layout,
    prepare_dataloaders,
    summarize_dataset,
)
from nepali_letter_detector.model import NepaliLetterCNN, count_trainable_parameters
from nepali_letter_detector.training import create_optimizer, train_one_batch


def run_dry_run(data_config: DataConfig, training_config: TrainingConfig) -> dict[str, str]:
    model = NepaliLetterCNN(
        num_classes=training_config.dry_run_num_classes,
        in_channels=data_config.input_channels,
    )
    optimizer = create_optimizer(model, training_config.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    images = torch.rand(
        training_config.dry_run_batch_size,
        data_config.input_channels,
        data_config.image_size,
        data_config.image_size,
    )
    labels = torch.tensor(
        [
            index % training_config.dry_run_num_classes
            for index in range(training_config.dry_run_batch_size)
        ],
        dtype=torch.long,
    )

    batch_result = train_one_batch(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        images=images,
        labels=labels,
        device=training_config.resolved_device,
    )

    return {
        "dry_run_device": str(training_config.resolved_device),
        "dry_run_batch_shape": str(tuple(images.shape)),
        "dry_run_logits_shape": str(batch_result.output_shape),
        "dry_run_loss": f"{batch_result.loss:.4f}",
        "dry_run_accuracy": f"{batch_result.accuracy:.4f}",
        "trainable_parameters": str(count_trainable_parameters(model)),
    }


def main() -> None:
    data_config = DataConfig.from_project_root(PROJECT_ROOT)
    training_config = TrainingConfig.from_project_root(PROJECT_ROOT)
    summaries = summarize_dataset(data_config)

    print("Nepali Handwritten Letter Detector")
    print("Data pipeline and training skeleton verification started.")
    print("model_type: custom CNN written from scratch with PyTorch layers")

    for split_name, summary in summaries.items():
        print(
            f"{split_name}: classes={summary.class_count}, "
            f"images={summary.image_count}, labels={list(summary.class_names)}"
        )

    if dataset_is_ready_for_training(summaries):
        prepared_data = prepare_dataloaders(data_config)
        first_batch_images, first_batch_labels = next(iter(prepared_data.train_loader))
        print(
            "dataset_loader_check: "
            f"batch_shape={tuple(first_batch_images.shape)}, "
            f"labels_shape={tuple(first_batch_labels.shape)}"
        )
        print(f"class_to_index: {prepared_data.class_to_index}")
    else:
        print("dataset_loader_check: skipped because no real labeled dataset is present yet.")
        print("expected_layout:")
        print(expected_dataset_layout(data_config))

    dry_run_results = run_dry_run(data_config, training_config)
    for key, value in dry_run_results.items():
        print(f"{key}: {value}")

    print("Verification passed. The training pipeline can load data and run a dry training step.")
    print("This script only verifies the pipeline. It does not train a new model.")


if __name__ == "__main__":
    main()

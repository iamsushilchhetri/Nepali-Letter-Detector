from __future__ import annotations

import argparse
import json
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
from nepali_letter_detector.training import (
    class_names_from_mapping,
    create_loss_function,
    create_optimizer,
    evaluate,
    save_checkpoint,
    set_random_seed,
    train_one_batch,
    train_one_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a custom CNN for Nepali handwritten letter classification."
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Target square image size in pixels.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, for example cpu, cuda, or auto.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run one in-memory training step without using dataset files.",
    )
    return parser


def apply_overrides(
    data_config: DataConfig,
    training_config: TrainingConfig,
    args: argparse.Namespace,
) -> None:
    if args.epochs is not None:
        training_config.epochs = args.epochs
    if args.batch_size is not None:
        data_config.batch_size = args.batch_size
    if args.image_size is not None:
        data_config.image_size = args.image_size
    if args.learning_rate is not None:
        training_config.learning_rate = args.learning_rate
    if args.device is not None:
        training_config.device_preference = args.device


def validate_config(data_config: DataConfig, training_config: TrainingConfig) -> None:
    if training_config.epochs <= 0:
        raise ValueError("epochs must be greater than 0.")
    if data_config.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if data_config.image_size <= 0:
        raise ValueError("image_size must be greater than 0.")
    if training_config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0.")


def print_dataset_summary(data_config: DataConfig) -> dict[str, object]:
    summaries = summarize_dataset(data_config)
    print("dataset_summary:")
    for split_name, summary in summaries.items():
        print(
            f"  {split_name}: classes={summary.class_count}, "
            f"images={summary.image_count}, labels={list(summary.class_names)}"
        )
    return summaries


def save_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_dry_run(data_config: DataConfig, training_config: TrainingConfig) -> None:
    print("training_mode: dry_run")
    print("model_type: custom CNN written from scratch with PyTorch layers")

    model = NepaliLetterCNN(
        num_classes=training_config.dry_run_num_classes,
        in_channels=data_config.input_channels,
    )
    optimizer = create_optimizer(model, training_config.learning_rate)
    criterion = create_loss_function()

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

    print(f"device: {training_config.resolved_device}")
    print(f"batch_shape: {tuple(images.shape)}")
    print(f"output_shape: {batch_result.output_shape}")
    print(f"loss: {batch_result.loss:.4f}")
    print(f"accuracy: {batch_result.accuracy:.4f}")
    print(f"trainable_parameters: {count_trainable_parameters(model)}")
    print("dry_run_status: passed")
    print("No real dataset was used and no trained model was claimed.")


def run_training(data_config: DataConfig, training_config: TrainingConfig) -> None:
    print("training_mode: dataset")
    print("model_type: custom CNN written from scratch with PyTorch layers")
    print(f"device: {training_config.resolved_device}")
    print(f"epochs: {training_config.epochs}")
    print(f"batch_size: {data_config.batch_size}")
    print(f"image_size: {data_config.image_size}")
    print(f"learning_rate: {training_config.learning_rate}")

    summaries = print_dataset_summary(data_config)
    if not dataset_is_ready_for_training(summaries):
        print("training_start: skipped because the dataset is not ready yet.")
        print("To train the model, add real images using this layout:")
        print(expected_dataset_layout(data_config))
        return

    prepared_data = prepare_dataloaders(data_config)
    class_count = len(prepared_data.class_to_index)

    model = NepaliLetterCNN(
        num_classes=class_count,
        in_channels=data_config.input_channels,
    )
    optimizer = create_optimizer(model, training_config.learning_rate)
    criterion = create_loss_function()

    checkpoint_dir = training_config.checkpoint_dir
    history_path = checkpoint_dir / "training_history.json"
    class_map_path = checkpoint_dir / "class_to_index.json"
    best_checkpoint_path = checkpoint_dir / "best_model.pt"
    final_checkpoint_path = checkpoint_dir / "final_model.pt"
    exported_model_path = training_config.models_dir / "model.pt"
    exported_classes_path = training_config.models_dir / "classes.json"

    print(f"class_to_index: {prepared_data.class_to_index}")
    print(f"trainable_parameters: {count_trainable_parameters(model)}")

    history: list[dict[str, float | int]] = []
    best_validation_loss: float | None = None
    best_epoch: int | None = None

    for epoch in range(1, training_config.epochs + 1):
        train_result = train_one_epoch(
            model=model,
            dataloader=prepared_data.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=training_config.resolved_device,
        )
        validation_result = evaluate(
            model=model,
            dataloader=prepared_data.validation_loader,
            criterion=criterion,
            device=training_config.resolved_device,
        )

        history_entry = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "train_sample_count": train_result.sample_count,
            "validation_loss": validation_result.loss,
            "validation_accuracy": validation_result.accuracy,
            "validation_sample_count": validation_result.sample_count,
        }
        history.append(history_entry)

        print(
            f"epoch {epoch}: "
            f"train_loss={train_result.loss:.4f}, "
            f"train_accuracy={train_result.accuracy:.4f}, "
            f"validation_loss={validation_result.loss:.4f}, "
            f"validation_accuracy={validation_result.accuracy:.4f}"
        )

        if best_validation_loss is None or validation_result.loss < best_validation_loss:
            best_validation_loss = validation_result.loss
            best_epoch = epoch
            save_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                class_to_index=prepared_data.class_to_index,
                epoch=epoch,
                metrics=validation_result,
            )
            torch.save(model.state_dict(), exported_model_path)
            save_json_file(
                exported_classes_path,
                class_names_from_mapping(prepared_data.class_to_index),
            )

    save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=model,
        class_to_index=prepared_data.class_to_index,
        epoch=training_config.epochs,
        metrics=validation_result,
    )
    save_json_file(class_map_path, prepared_data.class_to_index)
    save_json_file(history_path, history)

    print("training_complete: yes")
    print(f"best_epoch: {best_epoch}")
    print(f"best_checkpoint: {best_checkpoint_path}")
    print(f"final_checkpoint: {final_checkpoint_path}")
    print(f"exported_model: {exported_model_path}")
    print(f"exported_classes: {exported_classes_path}")
    print(f"class_mapping: {class_map_path}")
    print(f"training_history: {history_path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    data_config = DataConfig.from_project_root(PROJECT_ROOT)
    training_config = TrainingConfig.from_project_root(PROJECT_ROOT)
    apply_overrides(data_config, training_config, args)
    validate_config(data_config, training_config)
    set_random_seed(training_config.random_seed)

    if args.dry_run:
        run_dry_run(data_config, training_config)
    else:
        run_training(data_config, training_config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

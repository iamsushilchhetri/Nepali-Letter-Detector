from __future__ import annotations

from importlib.metadata import version
import os
import platform
from pathlib import Path


REQUIRED_DIRECTORIES = (
    "src",
    "src/nepali_letter_detector",
    "data/train",
    "data/validation",
    "data/test",
    "models",
    "notebooks",
    "scripts",
)


def assert_directories_exist(project_root: Path) -> None:
    missing = [
        relative_path
        for relative_path in REQUIRED_DIRECTORIES
        if not (project_root / relative_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required project paths: " + ", ".join(sorted(missing))
        )


def configure_runtime_paths(project_root: Path) -> None:
    matplotlib_cache_dir = project_root / ".cache" / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))


def run_library_checks() -> dict[str, str]:
    import cv2
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    import torch
    import torchvision
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from torchvision import transforms

    sample_pixels = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
    pil_image = Image.fromarray(sample_pixels)
    resized_pixels = cv2.resize(sample_pixels, (32, 32), interpolation=cv2.INTER_AREA)

    tensor = transforms.ToTensor()(pil_image).unsqueeze(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 28 * 28, 5),
    )
    logits = model(tensor)

    train_indices, validation_indices = train_test_split(
        list(range(20)),
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    figure, axis = plt.subplots()
    axis.imshow(resized_pixels, cmap="gray")
    axis.set_title("Environment Verification")
    axis.axis("off")
    plt.close(figure)

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opencv": cv2.__version__,
        "pillow": Image.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "flask": version("Flask"),
        "tensor_shape": str(tuple(tensor.shape)),
        "logits_shape": str(tuple(logits.shape)),
        "train_split_size": str(len(train_indices)),
        "validation_split_size": str(len(validation_indices)),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert_directories_exist(project_root)
    configure_runtime_paths(project_root)

    print("Nepali Handwritten Letter Detector")
    print("Environment verification started.")

    results = run_library_checks()
    for key, value in results.items():
        print(f"{key}: {value}")

    print("Verification passed. The environment is ready for training and web inference.")
    print("This script only checks the environment. It does not train a model.")


if __name__ == "__main__":
    main()

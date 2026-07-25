from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .labels import display_name, sorted_label_names
from .model import NepaliLetterCNN


DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CHECKPOINT_CANDIDATES = (
    ("checkpoints", "best_model.pt"),
    ("checkpoints", "final_model.pt"),
    ("model.pt",),
)
CLASS_FILE_CANDIDATES = (
    ("classes.json",),
    ("checkpoints", "class_to_index.json"),
)


class Predictor:
    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = self._resolve_weights_path()

        raw_checkpoint = torch.load(self.weights_path, map_location=self.device)
        self.state_dict = self._extract_state_dict(raw_checkpoint)
        self.class_names = self._resolve_class_names(raw_checkpoint, self.state_dict)

        self.model = NepaliLetterCNN(num_classes=len(self.class_names))
        self.model.load_state_dict(self.state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_weights_path(self) -> Path:
        for parts in CHECKPOINT_CANDIDATES:
            candidate_path = self.models_dir.joinpath(*parts)
            if candidate_path.exists():
                return candidate_path

        raise FileNotFoundError(
            "No model checkpoint was found. Expected one of: "
            "models/checkpoints/best_model.pt, "
            "models/checkpoints/final_model.pt, or models/model.pt."
        )

    @staticmethod
    def _extract_state_dict(raw_checkpoint: object) -> dict[str, torch.Tensor]:
        if isinstance(raw_checkpoint, dict) and "model_state_dict" in raw_checkpoint:
            return raw_checkpoint["model_state_dict"]
        if isinstance(raw_checkpoint, dict) and "fc2.weight" in raw_checkpoint:
            return raw_checkpoint
        raise TypeError("Unsupported checkpoint format.")

    def _resolve_class_names(
        self,
        raw_checkpoint: object,
        state_dict: dict[str, torch.Tensor],
    ) -> list[str]:
        if isinstance(raw_checkpoint, dict) and "class_names" in raw_checkpoint:
            return list(raw_checkpoint["class_names"])
        if isinstance(raw_checkpoint, dict) and "class_to_index" in raw_checkpoint:
            return self._sorted_class_names(raw_checkpoint["class_to_index"])

        for parts in CLASS_FILE_CANDIDATES:
            candidate_path = self.models_dir.joinpath(*parts)
            if not candidate_path.exists():
                continue

            loaded = json.loads(candidate_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                return loaded
            if isinstance(loaded, dict):
                return self._sorted_class_names(loaded)

        built_in_labels = sorted_label_names()
        if self._class_count_from_state_dict(state_dict) == len(built_in_labels):
            return built_in_labels

        raise FileNotFoundError(
            "Could not determine the class order for the checkpoint. "
            "Add models/classes.json or models/checkpoints/class_to_index.json."
        )

    @staticmethod
    def _sorted_class_names(class_to_index: dict[str, int]) -> list[str]:
        return [
            class_name
            for class_name, _ in sorted(class_to_index.items(), key=lambda item: item[1])
        ]

    @staticmethod
    def _class_count_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
        if "fc2.weight" not in state_dict:
            raise KeyError("The checkpoint does not contain the expected output layer.")
        return int(state_dict["fc2.weight"].shape[0])

    @property
    def class_count(self) -> int:
        return len(self.class_names)

    @staticmethod
    def preprocess(image: Image.Image) -> torch.Tensor:
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            background = Image.new("RGB", image.size, (255, 255, 255))
            rgba_image = image.convert("RGBA")
            background.paste(rgba_image, mask=rgba_image.split()[-1])
            image = background

        grayscale = image.convert("L")
        pixels = np.array(grayscale, dtype=np.float32)

        border_pixels = np.concatenate(
            [pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]]
        )
        if border_pixels.mean() > 127:
            pixels = 255.0 - pixels

        minimum, maximum = pixels.min(), pixels.max()
        if maximum > minimum:
            pixels = (pixels - minimum) / (maximum - minimum) * 255.0

        ys, xs = np.nonzero(pixels > 20)
        normalized = Image.fromarray(pixels.astype(np.uint8))

        if len(xs) == 0:
            cropped = normalized
        else:
            padding = int(0.15 * max(xs.max() - xs.min(), ys.max() - ys.min(), 1))
            x0 = max(int(xs.min()) - padding, 0)
            x1 = min(int(xs.max()) + padding + 1, pixels.shape[1])
            y0 = max(int(ys.min()) - padding, 0)
            y1 = min(int(ys.max()) + padding + 1, pixels.shape[0])
            cropped = normalized.crop((x0, y0, x1, y1))

        square_size = max(cropped.size)
        square = Image.new("L", (square_size, square_size), color=0)
        square.paste(
            cropped,
            ((square_size - cropped.width) // 2, (square_size - cropped.height) // 2),
        )
        resized = square.resize((32, 32), Image.LANCZOS)

        tensor = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict[str, object]]:
        tensor = self.preprocess(image).to(self.device)
        logits = self.model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        top_indices = probabilities.argsort()[::-1][:top_k]

        return [
            {
                "class": self.class_names[index],
                "label": display_name(self.class_names[index]),
                "confidence": float(probabilities[index]),
            }
            for index in top_indices
        ]

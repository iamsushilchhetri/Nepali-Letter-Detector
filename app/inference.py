import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.labels import display_name
from app.model import DevanagariCNN

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class Predictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        classes_path = models_dir / "classes.json"
        weights_path = models_dir / "model.pt"
        if not classes_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {models_dir}. Run `uv run scripts/train.py` first."
            )

        self.classes = json.loads(classes_path.read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DevanagariCNN(num_classes=len(self.classes))
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def preprocess(image: Image.Image) -> torch.Tensor:
        """Center the drawn glyph and resize to the 32x32 format the model was trained on.

        Accepts input from three sources -- the drawing canvas (black background,
        white strokes, matching the training data directly), an uploaded photo, or a
        camera capture (typically dark ink on a light/paper background, the opposite
        polarity). Background polarity is auto-detected from the border pixels and
        the image is normalized to the training convention before cropping.
        """
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image.convert("RGBA"), mask=image.convert("RGBA").split()[-1])
            image = background

        gray = image.convert("L")
        arr = np.array(gray, dtype=np.float32)

        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        if border.mean() > 127:  # light background (photo/upload) -> invert to match training data
            arr = 255.0 - arr

        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        gray = Image.fromarray(arr.astype(np.uint8))

        ys, xs = np.nonzero(arr > 20)
        if len(xs) == 0:
            cropped = gray
        else:
            pad = int(0.15 * max(xs.max() - xs.min(), ys.max() - ys.min(), 1))
            x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, arr.shape[1])
            y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, arr.shape[0])
            cropped = gray.crop((x0, y0, x1, y1))

        side = max(cropped.size)
        square = Image.new("L", (side, side), color=0)
        square.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
        resized = square.resize((32, 32), Image.LANCZOS)

        tensor = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0).unsqueeze(0)  # 1x1x32x32

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5) -> list[dict]:
        x = self.preprocess(image).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        top_idx = probs.argsort()[::-1][:top_k]
        return [
            {
                "class": self.classes[i],
                "label": display_name(self.classes[i]),
                "confidence": float(probs[i]),
            }
            for i in top_idx
        ]

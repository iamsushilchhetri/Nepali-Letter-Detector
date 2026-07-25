from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nepali_letter_detector.inference import Predictor


class InferencePreprocessTests(unittest.TestCase):
    def test_preprocess_output_shape_and_range(self) -> None:
        image = Image.new("L", (280, 280), color=0)
        for x in range(100, 180):
            for y in range(100, 180):
                image.putpixel((x, y), 255)

        tensor = Predictor.preprocess(image)
        self.assertEqual(tuple(tensor.shape), (1, 1, 32, 32))
        self.assertGreaterEqual(float(tensor.min()), -1.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_preprocess_handles_blank_canvas(self) -> None:
        image = Image.new("L", (280, 280), color=0)
        tensor = Predictor.preprocess(image)
        self.assertEqual(tuple(tensor.shape), (1, 1, 32, 32))

    def test_preprocess_normalizes_photo_polarity_like_canvas(self) -> None:
        size = 280
        canvas_style = Image.new("L", (size, size), color=0)
        photo_style = Image.new("RGB", (size, size), color=(255, 255, 255))
        for x in range(100, 180):
            for y in range(100, 180):
                canvas_style.putpixel((x, y), 255)
                photo_style.putpixel((x, y), (0, 0, 0))

        tensor_canvas = Predictor.preprocess(canvas_style)
        tensor_photo = Predictor.preprocess(photo_style)
        self.assertTrue(torch.allclose(tensor_canvas, tensor_photo, atol=1e-4))

    def test_preprocess_handles_transparent_png(self) -> None:
        image = Image.new("RGBA", (280, 280), color=(0, 0, 0, 0))
        for x in range(100, 180):
            for y in range(100, 180):
                image.putpixel((x, y), (20, 20, 20, 255))

        tensor = Predictor.preprocess(image)
        self.assertEqual(tuple(tensor.shape), (1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()

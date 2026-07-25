from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nepali_letter_detector.model import NepaliLetterCNN


class ModelTests(unittest.TestCase):
    def test_forward_pass_output_shape(self) -> None:
        model = NepaliLetterCNN(num_classes=46)
        inputs = torch.randn(4, 1, 32, 32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (4, 46))

    def test_custom_num_classes(self) -> None:
        model = NepaliLetterCNN(num_classes=10)
        inputs = torch.randn(2, 1, 32, 32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 10))


if __name__ == "__main__":
    unittest.main()

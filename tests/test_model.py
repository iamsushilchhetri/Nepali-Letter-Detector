import torch

from app.model import DevanagariCNN


def test_forward_pass_output_shape():
    model = DevanagariCNN(num_classes=46)
    x = torch.randn(4, 1, 32, 32)
    out = model(x)
    assert out.shape == (4, 46)


def test_custom_num_classes():
    model = DevanagariCNN(num_classes=10)
    x = torch.randn(2, 1, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)

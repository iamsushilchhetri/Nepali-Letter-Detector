import torch
from PIL import Image

from app.inference import Predictor


def test_preprocess_output_shape_and_range():
    image = Image.new("L", (280, 280), color=0)
    for x in range(100, 180):
        for y in range(100, 180):
            image.putpixel((x, y), 255)

    tensor = Predictor.preprocess(image)
    assert tensor.shape == (1, 1, 32, 32)
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_preprocess_handles_blank_canvas():
    image = Image.new("L", (280, 280), color=0)
    tensor = Predictor.preprocess(image)
    assert tensor.shape == (1, 1, 32, 32)


def test_preprocess_normalizes_photo_polarity_like_canvas():
    """A photo (light background, dark ink) should preprocess to the same tensor
    as an equivalent canvas drawing (dark background, light strokes)."""
    size = 280
    canvas_style = Image.new("L", (size, size), color=0)
    photo_style = Image.new("RGB", (size, size), color=(255, 255, 255))
    for x in range(100, 180):
        for y in range(100, 180):
            canvas_style.putpixel((x, y), 255)
            photo_style.putpixel((x, y), (0, 0, 0))

    t_canvas = Predictor.preprocess(canvas_style)
    t_photo = Predictor.preprocess(photo_style)
    assert torch.allclose(t_canvas, t_photo, atol=1e-4)


def test_preprocess_handles_transparent_png():
    image = Image.new("RGBA", (280, 280), color=(0, 0, 0, 0))
    for x in range(100, 180):
        for y in range(100, 180):
            image.putpixel((x, y), (20, 20, 20, 255))

    tensor = Predictor.preprocess(image)
    assert tensor.shape == (1, 1, 32, 32)

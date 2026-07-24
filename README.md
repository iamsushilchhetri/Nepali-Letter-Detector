# Nepali Letter Detector

Draw a Devanagari letter or digit, upload a photo of one, or snap it with your camera —
a CNN predicts which of Nepali's 46 handwritten characters it is, in real time.

## How it works

- **Model**: a small CNN (PyTorch) trained on the [UCI Devanagari Handwritten Character
  Dataset](https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset)
  — 36 consonants + 10 digits, 92,000 labeled 32x32 images. **98.3% accuracy** on the
  13,800-image held-out test set.
- **Backend**: Flask REST API (`/predict`) that takes a base64 PNG (from the canvas, an
  uploaded photo, or a camera frame), auto-detects whether it's light-background/dark-ink
  (a photo) or dark-background/light-strokes (the drawing canvas), normalizes it to match
  the training distribution, crops/centers/rescales to 32x32, and returns the top-5
  predicted characters with confidence scores.
- **Frontend**: a plain HTML5 canvas + vanilla JS — no build step. Three ways to get a
  character onto the canvas: freehand drawing, image upload, or live camera capture
  (`getUserMedia`, falling back to the device's native camera picker where unsupported).

## Project layout

```
app/
  main.py         Flask app & routes
  inference.py    image preprocessing + model inference (Predictor)
  model.py        CNN architecture
  labels.py       folder-name -> Devanagari glyph mapping
  templates/      index.html
  static/         css/js for the drawing canvas
scripts/
  train.py        training script (torchvision ImageFolder + CNN)
models/           trained weights (model.pt) + class order (classes.json)
tests/            pytest unit tests
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync
```

## Train the model

1. Download the dataset and extract it so you have `data/raw/Train/` and `data/raw/Test/`,
   each containing 46 class subfolders (`character_1_ka`, ..., `digit_9`):

   ```bash
   curl -L -o dataset.zip "https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip"
   unzip dataset.zip -d data/raw_tmp
   mv data/raw_tmp/DevanagariHandwrittenCharacterDataset/* data/raw/
   ```

2. Train:

   ```bash
   uv run scripts/train.py --epochs 10
   ```

   This saves the best checkpoint to `models/model.pt` and the class order to
   `models/classes.json`. A pretrained checkpoint is already committed to this repo,
   so this step is optional unless you want to retrain.

## Run the app

```bash
uv run flask --app app.main run --debug
```

Open http://127.0.0.1:5000, draw a letter (or upload/capture a photo of one), and click **Predict**.

## Run with Docker

```bash
docker build -t nepali-letter-detector .
docker run -p 8000:8000 nepali-letter-detector
```

## Tests

```bash
uv run pytest
uv run ruff check .
```

## Tech stack

Python, PyTorch, torchvision, Flask, Docker, GitHub Actions, HTML/CSS/JS (Canvas API).

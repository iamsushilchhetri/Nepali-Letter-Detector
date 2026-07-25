# Nepali Handwritten Letter Detector

This project recognizes handwritten Nepali Devanagari letters and digits with a custom PyTorch convolutional neural network.

The app lets a user:

- draw a character on a browser canvas
- upload a handwritten image
- get the predicted class
- see ranked confidence scores


The classifier in `src/nepali_letter_detector/model.py` is my own CNN built from scratch with basic PyTorch layers:

- `Conv2d`
- `BatchNorm2d`
- `ReLU`
- `MaxPool2d`
- `Dropout`
- `Linear`

No pretrained model or transfer learning is used.

## Project structure

```text
Nepali-Letter-Detector/
|-- app/
|   |-- main.py
|   |-- static/
|   |   |-- css/
|   |   |   `-- style.css
|   |   `-- js/
|   |       `-- app.js
|   `-- templates/
|       `-- index.html
|-- data/
|   |-- downloads/
|   |-- train/
|   |-- validation/
|   `-- test/
|-- models/
|   |-- classes.json
|   |-- model.pt
|   `-- checkpoints/
|-- notebooks/
|-- scripts/
|   |-- import_uci_dataset.py
|   |-- predict_image.py
|   |-- run_training.py
|   |-- run_web_app.py
|   |-- verify_environment.py
|   |-- verify_training_pipeline.py
|   `-- verify_web_app.py
|-- src/
|   `-- nepali_letter_detector/
|       |-- __init__.py
|       |-- config.py
|       |-- data.py
|       |-- inference.py
|       |-- labels.py
|       |-- model.py
|       `-- training.py
|-- tests/
|-- README.md
`-- requirements.txt
```

## Why the important files exist

- `src/nepali_letter_detector/model.py`: defines the custom CNN.
- `src/nepali_letter_detector/data.py`: loads class folders and builds dataloaders.
- `src/nepali_letter_detector/training.py`: training and evaluation helpers.
- `src/nepali_letter_detector/inference.py`: preprocessing and prediction logic.
- `src/nepali_letter_detector/labels.py`: maps class folders to Nepali glyphs and romanized names.
- `app/main.py`: Flask routes for the page and prediction API.
- `app/templates/index.html`: page layout.
- `app/static/js/app.js`: browser-side drawing, upload, and prediction flow.
- `scripts/run_training.py`: trains the model locally.
- `scripts/predict_image.py`: predicts from a single image file in the terminal.

## Dataset layout

Each split must contain one folder per class label:

```text
data/
|-- train/
|   |-- character_1_ka/
|   |-- character_2_kha/
|   `-- digit_0/
|-- validation/
|   |-- character_1_ka/
|   |-- character_2_kha/
|   `-- digit_0/
`-- test/
    |-- character_1_ka/
    |-- character_2_kha/
    `-- digit_0/
```

The project currently contains a real imported dataset in this format.

## Current model artifacts

The app looks for model files in this order:

1. `models/checkpoints/best_model.pt`
2. `models/checkpoints/final_model.pt`
3. `models/model.pt`

Latest verified local training run on July 25, 2026:

- epochs: 2
- training images: 70,380
- validation images: 7,820
- validation accuracy: 97.15%

## Setup

Create and activate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run verification

```powershell
python scripts/verify_environment.py
python scripts/verify_training_pipeline.py
python scripts/verify_web_app.py
python -m unittest discover -s tests -v
```

## Train locally

```powershell
python scripts/run_training.py --epochs 10 --batch-size 16 --image-size 32
```

## Run the web app

```powershell
python scripts/run_web_app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Predict from the command line

```powershell
python scripts/predict_image.py path\to\image.png
```

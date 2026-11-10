# Simple Pothole Image Classifier

This is a small final-output project made with Python and scikit-learn. It answers one question:

> Does this uploaded road image contain a pothole?

The output is either **Pothole** or **No Pothole**, with a confidence percentage. It does not locate potholes with bounding boxes and does not use a camera, GPS, maps, or YOLO.

## Why this fits the course instructions

- It uses Python with scikit-learn in VS Code.
- It trains a real model from labeled images.
- It predicts new images that were not used for training.
- The model and features are small enough to explain.
- The interface accepts uploaded images, so a camera is not required.

## How it works

1. Each image is reduced to a 16 x 16 thumbnail.
2. The program records grayscale pixels, simple edges, and basic color statistics.
3. A scikit-learn logistic-regression model learns which feature patterns are associated with potholes.
4. The trained model returns a Pothole or No Pothole prediction.

For the reused YOLO-format dataset, a non-empty annotation file means **Pothole** and an empty annotation file means **No Pothole**. Bounding-box coordinates are not used.

## Setup

### Easiest method on Windows

Double-click `RUN_APP.bat`. On the first run, it creates a local virtual
environment and installs the required packages. Internet access is needed only
for that first package installation. After setup, the application opens
automatically.

The upload application uses `model_runtime.json`, a lightweight export of the
same StandardScaler and logistic-regression model stored in `model.joblib`.
This keeps the first-time installation small. The original scikit-learn model
and training code are still included for review and retraining.

Two ready-to-use road images are included in the `sample_inputs` folder. Use
the **Choose Image and Predict** button to test either image.

### Manual method

Open this folder in VS Code, then run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the model

Training requires the additional scientific Python packages:

```powershell
python -m pip install -r requirements-training.txt
```

The default dataset location matches the supplied workspace:

```powershell
python train.py
```

For a different YOLO-format dataset:

```powershell
python train.py --dataset "C:\path\to\dataset"
```

Training creates:

- `model.joblib` - the trained scikit-learn model
- `metrics.json` - training, validation, and unseen-test measurements
- `sample_predictions.csv` - sample unseen-test predictions

## Run the upload interface

```powershell
python app.py
```

Drag a road image onto the large drop zone, or click **Choose Image and
Predict** and select one. You can drop a local image file from File Explorer;
supported browsers can also provide a direct image URL when you drag an image
from a web page.

## Command-line prediction

```powershell
python predict.py "C:\path\to\new_road_image.jpg"
```

## Honest limitation

This is an image-level classifier, not an object detector. It says whether an entire image appears to contain a pothole, but it cannot draw a box around the pothole. Its predictions may also be less reliable for images that differ from the training data, especially unusual lighting, rain, shadows, or unfamiliar road surfaces.


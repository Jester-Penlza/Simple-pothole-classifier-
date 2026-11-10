"""Train and evaluate a simple scikit-learn pothole image classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import extract_features


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = (
    PROJECT_DIR.parent
    / "POTHOLE THESIS-20261012T061637Z-1-001"
    / "POTHOLE THESIS"
    / "Real-Time-Pothole-Detection-System"
    / "roboflow_download"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_split(dataset_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    image_dir = dataset_dir / split / "images"
    label_dir = dataset_dir / split / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"Missing images or labels for split: {split}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    features: list[np.ndarray] = []
    labels: list[int] = []
    kept_paths: list[Path] = []

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Skipping {image_path.name}: matching annotation not found")
            continue

        # A non-empty YOLO annotation means the image contains at least one pothole.
        has_pothole = bool(label_path.read_text(encoding="utf-8").strip())
        try:
            feature_vector = extract_features(image_path)
        except (OSError, ValueError) as error:
            print(f"Skipping unreadable image {image_path.name}: {error}")
            continue

        features.append(feature_vector)
        labels.append(int(has_pothole))
        kept_paths.append(image_path)

    if not features:
        raise ValueError(f"No usable images were found in {image_dir}")

    return np.vstack(features), np.asarray(labels, dtype=np.int64), kept_paths


def evaluate(model: Pipeline, x_data: np.ndarray, y_data: np.ndarray) -> dict:
    predictions = model.predict(x_data)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_data, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_data, predictions)),
        "pothole_precision": float(precision),
        "pothole_recall": float(recall),
        "pothole_f1": float(f1),
        "confusion_matrix": confusion_matrix(y_data, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_data,
            predictions,
            labels=[0, 1],
            target_names=["No Pothole", "Pothole"],
            zero_division=0,
            output_dict=True,
        ),
    }


def write_sample_predictions(
    output_path: Path,
    model: Pipeline,
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_paths: list[Path],
    limit: int = 12,
) -> None:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Choose a repeatable mix of positive and negative test examples.
    chosen: list[int] = []
    for target_class in (1, 0):
        chosen.extend(np.where(y_test == target_class)[0][: limit // 2].tolist())

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["image", "actual", "predicted", "pothole_probability", "correct"]
        )
        for index in chosen:
            writer.writerow(
                [
                    test_paths[index].name,
                    "Pothole" if y_test[index] else "No Pothole",
                    "Pothole" if predictions[index] else "No Pothole",
                    f"{probabilities[index]:.4f}",
                    bool(predictions[index] == y_test[index]),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-output", type=Path, default=PROJECT_DIR / "model.joblib")
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    print(f"Dataset: {dataset_dir}")

    x_train, y_train, _ = load_split(dataset_dir, "train")
    x_valid, y_valid, _ = load_split(dataset_dir, "valid")
    x_test, y_test, test_paths = load_split(dataset_dir, "test")

    print(
        "Loaded "
        f"{len(y_train)} training, {len(y_valid)} validation, and {len(y_test)} test images."
    )

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    # Strong regularization keeps this small model from memorizing
                    # the training images. The value was chosen on validation data.
                    C=0.003,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    metrics = {
        "model": "StandardScaler + LogisticRegression",
        "decision_threshold": 0.5,
        "class_names": ["No Pothole", "Pothole"],
        "split_counts": {
            "train": int(len(y_train)),
            "validation": int(len(y_valid)),
            "test": int(len(y_test)),
        },
        "train": evaluate(model, x_train, y_train),
        "validation": evaluate(model, x_valid, y_valid),
        "test": evaluate(model, x_test, y_test),
    }

    bundle = {
        "model": model,
        "class_names": ["No Pothole", "Pothole"],
        "feature_description": (
            "16x16 grayscale pixels, horizontal and vertical edge strengths, "
            "RGB histograms, and RGB mean/standard-deviation values"
        ),
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)

    scaler = model.named_steps["scale"]
    classifier = model.named_steps["classifier"]
    runtime_model = {
        "model": "StandardScaler + LogisticRegression",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
    }
    runtime_path = PROJECT_DIR / "model_runtime.json"
    runtime_path.write_text(json.dumps(runtime_model), encoding="utf-8")

    metrics_path = PROJECT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_sample_predictions(
        PROJECT_DIR / "sample_predictions.csv", model, x_test, y_test, test_paths
    )

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {args.model_output}")
    print(f"Saved lightweight runtime model: {runtime_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()


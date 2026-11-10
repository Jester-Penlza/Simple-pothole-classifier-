"""Predict whether one or more uploaded road images contain a pothole."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from features import extract_features


PROJECT_DIR = Path(__file__).resolve().parent


def load_bundle(model_path: Path) -> dict:
    runtime_path = model_path.with_name("model_runtime.json")

    if model_path.exists():
        try:
            import joblib

            return joblib.load(model_path)
        except Exception:
            # A lightweight runtime may not include scikit-learn. In that case,
            # use the exported scaler and logistic-regression coefficients.
            pass

    if runtime_path.exists():
        return {"runtime_model": json.loads(runtime_path.read_text(encoding="utf-8"))}

    raise FileNotFoundError(
        f"Model not found. Expected {model_path} or {runtime_path}."
    )


def predict_image(bundle: dict, image_path: Path) -> tuple[str, float]:
    feature_vector = extract_features(image_path)
    runtime_model = bundle.get("runtime_model")
    if runtime_model is not None:
        standardized = [
            (value - mean) / scale
            for value, mean, scale in zip(
                feature_vector,
                runtime_model["scaler_mean"],
                runtime_model["scaler_scale"],
            )
        ]
        logit = runtime_model["intercept"] + sum(
            coefficient * value
            for coefficient, value in zip(
                runtime_model["coefficients"], standardized
            )
        )
        if logit >= 0:
            pothole_probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exp_logit = math.exp(logit)
            pothole_probability = exp_logit / (1.0 + exp_logit)
    else:
        pothole_probability = float(
            bundle["model"].predict_proba([feature_vector])[0, 1]
        )
    label = "Pothole" if pothole_probability >= 0.5 else "No Pothole"
    confidence = pothole_probability if label == "Pothole" else 1.0 - pothole_probability
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="Image file(s) to classify")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "model.joblib")
    args = parser.parse_args()

    bundle = load_bundle(args.model)
    for image_path in args.images:
        label, confidence = predict_image(bundle, image_path)
        print(f"{image_path}: {label} ({confidence:.1%} confidence)")


if __name__ == "__main__":
    main()


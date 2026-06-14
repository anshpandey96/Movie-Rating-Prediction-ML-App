from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config, project_path


def load_model(model_path: str | Path | None = None) -> dict:
    config = load_config()
    path = Path(model_path) if model_path else project_path(config["paths"]["model"])
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_rating(input_data: dict, model_bundle: dict | None = None) -> float:
    bundle = model_bundle or load_model()
    features = bundle["features"]
    row = pd.DataFrame([{feature: input_data.get(feature) for feature in features}])
    prediction = float(bundle["pipeline"].predict(row)[0])
    return round(max(0.0, min(10.0, prediction)), 2)

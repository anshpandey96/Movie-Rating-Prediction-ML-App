from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config import load_config, project_path
from src.data_loader import load_raw_data, save_processed_data, standardize_movie_data
from src.logger import get_logger
from src.preprocessing import build_preprocessor, clean_movie_data


logger = get_logger(__name__)


def _get_models(random_state: int) -> dict[str, object]:
    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=8),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            max_depth=12,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    except ImportError:
        logger.warning("XGBoost is not installed. Skipping XGBoost model.")
    return models


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def prepare_dataset() -> pd.DataFrame:
    raw_df = load_raw_data()
    standardized_df = standardize_movie_data(raw_df)
    clean_df = clean_movie_data(standardized_df)
    save_processed_data(clean_df)
    return clean_df


def train_models() -> tuple[Pipeline, pd.DataFrame, Path]:
    warnings.filterwarnings("ignore", category=UserWarning)
    config = load_config()
    random_state = config["project"]["random_state"]
    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    all_features = numeric_features + categorical_features

    df = prepare_dataset()
    X = df[all_features]
    y = df["rating"]

    test_size = config["training"]["test_size"]
    if len(df) < 30:
        logger.warning("Small dataset detected. Metrics are illustrative until Kaggle data is added.")
        test_size = 0.25

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    results = []
    trained_pipelines: dict[str, Pipeline] = {}

    for model_name, model in _get_models(random_state).items():
        logger.info("Training %s", model_name)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = np.clip(pipeline.predict(X_test), 0, 10)
        model_metrics = _metrics(y_test, predictions)
        results.append({"model": model_name, **model_metrics})
        trained_pipelines[model_name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(
        by=["r2", "rmse"],
        ascending=[False, True],
    )

    best_model_name = metrics_df.iloc[0]["model"]
    best_pipeline = trained_pipelines[str(best_model_name)]

    metrics_path = project_path(config["paths"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    model_path = project_path(config["paths"]["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "features": all_features,
            "metrics": metrics_df.to_dict(orient="records"),
            "best_model": best_model_name,
        },
        model_path,
    )
    logger.info("Best model: %s saved to %s", best_model_name, model_path)
    return best_pipeline, metrics_df, model_path


if __name__ == "__main__":
    _, metrics, saved_model_path = train_models()
    print(metrics.to_string(index=False))
    print(f"\nSaved best model to: {saved_model_path}")

from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "Movie Rating Prediction Using Python",
        "random_state": 42,
    },
    "paths": {
        "raw_data": "data/raw/tmdb_movies.csv",
        "fallback_data": "data/raw/sample_movies.csv",
        "processed_data": "data/processed/cleaned_movies.csv",
        "model": "models/best_movie_rating_model.pkl",
        "metrics": "reports/model_metrics.csv",
        "figures_dir": "reports/figures",
    },
    "features": {
        "numeric": ["runtime", "vote_count", "popularity", "budget", "release_year"],
        "categorical": ["main_genre", "director", "lead_actor"],
    },
    "training": {
        "test_size": 0.2,
        "cv_folds": 5,
    },
}


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load project configuration from YAML, with a no-dependency fallback."""
    try:
        import yaml

        with config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except ModuleNotFoundError:
        return DEFAULT_CONFIG.copy()


def project_path(relative_path: str) -> Path:
    """Resolve a project-relative path."""
    return ROOT_DIR / relative_path

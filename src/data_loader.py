import ast
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import load_config, project_path
from src.logger import get_logger


logger = get_logger(__name__)


TARGET_CANDIDATES = ["vote_average", "rating", "imdb_rating", "score", "Rating"]


def _first_existing(columns: Iterable[str], candidates: list[str]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def _parse_people(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            names = []
            for item in parsed:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("actor") or item.get("cast")
                    if name:
                        names.append(str(name))
                else:
                    names.append(str(item))
            return names
    except (ValueError, SyntaxError):
        pass
    separator = "|" if "|" in text else ","
    return [item.strip() for item in text.split(separator) if item.strip()]


def _parse_genres(value: object) -> list[str]:
    return _parse_people(value)


def _extract_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        text = str(value)
        return int(text[:4]) if text[:4].isdigit() else None
    return int(parsed.year)


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load Kaggle data, falling back to the bundled sample dataset."""
    config = load_config()
    raw_path = project_path(path or config["paths"]["raw_data"])
    fallback_path = project_path(config["paths"]["fallback_data"])
    data_path = raw_path if raw_path.exists() else fallback_path

    logger.info("Loading data from %s", data_path)
    return pd.read_csv(data_path)


def standardize_movie_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common TMDB/IMDb column variants into a consistent schema."""
    data = df.copy()
    target_column = _first_existing(data.columns, TARGET_CANDIDATES)
    if target_column is None:
        raise ValueError(f"No target column found. Expected one of: {TARGET_CANDIDATES}")

    column_map = {
        target_column: "rating",
        _first_existing(data.columns, ["runtime", "duration", "Runtime"]): "runtime",
        _first_existing(data.columns, ["vote_count", "votes", "num_votes", "Votes"]): "vote_count",
        _first_existing(data.columns, ["popularity", "Popularity"]): "popularity",
        _first_existing(data.columns, ["budget", "Budget"]): "budget",
        _first_existing(data.columns, ["release_date", "year", "release_year", "Year"]): "release_date",
        _first_existing(data.columns, ["genres", "genre", "Genre"]): "genres",
        _first_existing(data.columns, ["director", "Director"]): "director",
        _first_existing(data.columns, ["cast", "actors", "stars", "Cast"]): "cast",
    }
    column_map = {source: target for source, target in column_map.items() if source}
    data = data.rename(columns=column_map)

    required_defaults = {
        "runtime": 0,
        "vote_count": 0,
        "popularity": 0,
        "budget": 0,
        "genres": "Unknown",
        "director": "Unknown",
        "cast": "Unknown",
    }
    for column, default in required_defaults.items():
        if column not in data.columns:
            data[column] = default

    if "release_date" not in data.columns:
        data["release_date"] = None

    data["main_genre"] = data["genres"].apply(lambda value: (_parse_genres(value) or ["Unknown"])[0])
    data["lead_actor"] = data["cast"].apply(lambda value: (_parse_people(value) or ["Unknown"])[0])
    data["director"] = data["director"].apply(lambda value: (_parse_people(value) or ["Unknown"])[0])
    data["release_year"] = data["release_date"].apply(_extract_year)

    output_columns = [
        "runtime",
        "vote_count",
        "popularity",
        "budget",
        "release_year",
        "main_genre",
        "director",
        "lead_actor",
        "rating",
    ]
    return data[output_columns]


def save_processed_data(df: pd.DataFrame, path: str | None = None) -> Path:
    config = load_config()
    output_path = project_path(path or config["paths"]["processed_data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved processed data to %s", output_path)
    return output_path

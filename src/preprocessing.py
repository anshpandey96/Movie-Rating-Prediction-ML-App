import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_movie_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, invalid targets, and normalize obvious numeric issues."""
    data = df.copy()
    data = data.drop_duplicates()
    data["rating"] = pd.to_numeric(data["rating"], errors="coerce")
    data = data.dropna(subset=["rating"])
    data = data[(data["rating"] >= 0) & (data["rating"] <= 10)]

    numeric_columns = ["runtime", "vote_count", "popularity", "budget", "release_year"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[data[column] < 0, column] = None

    categorical_columns = ["main_genre", "director", "lead_actor"]
    for column in categorical_columns:
        data[column] = data[column].fillna("Unknown").astype(str).str.strip()
        data.loc[data[column] == "", column] = "Unknown"

    return data


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical movie features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import load_config, project_path
from src.data_loader import load_raw_data, save_processed_data, standardize_movie_data
from src.preprocessing import clean_movie_data


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def run_eda() -> pd.DataFrame:
    """Run EDA, save plots, and return a compact insight table."""
    config = load_config()
    figures_dir = project_path(config["paths"]["figures_dir"])

    raw_df = load_raw_data()
    standardized_df = standardize_movie_data(raw_df)
    clean_df = clean_movie_data(standardized_df)
    save_processed_data(clean_df)

    insights = pd.DataFrame(
        {
            "metric": [
                "raw_rows",
                "raw_columns",
                "duplicate_rows",
                "clean_rows",
                "average_rating",
                "highest_rating",
                "lowest_rating",
            ],
            "value": [
                len(raw_df),
                raw_df.shape[1],
                raw_df.duplicated().sum(),
                len(clean_df),
                round(clean_df["rating"].mean(), 2),
                clean_df["rating"].max(),
                clean_df["rating"].min(),
            ],
        }
    )

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(clean_df["rating"], bins=20, kde=True, color="#e50914")
    plt.title("Movie Rating Distribution")
    plt.xlabel("Rating")
    _save_plot(figures_dir / "rating_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=clean_df, x="vote_count", y="rating", hue="main_genre")
    plt.title("Votes vs Rating")
    plt.xlabel("Vote Count")
    plt.ylabel("Rating")
    _save_plot(figures_dir / "votes_vs_rating.png")

    numeric_columns = ["runtime", "vote_count", "popularity", "budget", "release_year", "rating"]
    plt.figure(figsize=(9, 6))
    sns.heatmap(clean_df[numeric_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numeric Feature Correlation")
    _save_plot(figures_dir / "correlation_heatmap.png")

    top_genres = clean_df["main_genre"].value_counts().head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=top_genres.values, y=top_genres.index, color="#e50914")
    plt.title("Top Genres by Movie Count")
    plt.xlabel("Count")
    _save_plot(figures_dir / "top_genres.png")

    insights_path = project_path("reports/eda_insights.csv")
    insights.to_csv(insights_path, index=False)
    return insights


if __name__ == "__main__":
    print(run_eda().to_string(index=False))

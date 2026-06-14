# Movie Rating Prediction Using Python

A complete machine learning project that predicts movie ratings on a 0-10 scale using historical movie metadata such as genre, director, cast, duration, votes, popularity, budget, and release year.

## Complete Folder Structure

```text
movie prediction project/
├── app.py
├── config/
│   └── config.yaml
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       ├── .gitkeep
│       └── sample_movies.csv
├── logs/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── reports/
│   ├── figures/
│   │   └── .gitkeep
│   ├── eda_insights.csv
│   └── model_metrics.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── logger.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── templates/
│   ├── index.html
│   └── result.html
├── .gitignore
├── README.md
└── requirements.txt
```

## Features

- Industry-style modular project structure
- EDA for nulls, duplicates, rating distribution, genre counts, vote-rating trends, and correlations
- Data cleaning, feature extraction, encoding, scaling, and feature engineering
- Multiple regression models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Model comparison using R2, MAE, MSE, and RMSE
- Automatic best-model selection
- Trained model saved with Joblib as `models/best_movie_rating_model.pkl`
- Flask web app with a modern Netflix-style interface

## Dataset

Place your Kaggle TMDB/IMDb CSV file at:

```text
data/raw/tmdb_movies.csv
```

The code supports common column names from movie datasets:

- Target: `vote_average`, `rating`, `imdb_rating`, `score`
- Runtime: `runtime`, `duration`
- Votes: `vote_count`, `votes`, `num_votes`
- Genre: `genres`, `genre`
- Cast: `cast`, `actors`, `stars`
- Director: `director`
- Release date/year: `release_date`, `year`, `release_year`

If `data/raw/tmdb_movies.csv` is missing, the project uses `data/raw/sample_movies.csv` so the pipeline can still run for demonstration.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run EDA

```bash
python -m src.eda
```

Outputs:

- `data/processed/cleaned_movies.csv`
- `reports/eda_insights.csv`
- `reports/figures/rating_distribution.png`
- `reports/figures/votes_vs_rating.png`
- `reports/figures/correlation_heatmap.png`
- `reports/figures/top_genres.png`

## Train Models

```bash
python -m src.train
```

Outputs:

- `reports/model_metrics.csv`
- `models/best_movie_rating_model.pkl`

## Run Flask App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Deployment Notes

For deployment, make sure the trained model file exists:

```text
models/best_movie_rating_model.pkl
```

Recommended production command:

```bash
gunicorn app:app
```

For platforms such as Render or Railway, set the start command to `gunicorn app:app` and include `requirements.txt`.

## Project Workflow

1. Add Kaggle CSV to `data/raw/tmdb_movies.csv`
2. Run EDA with `python -m src.eda`
3. Train models with `python -m src.train`
4. Review `reports/model_metrics.csv`
5. Start Flask app with `python app.py`
6. Enter movie features and get the predicted rating

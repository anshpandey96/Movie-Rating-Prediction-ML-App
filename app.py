try:
    from flask import Flask, render_template, request  # type: ignore[import]
except ImportError as e:
    raise ImportError("Flask is not installed. Please install it using: pip install flask") from e

from src.predict import load_model, predict_rating


app = Flask(__name__)
model_bundle = None
model_error = None

try:
    model_bundle = load_model()
except FileNotFoundError as error:
    model_error = str(error)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_error=model_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model_bundle is None:
        return render_template("index.html", model_error=model_error)

    form_data = {
        "runtime": float(request.form.get("runtime", 0)),
        "vote_count": float(request.form.get("vote_count", 0)),
        "popularity": float(request.form.get("popularity", 0)),
        "budget": float(request.form.get("budget", 0)),
        "release_year": float(request.form.get("release_year", 0)),
        "main_genre": request.form.get("main_genre", "Unknown"),
        "director": request.form.get("director", "Unknown"),
        "lead_actor": request.form.get("lead_actor", "Unknown"),
    }
    rating = predict_rating(form_data, model_bundle)
    return render_template("result.html", rating=rating, movie=form_data)


if __name__ == "__main__":
    app.run(debug=True)

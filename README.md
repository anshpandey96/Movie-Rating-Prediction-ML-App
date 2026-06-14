# 🎬 Movie Rating Prediction Web App

![Uploading ChatGPT Image Jun 14, 2026, 08_27_47 PM.png…]()


## 📌 Overview
This project predicts a movie’s rating (0.0 – 10.0) using **Machine Learning regression models** trained on features like genre, director, lead actor, runtime, release year, votes, popularity, and budget.  
It includes a **Flask web interface** for interactive predictions and is deployment‑ready on **Render.com**.

## ⚙️ Features
- Exploratory Data Analysis (EDA): nulls, duplicates, distributions, correlations
- Feature scaling and one‑hot encoding for numeric & categorical data
- Regression model comparison (R², MAE, MSE, RMSE)
- Automatic best‑model selection with Joblib persistence
- Flask‑based UI for user‑friendly predictions
- Deployment setup with `requirements.txt`, `Procfile`, and `runtime.txt`

## 🗂️ Project Structure

Movie-Rating-Prediction-ML-App/
│── app.py                # Flask main app
│── requirements.txt      # Dependencies
│── Procfile              # For Render deployment
│── runtime.txt           # Python version
│── models/               # Saved joblib models
│── static/               # CSS, JS, images
│── templates/            # HTML files (index.html, result.html)
│── README.md             # Project description
│── data/                 # Dataset (if small, else link in README)

## 🚀 Deployment (Render.com)
1. Push repo to GitHub.  
2. On Render → New Web Service → Connect GitHub repo.  
3. **Build Command:**  

pip install -r requirements.txt

Code
4. **Start Command:**  
gunicorn app:app

Code
5. Add `runtime.txt` → `python-3.10` (or your version).  
6. Deploy → Your app goes live!

## 📷 Demo
### Input Form
![Movie Features Form](static/form.png)

### Prediction Result
![Predicted Rating Result](static/result.png)

## 🛠️ Tech Stack
- Python, Flask  
- Scikit-learn, Pandas, NumPy  
- Joblib (model persistence)  
- HTML, CSS, Bootstrap  

## 📌 Author
👤 **Ansh Pandey**  
- GitHub: [AnshPandey-96](https://github.com/anshpandey96)  
- LinkedIn: [Ansh Pandey](https://www.linkedin.com/in/ansh-pandey-b1ab34333
- 
---

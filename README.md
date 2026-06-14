# 🎬 Movie Rating Prediction Web App

## 📌 Overview
This project predicts a movie’s rating (0.0 – 10.0) using **Machine Learning regression models** trained on features like genre, director, lead actor, runtime, release year, votes, popularity, and budget.  
It includes a **Flask web interface** for interactive predictions and is deployment‑ready on **Render.com**.

<img width="1536" height="1024" alt="ChatGPT Image Jun 14, 2026, 08_27_47 PM" src="https://github.com/user-attachments/assets/846a7930-4068-4083-9b16-1510dd8dfeb6" />

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

<img width="1600" height="888" alt="image" src="https://github.com/user-attachments/assets/fb02b191-b85f-4579-92e4-36863d8184a6" />

### Prediction Result

<img width="1600" height="888" alt="image" src="https://github.com/user-attachments/assets/68d43225-b416-48f1-8cd3-c6b72bb939dd" />

## 🛠️ Tech Stack
- Python, Flask  
- Scikit-learn, Pandas, NumPy  
- Joblib (model persistence)  
- HTML, CSS, Bootstrap  

## 📌 Author
👤 **Ansh Pandey**  
- GitHub: [AnshPandey-96](https://github.com/anshpandey96)  
- LinkedIn: [Ansh Pandey](https://www.linkedin.com/in/ansh-pandey-b1ab34333
---

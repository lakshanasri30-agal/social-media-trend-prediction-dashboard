# Social Media Trend Prediction
## IEEE Research Project — Setup Guide

---

## Files
| File        | Purpose                                      |
|-------------|----------------------------------------------|
| `index.html`| Frontend dashboard (open in browser)         |
| `app.py`    | Python Flask backend with ML models          |

---

## Step 1 — Install Python dependencies

Open a terminal (Ctrl + `) in VS Code and run:

```bash
pip install flask flask-cors scikit-learn numpy pandas joblib
```

---

## Step 2 — Start the backend

In the VS Code terminal:

```bash
python app.py
```

You will see the ML training pipeline run automatically.
When you see **"Running on http://127.0.0.1:5000"**, the backend is ready.

Training takes about 30–60 seconds on first run.
After that, models are saved to `trained_models.pkl` and load instantly.

---

## Step 3 — Open the frontend

- Double-click `index.html` to open it in your browser, OR
- In VS Code, right-click `index.html` → **Open with Live Server** (install the Live Server extension if needed)

---

## API Endpoints

| Method | URL                              | Description                  |
|--------|----------------------------------|------------------------------|
| GET    | `/health`                        | Check if API is running      |
| POST   | `/predict`                       | Predict single tweet         |
| POST   | `/batch_predict`                 | Predict multiple tweets      |
| GET    | `/models`                        | Get all model metrics        |
| GET    | `/model_summary`                 | Paper-ready summary table    |

### Example POST /predict
```json
Request:  { "text": "#AIRevolution new model beats GPT", "model": "svm" }
Response: {
  "prediction": "Trending",
  "confidence": 0.9134,
  "trending_prob": 0.9134,
  "non_trending_prob": 0.0866,
  "model_used": "SVM (RBF Kernel)",
  "model_accuracy": 91.3
}
```

---

## ML Models Used

| Model               | After GridSearchCV |
|---------------------|--------------------|
| Logistic Regression | ~87–89%            |
| SVM (RBF kernel)    | ~89–92% ← Best     |
| Random Forest       | ~87–90%            |

Exact scores depend on the dataset generation seed.

---

## Note on Offline Mode
If the backend is not running, the frontend still works using a
local JavaScript simulation engine. The API status indicator in
the top-right corner shows whether you are connected to the real backend.

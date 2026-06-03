# 🚕 Taxi Fare Prediction — Regression

A production-style ML project that predicts taxi fares from trip features. Part of a 6-project portfolio targeting a junior → mid-level ML engineer signal.

## Live Demo

| | |
|---|---|
| 🖥️ **Streamlit App** | https://taxi-fare-prediction-portfolio.streamlit.app/ |
| 📡 **API Docs** | https://taxi-fare-vo0u.onrender.com/docs |

> ⚠️ Render free tier spins down after inactivity — first request may take ~30s to wake up.

## Tech Stack

| Layer | Tools |
|---|---|
| Training | scikit-learn, XGBoost, Optuna, MLflow |
| Explainability | SHAP |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, Plotly |
| Infra | Docker, docker-compose |
| CI | GitHub Actions |
| Tests | pytest, FastAPI TestClient |

## What it does

- Compares 6 regression models (LinearRegression, Ridge, Lasso, RandomForest, XGBoost, MLP) using 5-fold cross-validation
- Tunes the best model (RandomForest) with Optuna (15 trials)
- Logs all runs, params, and metrics to MLflow for local experiment tracking and comparison
- Serves predictions via a FastAPI endpoint with SHAP explainability per prediction
- Streamlit frontend with sidebar metrics, predicted vs actual scatter plot, and SHAP bar chart per prediction

## Project Structure

```
├── model.py              # Training, CV, Optuna tuning, MLflow logging
├── eda.py                # Exploratory data analysis
├── api.py                # FastAPI — /health /model/info /predict
├── app.py                # Streamlit frontend
├── features.py           # Shared feature engineering (AddFeature transformer)
├── model.pkl             # Serialised best pipeline for deployment
├── metrics.json          # Best model metrics for API /model/info endpoint
├── data/                 # Dataset (taxi_trip_pricing.csv)
├── tests/
│   └── test_api.py       # pytest — health, predict, validation
├── Dockerfile            # API container
├── docker-compose.yml    # MLflow + API services
├── .github/workflows/
│   └── ci.yml            # Install → docker build
└── requirements.txt
```

## Run Locally

**Train the model:**
```bash
python model.py
```

**Start the API:**
```bash
uvicorn api:app --reload --port 8000
```

**Start the Streamlit app:**
```bash
streamlit run app.py
```

**Run with Docker:**
```bash
docker-compose up
```

**Run tests:**
```bash
pytest tests/test_api.py -v
```

## Limitations

- **Dataset is small (~950 rows)** — model performance is limited by data size, not architecture
- **MLflow is local only** — experiment tracking and model registry run locally. For deployment, the best model is exported to `model.pkl` to avoid cross-platform MLflow artifact path issues. In a real system, MLflow would run on a remote server with S3/GCS artifact storage
- **Model auto-registers on every training run** — in production you would manually promote a model to the registry only after reviewing metrics
- **No authentication on /predict** — a production API would require API key or OAuth
- **Render free tier cold starts** — API sleeps after inactivity, first request takes ~30s

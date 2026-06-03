# Taxi Fare Prediction — Regression

A production-style ML project that predicts taxi fares from trip features. 

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
- Logs all runs, params, and metrics to MLflow and registers the best model in the MLflow Model Registry
- Serves predictions via a FastAPI endpoint that loads the registered model on startup
- Explains each prediction with SHAP — shows which features pushed the fare up or down
- Streamlit frontend with sidebar metrics, predicted vs actual scatter plot, and SHAP bar chart

## Project Structure

```
├── model.py              # Training, CV, Optuna tuning, MLflow logging
├── eda.py                # Exploratory data analysis
├── api.py                # FastAPI — /health /model/info /predict
├── app.py                # Streamlit frontend
├── data/                 # Dataset (taxi_trip_pricing.csv)
├── mlruns/               # MLflow tracking and model artifacts
├── tests/
│   └── test_api.py       # pytest — health, predict, validation
├── Dockerfile            # API container
├── docker-compose.yml    # MLflow + API services
├── .github/workflows/
│   └── ci.yml            # Install → test → docker build
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
- **mlruns/ committed to the repo** — in a real system MLflow artifacts would live on a remote store (S3, GCS). Committed here to keep deployment simple for a portfolio project
- **Model auto-registers on every training run** — in production you would manually promote a model to the registry only after reviewing metrics and deciding it's better than the current version
- **No authentication on /predict** — a production API would require API key or OAuth


from fastapi import FastAPI
from typing import Literal
from pydantic import BaseModel
from contextlib import asynccontextmanager
import mlflow
import pandas as pd
import shap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxiFareFeatures(BaseModel):
    Trip_Distance_km:      float
    Time_of_Day:           Literal["Morning", "Afternoon", "Evening", "Night"]
    Day_of_Week:           Literal['Weekday','Weekend']
    Passenger_Count:       float
    Traffic_Conditions:    Literal['Low' ,'Medium' ,'High']
    Weather:               Literal['Clear', 'Rain', 'Snow']
    Base_Fare:             float
    Per_Km_Rate:           float
    Per_Minute_Rate:       float
    Trip_Duration_Minutes: float


#Startup/Shutdown
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = mlflow.sklearn.load_model("models:/taxi-fare-best/latest")
    logger.info("Model loaded from MLflow registry")
    yield


app = FastAPI(title='Taxi Fare Prediction API', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': model is not None}

@app.get('/model/info')
def model_info():
    client = mlflow.tracking.MlflowClient()
    version = client.get_latest_versions("taxi-fare-best")[0]
    run = client.get_run(version.run_id)
    return {
        "model_name": "taxi-fare-best",
        "version": version.version,
        "metrics": run.data.metrics
    }

@app.post('/predict')
def predict(features: TaxiFareFeatures):
    input_df = pd.DataFrame([features.model_dump()])

    prediction = round(float(model.predict(input_df)[0]), 2)

    fe         = model.named_steps["feature_engineer"]
    pre        = model.named_steps["preprocessor"]
    model_step = model.named_steps["model"]

    X_fe       = fe.transform(input_df)
    X_pre      = pre.transform(X_fe)

    explainer  = shap.TreeExplainer(model_step)
    shap_vals  = explainer.shap_values(X_pre)[0]
    feat_names = pre.get_feature_names_out()

    shap_dict  = dict(zip(feat_names.tolist(), [round(float(v), 4) for v in shap_vals]))

    logger.info(f"Prediction request: {features.model_dump()} → {prediction}")
    return {'predicted_fare': prediction, 'shap_values': shap_dict}



    
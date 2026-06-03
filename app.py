import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from features import AddFeature  # noqa: F401 — required for joblib to unpickle the pipeline

API_URL = "https://taxi-fare-vo0u.onrender.com"

st.set_page_config(page_title="Taxi Fare Prediction", page_icon="🚕", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def get_test_predictions(_pipeline):
    df = pd.read_csv("data/taxi_trip_pricing.csv")
    df = df.dropna(subset=["Trip_Price"])
    X, y = df.drop(columns=["Trip_Price"]), df["Trip_Price"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = _pipeline.predict(X_test)
    return y_test.values, y_pred


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("Model Info")

try:
    info = requests.get(f"{API_URL}/model/info", timeout=3).json()
    m = info["metrics"]

    st.sidebar.markdown(f"**Model:** `{info['model_name']}`")
    st.sidebar.divider()

    c1, c2 = st.sidebar.columns(2)
    c1.metric("Train MAE",  f"{m['train_mae']:.2f}")
    c2.metric("Train RMSE", f"{m['train_rmse']:.2f}")
    c1.metric("Test MAE",   f"{m['test_mae']:.2f}")
    c2.metric("Test RMSE",  f"{m['test_rmse']:.2f}")
    st.sidebar.metric("Test R²", f"{m['test_r2']:.4f}")

    st.sidebar.divider()
    pipeline = load_model()
    y_test, y_pred = get_test_predictions(pipeline)

    fig_scatter = px.scatter(
        x=y_test, y=y_pred,
        labels={"x": "Actual Fare", "y": "Predicted Fare"},
        title="Predicted vs Actual (test set)",
        opacity=0.6,
    )
    fig_scatter.add_shape(
        type="line",
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color="red", dash="dash"),
    )
    st.sidebar.plotly_chart(fig_scatter, width='stretch')

except Exception:
    st.sidebar.error("API not reachable — start the API first.")


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🚕 Taxi Fare Prediction")
st.markdown("Fill in the trip details below and click **Predict**.")

with st.form("prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        trip_distance     = st.number_input("Trip Distance (km)",      min_value=0.1,  value=5.0,  step=0.1)
        base_fare         = st.number_input("Base Fare",                min_value=0.0,  value=2.5,  step=0.1)
        per_km_rate       = st.number_input("Per Km Rate",              min_value=0.0,  value=1.5,  step=0.1)
        per_minute_rate   = st.number_input("Per Minute Rate",          min_value=0.0,  value=0.5,  step=0.05)

    with c2:
        trip_duration     = st.number_input("Trip Duration (minutes)",  min_value=1,    value=20,   step=1)
        passenger_count   = st.number_input("Passenger Count",          min_value=1,    max_value=6, value=1, step=1)
        time_of_day       = st.selectbox("Time of Day",   ["Morning", "Afternoon", "Evening", "Night"])
        day_of_week       = st.selectbox("Day of Week",   ["Weekday", "Weekend"])

    with c3:
        traffic           = st.selectbox("Traffic Conditions", ["Low", "Medium", "High"])
        weather           = st.selectbox("Weather",            ["Clear", "Rain", "Snow"])

    submitted = st.form_submit_button("Predict Fare", width='stretch')


if submitted:
    payload = {
        "Trip_Distance_km":     trip_distance,
        "Time_of_Day":          time_of_day,
        "Day_of_Week":          day_of_week,
        "Passenger_Count":      float(passenger_count),
        "Traffic_Conditions":   traffic,
        "Weather":              weather,
        "Base_Fare":            base_fare,
        "Per_Km_Rate":          per_km_rate,
        "Per_Minute_Rate":      per_minute_rate,
        "Trip_Duration_Minutes": float(trip_duration),
    }

    try:
        response   = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        result     = response.json()
        fare       = result["predicted_fare"]
        st.success(f"Estimated Fare: **${fare}**")

        shap_vals  = result["shap_values"]
        shap_df = (
            pd.DataFrame({"Feature": list(shap_vals.keys()), "SHAP Value": list(shap_vals.values())})
            .reindex(pd.Series(list(shap_vals.values())).abs().sort_values(ascending=False).index)
        )

        fig_shap = px.bar(
            shap_df, x="SHAP Value", y="Feature", orientation="h",
            color="SHAP Value", color_continuous_scale="RdBu_r",
            title="Feature Contributions (SHAP) — what drove this prediction",
        )
        fig_shap.update_layout(yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_shap, width='stretch')

    except Exception as e:
        st.error(f"Prediction failed: {e}")

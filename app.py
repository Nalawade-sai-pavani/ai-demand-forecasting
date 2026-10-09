"""Streamlit dashboard for the AI demand forecasting prototype.
Run: streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

FEATS = ["promo", "dow", "month", "doy", "lag7", "lag14", "roll7"]


@st.cache_data
def make_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    t = np.arange(len(dates))
    promo = rng.random(len(dates)) < 0.08
    sales = (200 + 0.05 * t + 30 * np.sin(2 * np.pi * t / 365.25)
             + 25 * (dates.dayofweek >= 5) + 60 * promo
             + rng.normal(0, 12, len(dates)))
    df = pd.DataFrame({"date": dates, "sales": sales.round(),
                       "promo": promo.astype(int)})
    df["dow"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.month
    df["doy"] = df.date.dt.dayofyear
    df["lag7"] = df.sales.shift(7)
    df["lag14"] = df.sales.shift(14)
    df["roll7"] = df.sales.shift(1).rolling(7).mean()
    return df.dropna().reset_index(drop=True)


@st.cache_data
def run_model(trees: int, test_days: int):
    df = make_data()
    train, test = df.iloc[:-test_days], df.iloc[-test_days:]
    model = RandomForestRegressor(n_estimators=trees, random_state=42, n_jobs=-1)
    model.fit(train[FEATS], train.sales)
    pred = model.predict(test[FEATS])
    base = test["lag7"].to_numpy()
    actual = test.sales.to_numpy()
    res = {
        "mae": mean_absolute_error(actual, pred),
        "base_mae": mean_absolute_error(actual, base),
        "mape": float((np.abs(actual - pred) / actual).mean() * 100),
        "base_mape": float((np.abs(actual - base) / actual).mean() * 100),
    }
    chart = pd.DataFrame({"Actual": actual, "AI forecast": pred,
                          "Naive baseline": base}, index=test.date.to_numpy())
    imp = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)
    return res, chart, imp


st.set_page_config(page_title="AI Demand Forecasting", layout="wide")
st.title("AI Demand Forecasting Prototype")
st.caption("Synthetic retail data | Random Forest vs naive baseline")

trees = st.sidebar.slider("Number of trees", 50, 500, 300, 50)
test_days = st.sidebar.slider("Test period (days)", 30, 180, 90, 10)

res, chart, imp = run_model(trees, test_days)
gain = (1 - res["mae"] / res["base_mae"]) * 100

c1, c2, c3 = st.columns(3)
c1.metric("AI model MAE", f"{res['mae']:.2f}")
c2.metric("Baseline MAE", f"{res['base_mae']:.2f}")
c3.metric("Error reduction", f"{gain:.1f}%")
st.write(f"MAPE: AI {res['mape']:.2f}% vs baseline {res['base_mape']:.2f}%")

st.subheader("Forecast vs actual")
st.line_chart(chart)
st.subheader("Feature importance")
st.bar_chart(imp)
st.subheader("Forecast table")
st.dataframe(chart.round(1))
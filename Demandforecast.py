"""AI Demand Forecasting prototype (synthetic retail data).
Compares a naive baseline with a Random Forest model.
Run: python demand_forecast.py
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)

# 1. Generate 3 years of daily sales data (trend + weekly + yearly seasonality + promos)
dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
t = np.arange(len(dates))
promo = rng.random(len(dates)) < 0.08
sales = (200 + 0.05 * t
         + 30 * np.sin(2 * np.pi * t / 365.25)
         + 25 * (dates.dayofweek >= 5)
         + 60 * promo
         + rng.normal(0, 12, len(dates)))
df = pd.DataFrame({"date": dates, "sales": sales.round(), "promo": promo.astype(int)})

# 2. Feature engineering
df["dow"] = df.date.dt.dayofweek
df["month"] = df.date.dt.month
df["doy"] = df.date.dt.dayofyear
df["lag7"] = df.sales.shift(7)
df["lag14"] = df.sales.shift(14)
df["roll7"] = df.sales.shift(1).rolling(7).mean()
df = df.dropna().reset_index(drop=True)

# 3. Time-based split (last 90 days = test)
train, test = df.iloc[:-90], df.iloc[-90:]
feats = ["promo", "dow", "month", "doy", "lag7", "lag14", "roll7"]

# 4. Baseline: same value as 7 days ago
base_pred = test["lag7"]
base_mae = mean_absolute_error(test.sales, base_pred)

# 5. AI model
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(train[feats], train.sales)
pred = model.predict(test[feats])
mae = mean_absolute_error(test.sales, pred)
mape = (np.abs(test.sales - pred) / test.sales).mean() * 100
base_mape = (np.abs(test.sales - base_pred) / test.sales).mean() * 100

print(f"Baseline (naive) MAE : {base_mae:.2f}  MAPE: {base_mape:.2f}%")
print(f"Random Forest   MAE : {mae:.2f}  MAPE: {mape:.2f}%")
print(f"Error reduction     : {(1 - mae/base_mae)*100:.1f}%")
print("Feature importance:")
for f, i in sorted(zip(feats, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f:6s} {i:.3f}")

# 6. Chart
plt.figure(figsize=(9, 4))
plt.plot(test.date, test.sales, label="Actual", color="#1F3A5F")
plt.plot(test.date, pred, label="AI forecast", color="#E07A1F")
plt.plot(test.date, base_pred, label="Naive baseline", color="#999", alpha=.6)
plt.title("Demand forecast vs actual (last 90 days)")
plt.legend(); plt.tight_layout()
plt.savefig("forecast.png", dpi=150)
print("Saved forecast.png")
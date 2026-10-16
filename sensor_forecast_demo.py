"""
Sensor Forecasting Demo
A small time-series forecasting example using XGBoost with lag features.
Part of continuous learning: AI-Powered Time Series Forecasting with Python.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------
# 1. Simulate a sensor time series (hourly readings)
# ---------------------------------------------------------
np.random.seed(42)
hours = 1000
timestamps = pd.date_range("2024-01-01", periods=hours, freq="h")

# Sensor reading = daily cycle + slight drift + noise
daily_cycle = 10 * np.sin(np.arange(hours) / 24 * 2 * np.pi)
drift = np.linspace(0, 3, hours)
noise = np.random.normal(0, 1.5, hours)
values = 50 + daily_cycle + drift + noise

df = pd.DataFrame({"timestamp": timestamps, "sensor_value": values})


# ---------------------------------------------------------
# 2. Feature engineering (lag features — classic TS approach)
# ---------------------------------------------------------
df["lag_1"]   = df["sensor_value"].shift(1)
df["lag_24"]  = df["sensor_value"].shift(24)   # same hour yesterday
df["lag_168"] = df["sensor_value"].shift(168)  # same hour last week
df["hour"]    = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek

df = df.dropna().reset_index(drop=True)


# ---------------------------------------------------------
# 3. Chronological train/test split (NOT random!)
# ---------------------------------------------------------
split_idx = int(len(df) * 0.8)
train, test = df.iloc[:split_idx], df.iloc[split_idx:]

features = ["lag_1", "lag_24", "lag_168", "hour", "dayofweek"]

X_train, y_train = train[features], train["sensor_value"]
X_test,  y_test  = test[features],  test["sensor_value"]


# ---------------------------------------------------------
# 4. Train XGBoost model
# ---------------------------------------------------------
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
model.fit(X_train, y_train)


# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"MAE on test set: {mae:.2f}")


# ---------------------------------------------------------
# 6. Visualise
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(test["timestamp"], y_test.values, label="Actual", linewidth=1.5)
plt.plot(test["timestamp"], preds, label="Predicted", linewidth=1.5, linestyle="--")
plt.title("Sensor Forecasting with XGBoost (lag features)")
plt.xlabel("Time")
plt.ylabel("Sensor Value")
plt.legend()
plt.tight_layout()
plt.savefig("forecast.png", dpi=120)
plt.show()
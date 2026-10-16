# Sensor Forecasting Demo

Small time-series forecasting example using **XGBoost** with lag features — applied to simulated sensor data.

Built as part of continuous learning after completing
*"AI-Powered Time Series Forecasting with Python"* (LinkedIn Learning).

## What it does

- Simulates hourly sensor readings (daily cycle + drift + noise)
- Engineers lag features (1h, 24h, 168h)
- Splits chronologically (no data leakage)
- Trains an XGBoost regressor
- Evaluates with MAE
- Visualises actual vs predicted

## Why it matters

Sensor and telemetry data is **time-series data**.
This pattern — lag features + tree-based models + chronological validation —
maps directly to predictive maintenance, anomaly detection, and infrastructure reliability.

## Run it

```bash
pip install numpy pandas matplotlib xgboost scikit-learn
python sensor_forecast_demo.py
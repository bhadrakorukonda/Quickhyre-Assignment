# Microgcc Sales Forecasting System

End-to-end time series forecasting system that predicts 8 weeks of weekly
beverage sales for 43 US states, served via a production-ready REST API.

Built as part of the Microgcc Data Science Apprenticeship assignment.

---

## Results Summary

| Model | Mean MAPE | States Won |
|-------|-----------|------------|
| **LSTM** | **7.81%** | **35** |
| ARIMA/SARIMA | 16.53% | 8 |
| Facebook Prophet | 33.84% | 0 |
| XGBoost | 44.70% | 0 |

LSTM was automatically selected as the best model for 35 of 43 states.
ARIMA won the remaining 8 states (Arizona 4.20%, Iowa 4.78%, Nebraska 5.04%).

---

## Project Structure
microgcc-forecasting/
├── data/                        # Raw Excel dataset
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis
├── src/
│   ├── data_prep.py             # Load, resample, train/val split
│   ├── features.py              # Lag, rolling, calendar features
│   ├── evaluate.py              # MAE, RMSE, MAPE metrics
│   ├── model_selector.py        # Auto model selection by MAPE
│   └── models/
│       ├── arima_model.py       # ARIMA/SARIMA (pmdarima auto_arima)
│       ├── prophet_model.py     # Facebook Prophet
│       ├── xgboost_model.py     # XGBoost with lag features
│       └── lstm_model.py        # 2-layer LSTM (TensorFlow)
├── api/
│   └── main.py                  # FastAPI REST API
├── artifacts/                   # Saved models + result CSVs
├── requirements.txt
└── README.md

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone <your-repo-url>
cd microgcc-forecasting
pip install -r requirements.txt
```

Place the dataset at `data/Forecasting Case- Study.xlsx`.

---

## Reproducing Results

Run each step in order:

```bash
# 1. Verify data pipeline
python src/data_prep.py

# 2. Build features
python src/features.py

# 3. Train and evaluate all models (run independently)
python src/models/prophet_model.py    # ~1 min
python src/models/arima_model.py      # ~15 min
python src/models/xgboost_model.py    # ~30 sec
python src/models/lstm_model.py       # ~20 min

# 4. Select best model per state
python src/model_selector.py

# 5. Start the API
uvicorn api.main:app --reload
```

---

## API Reference

Base URL: `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/states` | List all 43 available states |
| GET | `/models` | Model comparison table (all 4 models) |
| GET | `/best-models` | Best model selected per state |
| GET | `/forecast/{state}` | 8-week forecast for a state |

### Query Parameters

`/forecast/{state}?periods=8`

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `periods` | int | 8 | 1–26 | Weeks to forecast ahead |

### Example Requests

```bash
# Health check
curl http://127.0.0.1:8000/

# List states
curl http://127.0.0.1:8000/states

# Forecast California (uses LSTM, MAPE 5.57%)
curl http://127.0.0.1:8000/forecast/California

# Forecast Arizona (uses ARIMA, MAPE 4.20%)
curl http://127.0.0.1:8000/forecast/Arizona

# Custom horizon — 12 weeks ahead
curl http://127.0.0.1:8000/forecast/Texas?periods=12
```

### Example Response

```json
{
  "state": "California",
  "model_used": "LSTM",
  "mape_on_val": 5.5671,
  "periods": 8,
  "forecast": [
    {"week": "2023-12-11", "forecast": 981417700.0},
    {"week": "2023-12-18", "forecast": 1033648000.0},
    {"week": "2023-12-25", "forecast": 1058560000.0},
    {"week": "2024-01-01", "forecast": 1066083000.0},
    {"week": "2024-01-08", "forecast": 1062473000.0},
    {"week": "2024-01-15", "forecast": 1051940000.0},
    {"week": "2024-01-22", "forecast": 1037512000.0},
    {"week": "2024-01-29", "forecast": 1021471000.0}
  ]
}
```

---

## Technical Details

### Dataset
- 8,084 records across 43 US states (Beverages category)
- Date range: January 2019 – December 2023
- Raw data has irregular gaps (1–91 days) — resampled to weekly frequency

### Feature Engineering
- **Lag features:** lag-1, lag-7, lag-30 weeks
- **Rolling features:** 4-week and 8-week rolling mean and std
- **Calendar features:** month, quarter, year, week-of-year, US holiday flag
- **Train/val split:** last 12 weeks per state held out — no random shuffling, no leakage

### Model Details

**LSTM** — Best overall (mean MAPE 7.81%)
- 2-layer LSTM (64 → 32 units) with Dropout
- 12-week lookback window, trained per state
- MinMax scaled per state, 100 epochs

**ARIMA/SARIMA** — Runner up (mean MAPE 16.53%)
- auto_arima with stepwise search
- Weekly seasonality (m=52)
- auto_arima selected non-seasonal orders for all 43 states

**Facebook Prophet** — (mean MAPE 33.84%)
- Multiplicative seasonality mode
- US holidays included
- Yearly seasonality enabled

**XGBoost** — (mean MAPE 44.70%)
- Global model across all 43 states (state label-encoded)
- 500 estimators, learning rate 0.05, max depth 6
- Lag + rolling + calendar features as inputs

### Model Selection
The `ModelSelector` loads validation metrics for all 4 models and selects
the best by lowest MAPE per state. Selection is persisted to
`artifacts/model_selection.csv` and loaded by the API at startup.

---

## Hardware Used
- Asus Zephyrus G14, Ryzen 9, 16GB RAM
- TensorFlow CPU (AMD GPU — no CUDA support)
- All training completed locally, no cloud compute required
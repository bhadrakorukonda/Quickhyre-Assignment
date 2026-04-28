import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from typing import Optional

from src.data_prep import load_raw, resample_weekly, train_val_split
from src.model_selector import load_all_results, get_model_summary, load_selection

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Microgcc Sales Forecasting API",
    description="""
## 🏪 US Beverage Sales Forecasting System

A **production-ready forecasting API** that predicts the next 8 weeks of beverage sales
for 43 US states using an ensemble of trained time-series models.

---

### Models Trained
| Model | Mean MAPE | States Won |
|-------|-----------|------------|
| **LSTM** | 7.81% | 35 |
| ARIMA/SARIMA | 16.53% | 8 |
| Facebook Prophet | 33.84% | 0 |
| XGBoost | 44.70% | 0 |

### How it works
1. All 4 models were trained on weekly beverage sales data (Jan 2019 – Dec 2023)
2. Each model was evaluated on a **12-week held-out validation set** (no leakage)
3. The best model per state is selected automatically by lowest **MAPE**
4. This API serves predictions from the best model for each requested state

### Quick Start
- `GET /states` — see all 43 available states
- `GET /forecast/{state}` — get 8-week forecast for any state
- `GET /models` — compare all 4 models
- `GET /best-models` — see which model won each state
""",
    version="1.0.0",
    contact={
        "name": "Microgcc Data Science Assignment",
    },
    license_info={
        "name": "MIT",
    },
)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

# ── Lazy-loaded state ──────────────────────────────────────────────────────────
_weekly_df    = None
_train_df     = None
_val_df       = None
_selection_df = None
_lstm_trained = None
_arima_models = None
_xgb_model    = None
_xgb_le       = None


def get_data():
    global _weekly_df, _train_df, _val_df
    if _weekly_df is None:
        raw        = load_raw()
        _weekly_df = resample_weekly(raw)
        _train_df, _val_df = train_val_split(_weekly_df)
    return _weekly_df, _train_df, _val_df


def get_selection():
    global _selection_df
    if _selection_df is None:
        _selection_df = load_selection()
    return _selection_df


def get_lstm():
    global _lstm_trained
    if _lstm_trained is None:
        from src.models.lstm_model import load_lstm
        selection    = get_selection()
        lstm_states  = selection[selection["model"] == "LSTM"]["state"].tolist()
        _lstm_trained = load_lstm(lstm_states)
    return _lstm_trained


def get_arima():
    global _arima_models
    if _arima_models is None:
        from src.models.arima_model import load_arima
        selection    = get_selection()
        arima_states = selection[selection["model"] == "ARIMA"]["state"].tolist()
        _arima_models = {s: load_arima(s) for s in arima_states}
    return _arima_models


def get_xgb():
    global _xgb_model, _xgb_le, _xgb_scalers
    if _xgb_model is None:
        from src.models.xgboost_model import load_xgboost
        _xgb_model, _xgb_le, _xgb_scalers = load_xgboost()
    return _xgb_model, _xgb_le, _xgb_scalers


# ── Schemas ────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    message: str = Field(..., example="Forecasting API is running.")
    models_available: list[str] = Field(..., example=["LSTM", "ARIMA", "Prophet", "XGBoost"])
    states_covered: int = Field(..., example=43)


class ForecastPoint(BaseModel):
    week: str = Field(..., example="2024-01-08", description="Monday date of the forecast week (YYYY-MM-DD)")
    forecast: float = Field(..., example=412500000.0, description="Predicted weekly beverage sales in USD")


class ForecastResponse(BaseModel):
    state:       str   = Field(..., example="California", description="US state name")
    model_used:  str   = Field(..., example="LSTM", description="Best performing model selected for this state")
    mape_on_val: float = Field(..., example=5.57, description="Model's MAPE % on 12-week validation set (lower is better)")
    periods:     int   = Field(..., example=8, description="Number of weeks forecasted")
    forecast: list[ForecastPoint] = Field(..., description="Week-by-week forecast values")

    class Config:
        json_schema_extra = {
            "example": {
                "state": "California",
                "model_used": "LSTM",
                "mape_on_val": 5.57,
                "periods": 8,
                "forecast": [
                    {"week": "2023-12-11", "forecast": 981417700.0},
                    {"week": "2023-12-18", "forecast": 1033648000.0},
                    {"week": "2023-12-25", "forecast": 1058560000.0},
                    {"week": "2024-01-01", "forecast": 1066083000.0},
                    {"week": "2024-01-08", "forecast": 1062473000.0},
                    {"week": "2024-01-15", "forecast": 1051940000.0},
                    {"week": "2024-01-22", "forecast": 1037512000.0},
                    {"week": "2024-01-29", "forecast": 1021471000.0},
                ]
            }
        }


class ModelInfo(BaseModel):
    model:      str   = Field(..., example="LSTM",    description="Model name")
    mean_mape:  float = Field(..., example=7.81,      description="Average MAPE across all 43 states (lower is better)")
    mean_mae:   float = Field(..., example=25184349.96, description="Average Mean Absolute Error in USD")
    mean_rmse:  float = Field(..., example=60316354.01, description="Average Root Mean Squared Error in USD")
    wins:       int   = Field(..., example=35,         description="Number of states where this model had the lowest MAPE")


class BestModelEntry(BaseModel):
    state: str   = Field(..., example="Alabama",  description="US state")
    model: str   = Field(..., example="LSTM",     description="Best model for this state")
    mape:  float = Field(..., example=6.37,       description="Validation MAPE % for the winning model")


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get(
    "/",
    summary="Health check",
    description="Confirms the API is running and returns basic metadata about available models and states.",
    response_model=HealthResponse,
    tags=["System"],
)
def root():
    return {
        "status": "ok",
        "message": "Forecasting API is running.",
        "models_available": ["LSTM", "ARIMA", "Prophet", "XGBoost"],
        "states_covered": 43,
    }


@app.get(
    "/states",
    summary="List all available states",
    description="Returns all 43 US states covered by the forecasting system.",
    tags=["Data"],
    responses={
        200: {
            "description": "List of state names",
            "content": {
                "application/json": {
                    "example": {
                        "count": 43,
                        "states": ["Alabama", "Arizona", "Arkansas", "California", "..."]
                    }
                }
            }
        }
    }
)
def list_states():
    selection = get_selection()
    states    = sorted(selection["state"].tolist())
    return {"count": len(states), "states": states}


@app.get(
    "/models",
    summary="Model comparison table",
    description="""
Returns performance metrics for all 4 trained models averaged across all 43 states.

**Metrics explained:**
- **mean_mape**: Mean Absolute Percentage Error — primary ranking metric (lower = better)
- **mean_mae**: Mean Absolute Error in USD
- **mean_rmse**: Root Mean Squared Error in USD
- **wins**: Number of states where this model achieved the lowest MAPE
""",
    response_model=list[ModelInfo],
    tags=["Models"],
)
def model_comparison():
    results = load_all_results()
    summary = get_model_summary(results)
    return summary.to_dict(orient="records")


@app.get(
    "/best-models",
    summary="Best model selected per state",
    description="""
Returns the automatically selected best model for each of the 43 states,
based on lowest MAPE on the 12-week held-out validation set.

**Model selection logic:** For each state, all 4 models were evaluated on the
same validation period. The model with the lowest MAPE was selected.
Results: LSTM won 35 states, ARIMA won 8 states.
""",
    response_model=list[BestModelEntry],
    tags=["Models"],
)
def best_models():
    selection = get_selection()
    return selection[["state", "model", "mape"]].to_dict(orient="records")


@app.get(
    "/forecast/{state}",
    summary="Get 8-week sales forecast for a state",
    description="""
Generates a week-by-week sales forecast for the specified US state using the
best performing model for that state.

**Examples:**
- `/forecast/California` — uses LSTM (MAPE 5.57%)
- `/forecast/Arizona` — uses ARIMA (MAPE 4.20%)

**Notes:**
- State names are case-insensitive (`california` = `California`)
- Use `/states` to see all valid state names
- Use `periods` query param to change forecast horizon (default: 8)
- Forecast values are in USD (weekly beverage sales)
""",
    response_model=ForecastResponse,
    tags=["Forecast"],
    responses={
        200: {"description": "Successful forecast"},
        404: {
            "description": "State not found",
            "content": {
                "application/json": {
                    "example": {"detail": "State 'XYZ' not found. Call /states to see valid options."}
                }
            }
        }
    }
)
def forecast(
    state: str,
    periods: int = Query(
        default=8,
        ge=1,
        le=26,
        description="Number of weeks to forecast ahead (1–26). Default is 8.",
        example=8,
    ),
):
    selection = get_selection()
    matched   = selection[selection["state"].str.lower() == state.lower()]
    if matched.empty:
        raise HTTPException(
            status_code=404,
            detail=f"State '{state}' not found. Call /states to see valid options."
        )

    row        = matched.iloc[0]
    state_name = row["state"]
    model_name = row["model"]
    mape_val   = float(row["mape"])

    weekly, train, val = get_data()

    if model_name == "LSTM":
        from src.models.lstm_model import predict_lstm
        trained     = get_lstm()
        forecast_df = predict_lstm(trained, state_name, periods=periods)

    elif model_name == "ARIMA":
        from src.models.arima_model import predict_arima
        arima_models = get_arima()
        model        = arima_models[state_name]
        forecast_df  = predict_arima(model, periods=periods)
        last_date    = weekly[weekly["State"] == state_name]["ds"].max()
        dates        = [last_date + pd.Timedelta(weeks=i + 1) for i in range(periods)]
        forecast_df["ds"] = dates

    elif model_name == "XGBoost":
        from src.models.xgboost_model import predict_xgboost
        from src.features import build_features
        xgb_model, le = get_xgb()
        df_feat        = build_features(weekly)
        forecast_df    = predict_xgboost(xgb_model, df_feat, state_name, le, periods=periods)

    else:
        from src.models.prophet_model import load_prophet, predict_prophet
        model       = load_prophet(state_name)
        forecast_df = predict_prophet(model, periods=periods)

    points = [
        ForecastPoint(
            week=str(r["ds"])[:10],
            forecast=round(float(r["forecast"]), 2),
        )
        for _, r in forecast_df.iterrows()
    ]

    return ForecastResponse(
        state=state_name,
        model_used=model_name,
        mape_on_val=round(mape_val, 4),
        periods=periods,
        forecast=points,
    )
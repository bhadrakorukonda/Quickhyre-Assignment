import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def train_prophet(train_df: pd.DataFrame, state: str) -> Prophet:
    """
    Fit a Prophet model for a single state.
    train_df must have columns: ds (datetime), y (float)
    """
    df = train_df[train_df["State"] == state][["ds", "y"]].copy()
    df = df.dropna().sort_values("ds").reset_index(drop=True)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,   # data is weekly — not meaningful
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # better for sales data with trend
        changepoint_prior_scale=0.05,       # controls trend flexibility
    )
    model.add_country_holidays(country_name="US")
    model.fit(df)
    return model


def predict_prophet(model: Prophet, periods: int = 8) -> pd.DataFrame:
    """Forecast next `periods` weeks beyond the training data."""
    future = model.make_future_dataframe(periods=periods, freq="W-MON")
    forecast = model.predict(future)
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
    result = result.rename(columns={"yhat": "forecast"})
    return result.reset_index(drop=True)


def evaluate_prophet(model: Prophet, val_df: pd.DataFrame, state: str) -> dict:
    """Run predictions on validation dates and return metrics dict."""
    from src.evaluate import evaluate

    val = val_df[val_df["State"] == state][["ds", "y"]].copy()
    val = val.dropna().sort_values("ds").reset_index(drop=True)

    forecast = model.predict(val[["ds"]])
    y_pred = forecast["yhat"].values
    y_true = val["y"].values

    return evaluate(y_true, y_pred, model_name="Prophet", state=state)


def save_prophet(model: Prophet, state: str):
    path = ARTIFACTS_DIR / f"prophet_{state.replace(' ', '_')}.joblib"
    joblib.dump(model, path)


def load_prophet(state: str) -> Prophet:
    path = ARTIFACTS_DIR / f"prophet_{state.replace(' ', '_')}.joblib"
    return joblib.load(path)


def run_all_states(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Train + evaluate Prophet for all 43 states. Returns results DataFrame."""
    from src.evaluate import summarise_results

    states = sorted(train_df["State"].unique())
    results = []

    for i, state in enumerate(states, 1):
        print(f"  [{i:02d}/43] {state}...", end=" ")
        model = train_prophet(train_df, state)
        metrics = evaluate_prophet(model, val_df, state)
        save_prophet(model, state)
        results.append(metrics)
        print(f"MAPE={metrics['mape']:.2f}%")

    return summarise_results(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data_prep import load_raw, resample_weekly, train_val_split

    print("Loading data...")
    raw    = load_raw()
    weekly = resample_weekly(raw)
    train, val = train_val_split(weekly)

    print("Training Prophet for all 43 states...\n")
    results = run_all_states(train, val)

    print("\n--- Prophet Results ---")
    print(results.to_string(index=False))
    print(f"\nMean MAPE: {results['mape'].mean():.2f}%")
    print(f"Best state:  {results.loc[results['mape'].idxmin(), 'state']} ({results['mape'].min():.2f}%)")
    print(f"Worst state: {results.loc[results['mape'].idxmax(), 'state']} ({results['mape'].max():.2f}%)")

    results.to_csv(ARTIFACTS_DIR / "prophet_results.csv", index=False)
    print("\nResults saved to artifacts/prophet_results.csv")
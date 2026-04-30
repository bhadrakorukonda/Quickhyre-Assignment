import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from pmdarima import auto_arima

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def train_arima(train_df: pd.DataFrame, state: str):
    """
    Fit SARIMA for a single state using auto_arima.
    Returns fitted model.
    """
    df = train_df[train_df["State"] == state][["ds", "y"]].copy()
    df = df.dropna().sort_values("ds").reset_index(drop=True)
    y = df["y"].values

    model = auto_arima(
        y,
        seasonal=True,
        m=52,                    # weekly seasonality — 52 weeks per year
        stepwise=True,           # faster search
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3,
        max_P=1, max_Q=1,        # keep seasonal terms simple
        d=None,                  # auto-detect differencing
        information_criterion="aic",
        n_jobs=1,
    )
    return model


def predict_arima(model, periods: int = 8) -> pd.DataFrame:
    """Forecast next `periods` weeks. Returns DataFrame with forecast column."""
    preds, conf_int = model.predict(n_periods=periods, return_conf_int=True)
    return pd.DataFrame({
        "forecast":    preds,
        "lower_bound": conf_int[:, 0],
        "upper_bound": conf_int[:, 1],
    })


def evaluate_arima(model, val_df: pd.DataFrame, state: str) -> dict:
    """Predict on validation set and return metrics dict."""
    from src.evaluate import evaluate

    val = val_df[val_df["State"] == state][["ds", "y"]].copy()
    val = val.dropna().sort_values("ds").reset_index(drop=True)
    n = len(val)

    preds = model.predict(n_periods=n)
    return evaluate(val["y"].values, preds, model_name="ARIMA", state=state)


def save_arima(model, state: str):
    path = ARTIFACTS_DIR / f"arima_{state.replace(' ', '_')}.joblib"
    joblib.dump(model, path)


def load_arima(state: str):
    path = ARTIFACTS_DIR / f"arima_{state.replace(' ', '_')}.joblib"
    return joblib.load(path)


def run_all_states(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Train + evaluate ARIMA for all 43 states. Returns results DataFrame."""
    from src.evaluate import summarise_results

    states = sorted(train_df["State"].unique())
    results = []

    for i, state in enumerate(states, 1):
        print(f"  [{i:02d}/43] {state}...", end=" ", flush=True)
        try:
            model = train_arima(train_df, state)
            metrics = evaluate_arima(model, val_df, state)
            save_arima(model, state)
            results.append(metrics)
            print(f"MAPE={metrics['mape']:.2f}%  order={model.order}  seasonal={model.seasonal_order}")
        except Exception as e:
            print(f"FAILED — {e}")
            results.append({
                "model": "ARIMA", "state": state,
                "mae": np.nan, "rmse": np.nan, "mape": np.nan
            })

    return summarise_results(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data_prep import load_raw, resample_weekly, train_val_split

    print("Loading data...")
    raw    = load_raw()
    weekly = resample_weekly(raw)
    train, val = train_val_split(weekly)

    print("Training ARIMA/SARIMA for all 43 states...\n")
    results = run_all_states(train, val)

    print("\n--- ARIMA Results ---")
    print(results.to_string(index=False))
    print(f"\nMean MAPE: {results['mape'].mean():.2f}%")
    print(f"Best state:  {results.loc[results['mape'].idxmin(), 'state']} ({results['mape'].min():.2f}%)")
    print(f"Worst state: {results.loc[results['mape'].idxmax(), 'state']} ({results['mape'].max():.2f}%)")

    results.to_csv(ARTIFACTS_DIR / "arima_results.csv", index=False)
    print("\nResults saved to artifacts/arima_results.csv")
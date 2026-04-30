import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def train_holtwinters(train_df: pd.DataFrame, state: str):
    df = train_df[train_df["State"] == state][["ds", "y"]].copy()
    df = df.dropna().sort_values("ds").reset_index(drop=True)
    y  = df["y"].values

    best_model  = None
    best_aic    = np.inf

    # Try combinations — pick best by AIC
    configs = [
        dict(trend="add", damped_trend=False, seasonal="add",    seasonal_periods=52),
        dict(trend="add", damped_trend=True,  seasonal="add",    seasonal_periods=52),
        dict(trend="add", damped_trend=False, seasonal="mul",    seasonal_periods=52),
        dict(trend="add", damped_trend=True,  seasonal="mul",    seasonal_periods=52),
        dict(trend="add", damped_trend=False, seasonal=None),
        dict(trend="add", damped_trend=True,  seasonal=None),
    ]

    for cfg in configs:
        try:
            m = ExponentialSmoothing(y, **cfg).fit(optimized=True, remove_bias=True)
            if m.aic < best_aic:
                best_aic   = m.aic
                best_model = m
        except Exception:
            continue

    return best_model


def predict_holtwinters(model, periods: int = 8) -> pd.DataFrame:
    preds = model.forecast(periods)
    return pd.DataFrame({"forecast": preds}).reset_index(drop=True)


def evaluate_holtwinters(model, val_df: pd.DataFrame, state: str) -> dict:
    from src.evaluate import evaluate

    val = val_df[val_df["State"] == state][["ds", "y"]].dropna().sort_values("ds")
    n   = len(val)

    preds  = model.forecast(n)
    return evaluate(val["y"].values, preds,
                    model_name="HoltWinters", state=state)


def save_holtwinters(model, state: str):
    path = ARTIFACTS_DIR / f"holtwinters_{state.replace(' ', '_')}.joblib"
    joblib.dump(model, path)


def load_holtwinters(state: str):
    path = ARTIFACTS_DIR / f"holtwinters_{state.replace(' ', '_')}.joblib"
    return joblib.load(path)


def run_all_states(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    from src.evaluate import summarise_results

    states  = sorted(train_df["State"].unique())
    results = []

    for i, state in enumerate(states, 1):
        print(f"  [{i:02d}/43] {state}...", end=" ", flush=True)
        try:
            model   = train_holtwinters(train_df, state)
            metrics = evaluate_holtwinters(model, val_df, state)
            save_holtwinters(model, state)
            results.append(metrics)
            print(f"MAPE={metrics['mape']:.2f}%")
        except Exception as e:
            print(f"FAILED — {e}")
            results.append({
                "model": "HoltWinters", "state": state,
                "mae": np.nan, "rmse": np.nan, "mape": np.nan,
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

    print("Training Holt-Winters for all 43 states...\n")
    results = run_all_states(train, val)

    print("\n--- Holt-Winters Results ---")
    print(results.to_string(index=False))
    print(f"\nMean MAPE: {results['mape'].mean():.2f}%")
    print(f"Best state:  {results.loc[results['mape'].idxmin(), 'state']} ({results['mape'].min():.2f}%)")
    print(f"Worst state: {results.loc[results['mape'].idxmax(), 'state']} ({results['mape'].max():.2f}%)")

    results.to_csv(ARTIFACTS_DIR / "holtwinters_results.csv", index=False)
    print("\nResults saved to artifacts/holtwinters_results.csv")
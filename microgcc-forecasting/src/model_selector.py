import pandas as pd
import joblib
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def load_all_results() -> pd.DataFrame:
    """Load result CSVs for all 4 models and combine into one DataFrame."""
    files = {
        "prophet":  ARTIFACTS_DIR / "prophet_results.csv",
        "arima":    ARTIFACTS_DIR / "arima_results.csv",
        "xgboost":  ARTIFACTS_DIR / "xgboost_results.csv",
        "lstm":     ARTIFACTS_DIR / "lstm_results.csv",
    }
    frames = []
    for name, path in files.items():
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
        else:
            print(f"Warning: {path} not found — skipping {name}")

    return pd.concat(frames, ignore_index=True)


def get_best_model_per_state(results: pd.DataFrame) -> pd.DataFrame:
    """Return the best model per state ranked by lowest MAPE."""
    idx = results.groupby("state")["mape"].idxmin()
    best = results.loc[idx].reset_index(drop=True)
    best = best.sort_values("state").reset_index(drop=True)
    return best


def get_model_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Return average metrics per model across all states."""
    summary = results.groupby("model").agg(
        mean_mae=("mae",  "mean"),
        mean_rmse=("rmse", "mean"),
        mean_mape=("mape", "mean"),
        wins=("mape",  lambda x: 0),   # placeholder — filled below
    ).reset_index()

    best_per_state = get_best_model_per_state(results)
    win_counts = best_per_state["model"].value_counts().reset_index()
    win_counts.columns = ["model", "wins"]

    summary = summary.drop(columns=["wins"]).merge(win_counts, on="model", how="left")
    summary["wins"] = summary["wins"].fillna(0).astype(int)
    summary = summary.sort_values("mean_mape").reset_index(drop=True)

    summary["mean_mae"]  = summary["mean_mae"].round(2)
    summary["mean_rmse"] = summary["mean_rmse"].round(2)
    summary["mean_mape"] = summary["mean_mape"].round(4)

    return summary


def select_best_model_for_state(state: str, results: pd.DataFrame) -> str:
    """Return the name of the best model for a given state."""
    state_results = results[results["state"] == state]
    if state_results.empty:
        raise ValueError(f"No results found for state: {state}")
    best_row = state_results.loc[state_results["mape"].idxmin()]
    return best_row["model"]


def save_selection(best_per_state: pd.DataFrame):
    """Persist the model selection table for use by the API."""
    path = ARTIFACTS_DIR / "model_selection.csv"
    best_per_state.to_csv(path, index=False)
    print(f"Model selection saved to {path}")


def load_selection() -> pd.DataFrame:
    """Load the persisted model selection table."""
    path = ARTIFACTS_DIR / "model_selection.csv"
    return pd.read_csv(path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    print("Loading all model results...")
    results = load_all_results()
    print(f"Total records: {len(results)} ({results['model'].nunique()} models × {results['state'].nunique()} states)\n")

    print("=== Model Summary (ranked by mean MAPE) ===")
    summary = get_model_summary(results)
    print(summary.to_string(index=False))

    print("\n=== Best Model Per State ===")
    best = get_best_model_per_state(results)
    print(best[["state", "model", "mape"]].to_string(index=False))

    print(f"\n=== Win Count ===")
    print(best["model"].value_counts().to_string())

    save_selection(best)
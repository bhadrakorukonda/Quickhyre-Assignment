import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error. epsilon avoids division by zero."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


def evaluate(y_true, y_pred, model_name: str = "", state: str = "") -> dict:
    """Return a metrics dict for one model/state combination."""
    return {
        "model":  model_name,
        "state":  state,
        "mae":    round(mae(y_true, y_pred), 2),
        "rmse":   round(rmse(y_true, y_pred), 2),
        "mape":   round(mape(y_true, y_pred), 4),
    }


def summarise_results(results: list[dict]) -> pd.DataFrame:
    """
    Convert a list of evaluate() dicts into a sorted DataFrame.
    Lower MAPE = better.
    """
    df = pd.DataFrame(results)
    df = df.sort_values(["state", "mape"]).reset_index(drop=True)
    return df


def best_model_per_state(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return the single best model per state by lowest MAPE."""
    idx = results_df.groupby("state")["mape"].idxmin()
    return results_df.loc[idx].reset_index(drop=True)


if __name__ == "__main__":
    # Quick sanity check
    y_true = [100, 200, 300, 400]
    y_pred = [110, 190, 310, 390]

    result = evaluate(y_true, y_pred, model_name="test", state="Alabama")
    print("Metrics:", result)

    dummy_results = [
        {"model": "ARIMA",   "state": "Alabama", "mae": 5000, "rmse": 6000, "mape": 3.2},
        {"model": "Prophet", "state": "Alabama", "mae": 4000, "rmse": 5500, "mape": 2.8},
        {"model": "ARIMA",   "state": "Texas",   "mae": 8000, "rmse": 9000, "mape": 4.1},
        {"model": "Prophet", "state": "Texas",   "mae": 7500, "rmse": 8500, "mape": 3.9},
    ]
    df = summarise_results(dummy_results)
    print("\nAll results:")
    print(df)
    print("\nBest per state:")
    print(best_model_per_state(df))
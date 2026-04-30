import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "month", "quarter", "year", "week_of_year", "is_holiday_week",
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_4w", "rolling_mean_8w",
    "rolling_std_4w", "rolling_std_8w",
    "state_encoded",
]


def encode_states(train_df: pd.DataFrame, val_df: pd.DataFrame):
    le = LabelEncoder()
    train_df = train_df.copy()
    val_df   = val_df.copy()
    train_df["state_encoded"] = le.fit_transform(train_df["State"])
    val_df["state_encoded"]   = le.transform(val_df["State"])
    return train_df, val_df, le


def normalize_per_state(df: pd.DataFrame) -> tuple:
    """
    Normalize y and all lag/rolling features per state using train statistics.
    Returns normalized df and scaler dict {state: (mean, std)}.
    """
    df = df.copy()
    scalers = {}
    cols_to_scale = ["y", "lag_1", "lag_7", "lag_30",
                     "rolling_mean_4w", "rolling_mean_8w",
                     "rolling_std_4w",  "rolling_std_8w"]

    for state in df["State"].unique():
        mask = df["State"] == state
        state_data = df.loc[mask, "y"].dropna()
        mu  = state_data.mean()
        std = state_data.std() + 1e-8
        scalers[state] = (mu, std)
        for col in cols_to_scale:
            if col in df.columns:
                df.loc[mask, col] = (df.loc[mask, col] - mu) / std

    return df, scalers


def apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Apply pre-fitted scalers to a new dataframe (val/test)."""
    df = df.copy()
    cols_to_scale = ["y", "lag_1", "lag_7", "lag_30",
                     "rolling_mean_4w", "rolling_mean_8w",
                     "rolling_std_4w",  "rolling_std_8w"]
    for state in df["State"].unique():
        if state not in scalers:
            continue
        mu, std = scalers[state]
        mask = df["State"] == state
        for col in cols_to_scale:
            if col in df.columns:
                df.loc[mask, col] = (df.loc[mask, col] - mu) / std
    return df


def train_xgboost(train_df: pd.DataFrame) -> XGBRegressor:
    df = train_df.dropna(subset=FEATURE_COLS + ["y"]).copy()
    X  = df[FEATURE_COLS]
    y  = df["y"]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def evaluate_xgboost(
    model: XGBRegressor,
    val_df: pd.DataFrame,
    scalers: dict,
) -> pd.DataFrame:
    from src.evaluate import evaluate, summarise_results

    val_df = val_df.dropna(subset=FEATURE_COLS + ["y"]).copy()
    results = []

    for state in sorted(val_df["State"].unique()):
        mu, std = scalers[state]
        state_val = val_df[val_df["State"] == state]
        X_val  = state_val[FEATURE_COLS]
        y_true = state_val["y"].values * std + mu   # inverse scale
        y_pred_scaled = model.predict(X_val)
        y_pred = y_pred_scaled * std + mu            # inverse scale
        results.append(evaluate(y_true, y_pred, model_name="XGBoost", state=state))

    return summarise_results(results)


def predict_xgboost(
    model: XGBRegressor,
    last_known: pd.DataFrame,
    state: str,
    le: LabelEncoder,
    scalers: dict,
    periods: int = 8,
) -> pd.DataFrame:
    import holidays as hd

    df = last_known[last_known["State"] == state].sort_values("ds").copy()
    df["state_encoded"] = le.transform([state])[0]
    mu, std = scalers[state]
    us_holidays = hd.US(years=range(2019, 2026))

    # Work in normalized space
    def norm(val):  return (val - mu) / std
    def denorm(val): return val * std + mu

    history = df.copy()
    # normalize history y
    history["y"] = history["y"].apply(norm)

    forecasts = []
    for step in range(periods):
        last   = history.iloc[-1]
        next_ds = last["ds"] + pd.Timedelta(weeks=1)

        row = {
            "ds":             next_ds,
            "month":          next_ds.month,
            "quarter":        (next_ds.month - 1) // 3 + 1,
            "year":           next_ds.year,
            "week_of_year":   next_ds.isocalendar()[1],
            "is_holiday_week": int(any(
                (next_ds + pd.Timedelta(days=i)) in us_holidays for i in range(7)
            )),
            "lag_1":  history["y"].iloc[-1],
            "lag_7":  history["y"].iloc[-7]  if len(history) >= 7  else history["y"].iloc[0],
            "lag_30": history["y"].iloc[-30] if len(history) >= 30 else history["y"].iloc[0],
            "rolling_mean_4w": history["y"].iloc[-4:].mean(),
            "rolling_mean_8w": history["y"].iloc[-8:].mean(),
            "rolling_std_4w":  history["y"].iloc[-4:].std(ddof=0),
            "rolling_std_8w":  history["y"].iloc[-8:].std(ddof=0),
            "state_encoded":   le.transform([state])[0],
            "State": state,
            "y": None,
        }

        X_row      = pd.DataFrame([row])[FEATURE_COLS]
        pred_norm  = float(model.predict(X_row)[0])
        pred_actual = denorm(pred_norm)
        row["y"]   = pred_norm
        forecasts.append({"ds": next_ds, "forecast": pred_actual})
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    return pd.DataFrame(forecasts)


def save_xgboost(model: XGBRegressor, le: LabelEncoder, scalers: dict):
    joblib.dump(model,   ARTIFACTS_DIR / "xgboost_global.joblib")
    joblib.dump(le,      ARTIFACTS_DIR / "xgboost_label_encoder.joblib")
    joblib.dump(scalers, ARTIFACTS_DIR / "xgboost_scalers.joblib")


def load_xgboost():
    model   = joblib.load(ARTIFACTS_DIR / "xgboost_global.joblib")
    le      = joblib.load(ARTIFACTS_DIR / "xgboost_label_encoder.joblib")
    scalers = joblib.load(ARTIFACTS_DIR / "xgboost_scalers.joblib")
    return model, le, scalers


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data_prep import load_raw, resample_weekly, train_val_split
    from src.features  import build_features

    print("Loading and preparing data...")
    raw    = load_raw()
    weekly = resample_weekly(raw)
    df     = build_features(weekly)
    train, val = train_val_split(df)

    print("Encoding states...")
    train_enc, val_enc, le = encode_states(train, val)

    print("Normalizing per state...")
    train_norm, scalers = normalize_per_state(train_enc)
    val_norm            = apply_scalers(val_enc, scalers)

    print("Training XGBoost (1000 estimators)...")
    model = train_xgboost(train_norm)
    print("Done.\n")

    print("Evaluating on validation set...")
    results = evaluate_xgboost(model, val_norm, scalers)
    print("\n--- Improved XGBoost Results ---")
    print(results.to_string(index=False))
    print(f"\nMean MAPE: {results['mape'].mean():.2f}%")
    print(f"Best state:  {results.loc[results['mape'].idxmin(), 'state']} ({results['mape'].min():.2f}%)")
    print(f"Worst state: {results.loc[results['mape'].idxmax(), 'state']} ({results['mape'].max():.2f}%)")

    save_xgboost(model, le, scalers)
    results.to_csv(ARTIFACTS_DIR / "xgboost_results.csv", index=False)
    print("\nSaved to artifacts/")

    print("\nSample 8-week forecast for California:")
    df_feat = build_features(weekly)
    forecast = predict_xgboost(model, df_feat, "California", le, scalers, periods=8)
    print(forecast.to_string(index=False))
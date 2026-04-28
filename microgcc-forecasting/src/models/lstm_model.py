import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

LOOKBACK   = 12    # was 12 — captures 6 months of history
FORECAST_H = 8
EPOCHS     = 100   # was 100 — early stopping will kick in before this
BATCH_SIZE = 32    # was 32 — smaller batches = better gradient updates on small series


def build_sequences(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def scale_series(series: np.ndarray):
    s_min, s_max = series.min(), series.max()
    scaled = (series - s_min) / (s_max - s_min + 1e-8)
    return scaled, s_min, s_max


def inverse_scale(scaled, s_min, s_max):
    return scaled * (s_max - s_min + 1e-8) + s_min


def build_lstm_model(lookback: int):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm_per_state(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
) -> dict:
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau
    )
    tf.get_logger().setLevel("ERROR")

    states  = sorted(train_df["State"].unique())
    trained = {}

    for i, state in enumerate(states, 1):
        print(f"  [{i:02d}/43] {state}...", end=" ", flush=True)

        full = pd.concat([
            train_df[train_df["State"] == state][["ds", "y"]],
            val_df[val_df["State"] == state][["ds", "y"]],
        ]).sort_values("ds").dropna()

        series = full["y"].values.astype(float)
        scaled, s_min, s_max = scale_series(series)

        n_train      = len(train_df[train_df["State"] == state].dropna(subset=["y"]))
        train_scaled = scaled[:n_train]

        X_train, y_train = build_sequences(train_scaled, LOOKBACK)
        X_train = X_train.reshape(-1, LOOKBACK, 1)

        if len(X_train) == 0:
            print("SKIPPED (not enough data)")
            continue

        # Validation sequences for early stopping
        val_scaled = scaled[n_train - LOOKBACK:]
        X_val, y_val = build_sequences(val_scaled, LOOKBACK)
        X_val = X_val.reshape(-1, LOOKBACK, 1)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        model = build_lstm_model(LOOKBACK)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
        )

        best_epoch = len(history.history['loss'])
        best_val   = min(history.history['val_loss'])
        trained[state] = (model, s_min, s_max, scaled, n_train)
        print(f"done (epoch {best_epoch}, val_loss={best_val:.6f})")

    return trained


def evaluate_lstm(trained: dict, val_df: pd.DataFrame) -> pd.DataFrame:
    from src.evaluate import evaluate, summarise_results

    results = []
    for state, (model, s_min, s_max, full_scaled, n_train) in trained.items():
        val = val_df[val_df["State"] == state][["ds", "y"]].dropna().sort_values("ds")
        if len(val) == 0:
            continue

        seed = full_scaled[n_train - LOOKBACK: n_train]
        preds_scaled = []
        current = seed.copy()

        for _ in range(len(val)):
            X = current[-LOOKBACK:].reshape(1, LOOKBACK, 1)
            p = float(model.predict(X, verbose=0)[0][0])
            preds_scaled.append(p)
            current = np.append(current, p)

        y_pred = inverse_scale(np.array(preds_scaled), s_min, s_max)
        y_true = val["y"].values
        results.append(evaluate(y_true, y_pred, model_name="LSTM", state=state))

    return summarise_results(results)


def predict_lstm(
    trained: dict,
    state:   str,
    periods: int = 8,
) -> pd.DataFrame:
    model, s_min, s_max, full_scaled, n_train = trained[state]

    seed = full_scaled[-LOOKBACK:]
    preds_scaled = []
    current = seed.copy()

    for _ in range(periods):
        X = current[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        p = float(model.predict(X, verbose=0)[0][0])
        preds_scaled.append(p)
        current = np.append(current, p)

    forecasts = inverse_scale(np.array(preds_scaled), s_min, s_max)
    last_date  = pd.Timestamp("2023-12-04")
    dates      = [last_date + pd.Timedelta(weeks=i + 1) for i in range(periods)]
    return pd.DataFrame({"ds": dates, "forecast": forecasts})


def save_lstm(trained: dict):
    scalers = {}
    for state, (model, s_min, s_max, full_scaled, n_train) in trained.items():
        safe = state.replace(" ", "_")
        model.save(ARTIFACTS_DIR / f"lstm_{safe}.keras")
        scalers[state] = {
            "s_min": s_min, "s_max": s_max,
            "full_scaled": full_scaled, "n_train": n_train,
        }
    joblib.dump(scalers, ARTIFACTS_DIR / "lstm_scalers.joblib")
    print("LSTM models saved.")


def load_lstm(states: list) -> dict:
    import tensorflow as tf
    scalers = joblib.load(ARTIFACTS_DIR / "lstm_scalers.joblib")
    trained = {}
    for state in states:
        safe  = state.replace(" ", "_")
        model = tf.keras.models.load_model(ARTIFACTS_DIR / f"lstm_{safe}.keras")
        s     = scalers[state]
        trained[state] = (model, s["s_min"], s["s_max"],
                          s["full_scaled"], s["n_train"])
    return trained


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data_prep import load_raw, resample_weekly, train_val_split

    print("Loading data...")
    raw    = load_raw()
    weekly = resample_weekly(raw)
    train, val = train_val_split(weekly)

    print(f"Training improved LSTM (lookback={LOOKBACK}, epochs up to {EPOCHS} with early stopping)...\n")
    trained = train_lstm_per_state(train, val)

    print("\nEvaluating on validation set...")
    results = evaluate_lstm(trained, val)

    print("\n--- Improved LSTM Results ---")
    print(results.to_string(index=False))
    print(f"\nMean MAPE: {results['mape'].mean():.2f}%")
    print(f"Best state:  {results.loc[results['mape'].idxmin(), 'state']} ({results['mape'].min():.2f}%)")
    print(f"Worst state: {results.loc[results['mape'].idxmax(), 'state']} ({results['mape'].max():.2f}%)")

    print("\nSaving models...")
    save_lstm(trained)

    results.to_csv(ARTIFACTS_DIR / "lstm_results.csv", index=False)
    print("Results saved to artifacts/lstm_results.csv")
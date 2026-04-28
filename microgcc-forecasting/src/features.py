import pandas as pd
import numpy as np
import holidays


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, quarter, year, week-of-year and US holiday flag."""
    df = df.copy()
    df["month"]    = df["ds"].dt.month
    df["quarter"]  = df["ds"].dt.quarter
    df["year"]     = df["ds"].dt.year
    df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)

    us_holidays = holidays.US(years=range(2019, 2025))
    df["is_holiday_week"] = df["ds"].apply(
        lambda d: int(any(
            (d + pd.Timedelta(days=i)) in us_holidays for i in range(7)
        ))
    )
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 7, 30]) -> pd.DataFrame:
    """
    Add lag features per state.
    Lags are in weekly units: lag-1 = 1 week ago, lag-7 = 7 weeks ago, etc.
    """
    df = df.copy().sort_values(["State", "ds"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("State")["y"].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list = [4, 8]
) -> pd.DataFrame:
    """
    Add rolling mean and std per state.
    Windows are in weekly units. Uses shift(1) to avoid leakage.
    """
    df = df.copy().sort_values(["State", "ds"])
    for w in windows:
        rolled = df.groupby("State")["y"].shift(1).groupby(
            df["State"]
        ).transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"rolling_mean_{w}w"] = rolled

        rolled_std = df.groupby("State")["y"].shift(1).groupby(
            df["State"]
        ).transform(lambda x: x.rolling(w, min_periods=1).std())
        df[f"rolling_std_{w}w"] = rolled_std.fillna(0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run full feature engineering pipeline."""
    df = add_calendar_features(df)
    df = add_lag_features(df, lags=[1, 7, 30])
    df = add_rolling_features(df, windows=[4, 8])
    return df


FEATURE_COLS = [
    "month", "quarter", "year", "week_of_year", "is_holiday_week",
    "lag_1", "lag_7", "lag_30",
    "rolling_mean_4w", "rolling_mean_8w",
    "rolling_std_4w", "rolling_std_8w",
]


if __name__ == "__main__":
    from data_prep import load_raw, resample_weekly, train_val_split

    raw    = load_raw()
    weekly = resample_weekly(raw)
    df     = build_features(weekly)

    train, val = train_val_split(df)

    print("Feature columns:", FEATURE_COLS)
    print(f"\nTrain shape: {train.shape}")
    print(f"Val shape:   {val.shape}")
    print(f"\nMissing values in train:\n{train[FEATURE_COLS].isnull().sum()}")
    print(f"\nSample row (Alabama, last 3):")
    print(train[train["State"] == "Alabama"][["ds", "y"] + FEATURE_COLS].tail(3).to_string())
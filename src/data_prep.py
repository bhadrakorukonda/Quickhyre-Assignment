import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Forecasting Case- Study.xlsx"

def load_raw() -> pd.DataFrame:
    """Load raw Excel file and parse dates."""
    df = pd.read_excel(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "Total", "State"])
    df = df.sort_values(["State", "Date"]).reset_index(drop=True)
    return df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample each state to a clean weekly frequency (Monday anchor).
    Irregular gaps in the raw data are handled by summing within each week.
    Missing weeks after resample are forward-filled then back-filled.
    """
    frames = []
    for state, group in df.groupby("State"):
        group = group.set_index("Date")[["Total"]]
        weekly = group.resample("W-MON").sum()
        weekly["Total"] = weekly["Total"].replace(0, np.nan)
        weekly["Total"] = weekly["Total"].ffill()   # no limit — fill all gaps
        weekly["Total"] = weekly["Total"].bfill()   # catch any leading NaNs
        weekly["State"] = state
        frames.append(weekly.reset_index())
    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={"Date": "ds", "Total": "y"})
    return result


def train_val_split(df: pd.DataFrame, val_weeks: int = 12):
    """
    Chronological split — last val_weeks per state go to validation.
    No random shuffling. No data leakage.
    """
    train_frames, val_frames = [], []
    for state, group in df.groupby("State"):
        group = group.sort_values("ds").reset_index(drop=True)
        train_frames.append(group.iloc[:-val_weeks])
        val_frames.append(group.iloc[-val_weeks:])
    train = pd.concat(train_frames, ignore_index=True)
    val = pd.concat(val_frames, ignore_index=True)
    return train, val


def get_state_series(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Return the weekly series for a single state, sorted by date."""
    return df[df["State"] == state].sort_values("ds").reset_index(drop=True)


def get_all_states(df: pd.DataFrame) -> list:
    return sorted(df["State"].unique().tolist())


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")

    weekly = resample_weekly(raw)
    print(f"Weekly shape: {weekly.shape}")
    print(f"States: {weekly['State'].nunique()}")
    print(f"Date range: {weekly['ds'].min()} → {weekly['ds'].max()}")
    print(f"Weeks per state: {weekly.groupby('State')['ds'].count().describe()}")

    train, val = train_val_split(weekly)
    print(f"\nTrain shape: {train.shape}")
    print(f"Val shape:   {val.shape}")
    print(f"\nSample (Alabama):")
    print(get_state_series(weekly, "Alabama").tail(5))
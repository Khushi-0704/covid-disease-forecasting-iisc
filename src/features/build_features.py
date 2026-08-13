"""
build_features.py

Builds the modeling feature set from the master state-week table:
- Lagged case counts (t-1, t-2, t-3 weeks)
- Rolling case growth rate
- Lagged mobility features (t-1, t-2 weeks)
- Spatial-lag feature: average case rate among neighboring states (lag 1)
- Moran's I: spatial autocorrelation diagnostic (printed, not a model
  feature) confirming whether neighboring states cluster together
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from state_adjacency import STATE_ADJACENCY

PROCESSED_DIR = Path("data/processed")
MASTER_PATH = PROCESSED_DIR / "master_state_week.csv"
OUT_PATH = PROCESSED_DIR / "features_state_week.csv"

MOBILITY_COLS = [
    "retail_and_recreation_pct",
    "grocery_and_pharmacy_pct",
    "parks_pct",
    "transit_stations_pct",
    "workplaces_pct",
    "residential_pct",
]

LAG_WEEKS = [1, 2, 3]
MOBILITY_LAG_WEEKS = [1, 2]


def load_master(path: Path = MASTER_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week"])
    return df.sort_values(["State", "week"]).reset_index(drop=True)


def add_case_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in LAG_WEEKS:
        df[f"cases_lag{lag}"] = df.groupby("State")["weekly_cases"].shift(lag)
        df[f"case_rate_lag{lag}"] = df.groupby("State")["case_rate_per_100k"].shift(lag)
    return df


def add_growth_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["growth_rate_1w"] = (
        (df["weekly_cases"] - df["cases_lag1"]) / (df["cases_lag1"] + 1)
    )
    df["cases_rolling_mean_3w"] = (
        df.groupby("State")["weekly_cases"]
        .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
    )
    return df


def add_mobility_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lag in MOBILITY_LAG_WEEKS:
        for col in MOBILITY_COLS:
            df[f"{col}_lag{lag}"] = df.groupby("State")[col].shift(lag)
    return df


def add_spatial_lag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rate_lookup = df.set_index(["State", "week"])["case_rate_per_100k"].to_dict()

    weeks_sorted = sorted(df["week"].unique())
    week_to_prev = {w: weeks_sorted[i - 1] if i > 0 else None
                     for i, w in enumerate(weeks_sorted)}

    def spatial_lag_value(row):
        neighbors = STATE_ADJACENCY.get(row["State"], [])
        if not neighbors:
            return np.nan
        prev_week = week_to_prev[row["week"]]
        if prev_week is None:
            return np.nan
        vals = [rate_lookup.get((n, prev_week)) for n in neighbors]
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        if not vals:
            return np.nan
        return float(np.mean(vals))

    df["neighbor_case_rate_lag1"] = df.apply(spatial_lag_value, axis=1)
    return df


def compute_morans_i(df: pd.DataFrame, value_col: str = "case_rate_per_100k",
                      sample_week=None) -> float:
    if sample_week is None:
        sample_week = sorted(df["week"].unique())[len(df["week"].unique()) // 2]

    snapshot = df[df["week"] == sample_week].set_index("State")[value_col].dropna()
    states = snapshot.index.tolist()
    n = len(states)
    if n < 3:
        return np.nan

    x = snapshot.values
    x_bar = x.mean()

    W = 0.0
    numerator = 0.0
    for i, si in enumerate(states):
        for j, sj in enumerate(states):
            if i == j:
                continue
            w_ij = 1.0 if sj in STATE_ADJACENCY.get(si, []) else 0.0
            if w_ij == 0:
                continue
            W += w_ij
            numerator += w_ij * (x[i] - x_bar) * (x[j] - x_bar)

    denominator = np.sum((x - x_bar) ** 2)
    if W == 0 or denominator == 0:
        return np.nan

    morans_i = (n / W) * (numerator / denominator)
    print(f"[build_features] Moran's I for {value_col} on week {sample_week.date()}: "
          f"{morans_i:.4f} (n={n} states)")
    return morans_i


def run(path: Path = MASTER_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    df = load_master(path)

    df = add_case_lags(df)
    df = add_growth_rate(df)
    df = add_mobility_lags(df)
    df = add_spatial_lag(df)

    for week in sorted(df["week"].unique())[::20]:
        compute_morans_i(df, sample_week=week)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n[build_features] Feature table: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[build_features] Written to {out_path}")

    return df


if __name__ == "__main__":
    run()
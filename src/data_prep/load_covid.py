"""
load_covid.py

Loads and cleans the India COVID-19 state-level case data (states.csv from
data.incovid19.org). Converts cumulative counts to daily new cases, drops
invalid/aggregate rows, restricts to the analysis window, and aggregates to
weekly state-level case counts.

Source: https://data.incovid19.org/csv/latest/states.csv
Expected columns: Date, State, Confirmed, Recovered, Deceased, Other, Tested
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/states.csv")
OUT_PATH = Path("data/processed/covid_weekly_state.csv")

DROP_STATES = ["India", "State Unassigned"]

START_DATE = "2020-03-01"
END_DATE = "2022-10-15"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def clean_covid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df["State"].isin(DROP_STATES)]
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    df = df.sort_values(["State", "Date"])

    df["new_cases"] = df.groupby("State")["Confirmed"].diff()
    df["new_cases"] = df["new_cases"].fillna(0)

    n_negative = (df["new_cases"] < 0).sum()
    if n_negative > 0:
        print(f"[load_covid] Clipping {n_negative} negative daily deltas to 0 "
              f"(data correction artifacts).")
    df["new_cases"] = df["new_cases"].clip(lower=0)

    return df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = df["Date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    weekly = (
        df.groupby(["State", "week"], as_index=False)["new_cases"]
        .sum()
        .rename(columns={"new_cases": "weekly_cases"})
    )
    return weekly


def run(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    raw = load_raw(raw_path)
    cleaned = clean_covid(raw)
    weekly = aggregate_weekly(cleaned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_path, index=False)

    print(f"[load_covid] {len(weekly)} state-week rows written to {out_path}")
    print(f"[load_covid] States: {weekly['State'].nunique()}, "
          f"Weeks: {weekly['week'].nunique()}, "
          f"Date range: {weekly['week'].min()} to {weekly['week'].max()}")
    return weekly


if __name__ == "__main__":
    run()
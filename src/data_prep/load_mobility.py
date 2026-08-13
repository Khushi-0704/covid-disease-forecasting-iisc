"""
load_mobility.py

Loads and cleans Google COVID-19 Community Mobility data for India
(2020-2022 Region CSVs), filters to state-level rows, reconciles state
naming with the COVID dataset, and aggregates daily values to weekly
state-level averages for each of the 6 mobility categories.

Source: https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip
Expected files: {year}_IN_Region_Mobility_Report.csv for year in 2020, 2021, 2022
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
YEARS = [2020, 2021, 2022]
FILENAME_TEMPLATE = "{year}_IN_Region_Mobility_Report.csv"
OUT_PATH = Path("data/processed/mobility_weekly_state.csv")

MOBILITY_COLS = [
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]

# Dadra and Nagar Haveli + Daman and Diu merged into a single UT in Jan 2020;
# Google's mobility data still lists them separately. We average their
# mobility values and relabel to match the merged name used in COVID data.
# NOTE: 'Ladakh' has no mobility coverage at all in Google's data -- it will
# end up with all-NaN mobility features; document as a limitation.
MERGE_UT_MAP = {
    "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman and Diu": "Dadra and Nagar Haveli and Daman and Diu",
}


def load_all_years(raw_dir: Path = RAW_DIR, years: list = YEARS) -> pd.DataFrame:
    frames = []
    for year in years:
        path = raw_dir / FILENAME_TEMPLATE.format(year=year)
        df = pd.read_csv(path, low_memory=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def filter_state_level(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["sub_region_1"].notna() & df["sub_region_2"].isna()].copy()


def reconcile_state_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["State"] = df["sub_region_1"].replace(MERGE_UT_MAP)
    return df


def clean_mobility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    state_df = filter_state_level(df)
    state_df = reconcile_state_names(state_df)

    state_df = (
        state_df.groupby(["State", "date"], as_index=False)[MOBILITY_COLS]
        .mean()
    )
    return state_df


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W-SUN").apply(lambda p: p.start_time)

    weekly = (
        df.groupby(["State", "week"], as_index=False)[MOBILITY_COLS]
        .mean()
    )

    rename_map = {c: c.replace("_percent_change_from_baseline", "_pct") for c in MOBILITY_COLS}
    weekly = weekly.rename(columns=rename_map)
    return weekly


def run(raw_dir: Path = RAW_DIR, out_path: Path = OUT_PATH) -> pd.DataFrame:
    raw = load_all_years(raw_dir)
    cleaned = clean_mobility(raw)
    weekly = aggregate_weekly(cleaned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(out_path, index=False)

    print(f"[load_mobility] {len(weekly)} state-week rows written to {out_path}")
    print(f"[load_mobility] States: {weekly['State'].nunique()}, "
          f"Weeks: {weekly['week'].nunique()}, "
          f"Date range: {weekly['week'].min()} to {weekly['week'].max()}")

    pct_cols = [c.replace("_percent_change_from_baseline", "_pct") for c in MOBILITY_COLS]
    missing = weekly[pct_cols].isna().sum()
    print(f"[load_mobility] Missing values per category after weekly aggregation:\n{missing}")

    return weekly


if __name__ == "__main__":
    run()
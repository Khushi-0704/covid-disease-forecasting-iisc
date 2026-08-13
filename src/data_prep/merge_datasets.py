

"""
merge_datasets.py

Merges the four cleaned processed datasets into a single master state-week
table ready for feature engineering and modeling.

Join strategy:
- COVID weekly case data is the backbone (defines the full state-week grid).
- Mobility joins on (State, week) since it's also time-varying. Missing
  weeks (~1-6% per category) are forward-filled per state, since mobility
  doesn't jump discontinuously and this is more defensible than dropping
  rows or imputing a global mean.
- Socioeconomic and Census data are static (one value per state) and are
  broadcast across every week for that state.
- A population-normalized case rate (per 100,000) is added using Census
  population, since raw case counts aren't comparable across states of
  very different sizes.

KNOWN, DOCUMENTED GAPS carried forward from upstream loaders (not silently
imputed away here):
- Ladakh: no mobility data (Google has no coverage) -> mobility columns NaN
- Telangana: no census demographic data (Census 2011 predates its 2014
  creation from Andhra Pradesh) -> census columns NaN
- 6-7 states are missing from the socioeconomic file (only 29 states
  covered) -> socioeconomic columns NaN for those states
- Sikkim: growth/sex-ratio/literacy NaN due to a district-name mismatch
  in the source census files; population/area for Sikkim are unaffected
"""

import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
COVID_PATH = PROCESSED_DIR / "covid_weekly_state.csv"
MOBILITY_PATH = PROCESSED_DIR / "mobility_weekly_state.csv"
SOCIOECONOMIC_PATH = PROCESSED_DIR / "socioeconomic_state.csv"
CENSUS_PATH = PROCESSED_DIR / "census_state.csv"

OUT_PATH = PROCESSED_DIR / "master_state_week.csv"

MOBILITY_COLS = [
    "retail_and_recreation_pct",
    "grocery_and_pharmacy_pct",
    "parks_pct",
    "transit_stations_pct",
    "workplaces_pct",
    "residential_pct",
]


def load_processed():
    covid = pd.read_csv(COVID_PATH, parse_dates=["week"])
    mobility = pd.read_csv(MOBILITY_PATH, parse_dates=["week"])
    socioeconomic = pd.read_csv(SOCIOECONOMIC_PATH)
    census = pd.read_csv(CENSUS_PATH)
    return covid, mobility, socioeconomic, census


def check_state_alignment(covid, mobility, socioeconomic, census):
    covid_states = set(covid["State"].unique())
    mobility_states = set(mobility["State"].unique())
    socio_states = set(socioeconomic["State"].unique())
    census_states = set(census["State"].unique())

    print("[merge] State alignment check (relative to COVID backbone):")
    print(f"  In COVID but not Mobility: {sorted(covid_states - mobility_states)}")
    print(f"  In COVID but not Socioeconomic: {sorted(covid_states - socio_states)}")
    print(f"  In COVID but not Census: {sorted(covid_states - census_states)}")


def merge_mobility(covid: pd.DataFrame, mobility: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(covid, mobility, on=["State", "week"], how="left")
    merged = merged.sort_values(["State", "week"])
    merged[MOBILITY_COLS] = (
        merged.groupby("State")[MOBILITY_COLS].transform(lambda s: s.ffill().bfill())
    )
    return merged


def merge_static(df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(df, static_df, on="State", how="left")


def add_case_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["case_rate_per_100k"] = (df["weekly_cases"] / df["total_population"]) * 100_000
    return df


def summarize_missingness(df: pd.DataFrame):
    print("\n[merge] Missing value summary (post-merge):")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing)

    print("\n[merge] States with any missing values (any column):")
    states_with_na = df[df.isna().any(axis=1)]["State"].unique()
    print(sorted(states_with_na))


def run(out_path: Path = OUT_PATH) -> pd.DataFrame:
    covid, mobility, socioeconomic, census = load_processed()

    check_state_alignment(covid, mobility, socioeconomic, census)

    df = merge_mobility(covid, mobility)
    df = merge_static(df, socioeconomic)
    df = merge_static(df, census)
    df = add_case_rate(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n[merge] Master table: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"[merge] States: {df['State'].nunique()}, "
          f"Weeks: {df['week'].nunique()}, "
          f"Date range: {df['week'].min()} to {df['week'].max()}")
    print(f"[merge] Written to {out_path}")

    summarize_missingness(df)

    return df


if __name__ == "__main__":
    run()
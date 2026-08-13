"""
load_census.py

Loads and merges the two Census 2011 district-level files (area/population,
and growth/sex ratio/literacy), fixes known state-naming issues, and
aggregates up to state-level demographic features.

KNOWN LIMITATIONS (documented, not silently fixed):
- Census 2011 predates Telangana's creation (2014, carved from Andhra
  Pradesh). Districts now in Telangana are still labeled "Andhra Pradesh"
  in this data. We do NOT attempt to reconstruct 2014 state boundaries --
  Telangana will have NaN demographic features after merging with COVID
  data. This is stated explicitly in the report's limitations section.
- 33 districts in the area/population file had a missing State value in
  the raw file (e.g. "Mumbai", "Garhwal", "North" as a Delhi district
  name). These are recovered via a manual lookup table based on known
  Indian district geography, since no automatic match was found between
  the two source files.
- Sikkim's growth/sex-ratio/literacy figures come out as NaN due to a
  district-naming mismatch between the two source files for that state;
  population and area figures are unaffected.

Sources: census2011_area_population.csv, census2011_growth_sexratio_literacy.csv
(Kaggle: Indian Census 2011 Dataset)
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
AREA_POP_PATH = RAW_DIR / "census2011_area_population.csv"
GROWTH_PATH = RAW_DIR / "census2011_growth_sexratio_literacy.csv"
OUT_PATH = Path("data/processed/census_state.csv")

# State name reconciliation to match COVID/mobility naming conventions
STATE_RENAME_MAP = {
    "Orissa": "Odisha",
    "Andaman And Nicobar Islands": "Andaman and Nicobar Islands",
}

# Merge the two UTs that combined in Jan 2020, same as mobility loader
MERGE_UT_MAP = {
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman and Diu": "Dadra and Nagar Haveli and Daman and Diu",
}

# Manual lookup for districts missing a State value in the raw area/population
# file. Based on known Indian district geography.
DISTRICT_STATE_FALLBACK = {
    "Central": "Delhi",
    "Chhatarpur": "Madhya Pradesh",
    "Dadra & Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Dahod": "Gujarat",
    "Dakshin Bastar Dantewada": "Chhattisgarh",
    "East": "Delhi",
    "East District": "Sikkim",
    "Garhwal": "Uttarakhand",
    "Hardwar": "Uttarakhand",
    "Janjgir - Champa": "Chhattisgarh",
    "Kanker (Uttar Bastar Kanker)": "Chhattisgarh",
    "Kanpur Dehat": "Uttar Pradesh",
    "Khandwa": "Madhya Pradesh",
    "Khargone": "Madhya Pradesh",
    "Koriya": "Chhattisgarh",
    "Lahul & Spiti": "Himachal Pradesh",
    "Mumbai": "Maharashtra",
    "North": "Delhi",
    "North & Middle Andaman": "Andaman and Nicobar Islands",
    "North District": "Sikkim",
    "North East": "Delhi",
    "North West": "Delhi",
    "Papum Pare": "Arunachal Pradesh",
    "Purba Champaran": "Bihar",
    "Ribhoi": "Meghalaya",
    "Sahibzada Ajit Singh Nagar": "Punjab",
    "Saraikela-Kharsawan": "Jharkhand",
    "Siddharthnagar": "Uttar Pradesh",
    "South": "Delhi",
    "South District": "Sikkim",
    "South West": "Delhi",
    "West": "Delhi",
    "West District": "Sikkim",
}


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    area_pop = pd.read_csv(AREA_POP_PATH, encoding="utf-8-sig")
    growth = pd.read_csv(GROWTH_PATH, encoding="utf-8-sig")
    return area_pop, growth


def fix_state_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing_mask = df["State"].isna()
    df.loc[missing_mask, "State"] = df.loc[missing_mask, "District"].map(DISTRICT_STATE_FALLBACK)

    still_missing = df["State"].isna().sum()
    if still_missing > 0:
        print(f"[load_census] WARNING: {still_missing} districts still missing State "
              f"after fallback lookup: {df[df['State'].isna()]['District'].tolist()}")

    df["State"] = df["State"].replace(STATE_RENAME_MAP)
    df["State"] = df["State"].replace(MERGE_UT_MAP)

    return df


def merge_district_files(area_pop: pd.DataFrame, growth: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        area_pop, growth,
        on=["District", "State"],
        how="outer",
        suffixes=("", "_dup"),
    )
    return merged


def aggregate_to_state(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def weighted_avg(group, col):
        valid = group.dropna(subset=[col, "Population"])
        if valid["Population"].sum() == 0 or len(valid) == 0:
            return pd.NA
        return (valid[col] * valid["Population"]).sum() / valid["Population"].sum()

    records = []
    for state, group in df.groupby("State"):
        records.append({
            "State": state,
            "total_area_km2": group["Area_km2"].sum(),
            "total_population": group["Population"].sum(),
            "population_density": group["Population"].sum() / group["Area_km2"].sum()
                if group["Area_km2"].sum() > 0 else pd.NA,
            "growth_rate_weighted": weighted_avg(group, "Growth"),
            "sex_ratio_weighted": weighted_avg(group, "Sex_Ratio"),
            "literacy_rate_weighted": weighted_avg(group, "Literacy"),
            "n_districts": len(group),
        })

    return pd.DataFrame(records)


def run(out_path: Path = OUT_PATH) -> pd.DataFrame:
    area_pop, growth = load_raw()

    area_pop = fix_state_names(area_pop)
    growth = fix_state_names(growth)

    merged = merge_district_files(area_pop, growth)
    state_level = aggregate_to_state(merged)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_level.to_csv(out_path, index=False)

    print(f"[load_census] {len(state_level)} states written to {out_path}")
    print(f"[load_census] NOTE: Telangana will have NaN demographic values "
          f"(Census 2011 predates its 2014 creation from Andhra Pradesh) -- "
          f"documented limitation.")
    return state_level


if __name__ == "__main__":
    run()
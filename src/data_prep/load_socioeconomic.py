"""
load_socioeconomic.py

Loads the Statewise Socio-Economic Indicators file (data.gov.in / opencity.in).

IMPORTANT: values in this file are each state's RANK (1-29) on that
indicator relative to other states, not raw magnitudes. Treated as an
ordinal feature; documented as a limitation (rank tables lose magnitude
information, and this is a static ~2012-13 snapshot).

Only 29 of India's states/UTs are covered -- others get NaN after merging.

Source: https://data.opencity.in/dataset/statewise-socio-economic-indicators
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/Statewise_SocioEconomic.csv")
OUT_PATH = Path("data/processed/socioeconomic_state.csv")

SELECTED_INDICATORS = {
    "Total Literacy Rate (%)": "literacy_rate_rank",
    "Infant Mortality Rate (IMR) (per 1000 live births)": "infant_mortality_rank",
    "Per Capita NSDP at 2004-05 prices (Rs.)": "per_capita_nsdp_rank",
    "Unemployment Rate (Usual Status (adjusted)) (%)": "unemployment_rate_rank",
    "Urban Tele Density (%)": "urban_tele_density_rank",
    "Households with latrine facility within premises": "latrine_access_rank",
    "Safe Drinking Water (Tap/Handpump/Tubewell) (%age)": "safe_water_rank",
}


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_socioeconomic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    subset = df[df["Indicator"].isin(SELECTED_INDICATORS.keys())].copy()

    missing_indicators = set(SELECTED_INDICATORS.keys()) - set(subset["Indicator"])
    if missing_indicators:
        print(f"[load_socioeconomic] WARNING: indicators not found in file: {missing_indicators}")

    meta_cols = ["_id", "Sr. No", "Indicator", "Source", "Periodicity/ Latest available data"]
    state_cols = [c for c in df.columns if c not in meta_cols]

    subset = subset.set_index("Indicator")[state_cols].T
    subset.index.name = "State"
    subset = subset.reset_index()
    subset = subset.rename(columns=SELECTED_INDICATORS)

    return subset


def run(raw_path: Path = RAW_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    raw = load_raw(raw_path)
    cleaned = clean_socioeconomic(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)

    print(f"[load_socioeconomic] {len(cleaned)} states written to {out_path}")
    print(f"[load_socioeconomic] Columns: {list(cleaned.columns)}")
    return cleaned


if __name__ == "__main__":
    run()
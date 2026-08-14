# COVID-19 Disease Forecasting: A Spatio-Temporal Analysis of India

An end-to-end analytical workflow integrating COVID-19 case data, Google
mobility data, socio-economic indicators, and Census 2011 demographic data
to forecast disease trends at the Indian state-week level.

## Objective

Develop a spatio-temporal forecasting model for COVID-19 case rates across
Indian states, integrating heterogeneous public data sources, and evaluate
model performance against classical time-series baselines.

## Repository Structure

```
├── data/
│   ├── raw/              # Original downloaded datasets (not committed -- see Data Sources below)
│   └── processed/        # Cleaned, merged, and feature-engineered datasets (not committed)
├── src/
│   ├── data_prep/
│   │   ├── load_covid.py           # COVID-19 case data loader (data.incovid19.org)
│   │   ├── load_mobility.py        # Google Mobility data loader
│   │   ├── load_socioeconomic.py   # Statewise socio-economic indicators loader
│   │   ├── load_census.py          # Census 2011 demographic data loader
│   │   └── merge_datasets.py       # Merges all four into one master state-week table
│   ├── features/
│   │   ├── state_adjacency.py      # India state adjacency map (for spatial-lag features)
│   │   └── build_features.py       # Lag features, growth rate, spatial-lag, Moran's I
│   ├── models/
│   │   ├── baseline_models.py      # Naive persistence + per-state ARIMA(1,1,1)
│   │   ├── train_xgboost.py        # Main model: pooled XGBoost regressor
│   │   └── evaluate.py             # MAE/RMSE/MAPE comparison, overall and by burden tier
│   └── viz/
│       ├── eda_plots.py                  # Predicted-vs-actual trajectories, MAE chart, feature importance
│       └── spatial_temporal_plots.py     # State x Week heatmap, Moran's I over time
├── outputs/
│   ├── figures/           # All generated PNG visualizations
│   └── model_results/     # Metrics CSVs, feature importance CSV
├── report/
│   └── report.pdf         # 2-page analytical report
└── requirements.txt
```

## Data Sources

| Dataset | Source | Granularity | Notes |
|---|---|---|---|
| COVID-19 case data | [data.incovid19.org](https://data.incovid19.org/csv/latest/states.csv) | State, daily → aggregated to weekly | Cumulative counts converted to new-case deltas |
| Google Mobility | [google.com/covid19/mobility](https://www.google.com/covid19/mobility/) | State, daily → weekly | Data collection ended Oct 2022 |
| Socio-economic indicators | [data.opencity.in](https://data.opencity.in/dataset/statewise-socio-economic-indicators) | State (29 of 36 covered) | Values are **relative ranks (1-29)**, not raw magnitudes -- see Limitations |
| Census 2011 demographic data | Kaggle: Indian Census 2011 Dataset | District → aggregated to state | Predates Telangana's 2014 creation -- see Limitations |

Raw data files are not committed to this repository (see `.gitignore`) due to
size and licensing considerations. Download links above; place files in
`data/raw/` using the exact filenames referenced in each loader script's
docstring before running the pipeline.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

## How to Run

Scripts must be run in this order, as each stage depends on the previous
stage's output:

```bash
# 1. Clean and standardize each raw data source
python src/data_prep/load_covid.py
python src/data_prep/load_mobility.py
python src/data_prep/load_socioeconomic.py
python src/data_prep/load_census.py

# 2. Merge into one master state-week table
python src/data_prep/merge_datasets.py

# 3. Engineer lag, spatial-lag, and growth-rate features
python src/features/build_features.py

# 4. Train and evaluate models
python src/models/baseline_models.py
python src/models/train_xgboost.py
python src/models/evaluate.py

# 5. Generate visualizations
python src/viz/eda_plots.py
python src/viz/spatial_temporal_plots.py
```

## Methodology Summary

- **Spatio-temporal scale:** State-week (36 states/UTs x 137 weeks, Mar
  2020 - Oct 2022). District-level case data exists, but mobility and
  socio-economic sources are only available at state resolution; modeling
  at district level would have joined coarser state-level covariates onto
  finer case data, causing pseudo-replication. State-week was chosen as the
  common, statistically defensible resolution across all four sources.
- **Target variable:** `case_rate_per_100k` -- population-normalized weekly
  case rate, to allow fair comparison across states of very different sizes.
- **Features:** lagged case counts/rates (1-3 weeks), 3-week rolling case
  mean, week-over-week growth rate, lagged mobility indicators (1-2 weeks,
  since mobility changes precede case changes), a spatial-lag feature
  (average case rate among geographically adjacent states, lag 1), and
  static socio-economic/demographic covariates.
- **Spatial autocorrelation (Moran's I):** computed weekly using a binary
  state-adjacency matrix. Found to rise sharply during active outbreak
  waves (e.g. ~0.44 in Jul 2020, ~0.35 in mid-2021) and fall toward zero
  during inter-wave lulls -- consistent with geographic spread mattering
  most during periods of active community transmission.
- **Models compared:** naive persistence, per-state ARIMA(1,1,1), and a
  pooled XGBoost regressor (single model trained across all states, using
  each state's own feature values to differentiate predictions -- chosen
  over per-state models because ~137 weeks per state is insufficient data
  for 36 independent models).
- **Evaluation:** time-based train/test split (last 8 weeks held out;
  never randomly split, to avoid future-data leakage). Metrics (MAE, RMSE,
  MAPE) computed overall and separately for high- vs low-burden states
  (median split), since strong average performance can mask poor
  performance during actual outbreak spikes.

## Key Results

| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Naive Persistence | 3.74 | 10.30 | 165.5 |
| ARIMA(1,1,1) per state | 9.78 | 17.91 | 673.9 |
| **XGBoost (pooled, full features)** | **1.273** | **5.395** | **99.33** |

XGBoost outperforms both baselines overall and specifically on high-burden
states (MAE 2.03 vs. 7.03 for naive and 17.50 for ARIMA), including
correctly tracking Kerala's mid-September 2022 case spike that both
baselines missed (see `outputs/figures/predicted_vs_actual_trajectories.png`).

**Important caveat:** ARIMA was evaluated via true multi-step (8-week-ahead)
forecasting with no access to intervening actuals, while naive persistence
and XGBoost's lag features use real historical data at each step
(effectively one-step-ahead). This makes ARIMA's error inflated relative to
a fully fair comparison; a rolling one-step-ahead ARIMA would likely
perform closer to naive persistence. Documented here rather than silently
left out, per the same standard applied to imputation and missing-data
handling throughout this pipeline.

## Known Limitations

- **Telangana** has no Census 2011 demographic data (created in 2014 from
  Andhra Pradesh, after the census was conducted).
- **Ladakh** has no Google Mobility coverage.
- **7 states/UTs** are absent from the socio-economic indicators file
  (only 29 of 36 covered).
- **Sikkim's** growth rate, sex ratio, and literacy figures are missing due
  to a district-naming mismatch between the two Census source files
  (population and area are unaffected).
- Socio-economic indicators are **relative ranks (1-29)**, not raw
  magnitudes, and reflect a static ~2012-13 snapshot.
- Census demographic data is from **2011** -- the most recent full Indian
  census available (2021 census was postponed; the next one only began its
  house-listing phase in 2026).
- Missing values across all sources were deliberately **not imputed**;
  XGBoost handles them natively via learned split-direction, avoiding the
  introduction of synthetic values not reflecting real measurements.

## AI Usage Disclosure

AI tools (Claude) were used for:
- Scaffolding and writing the data-loading, cleaning, feature-engineering,
  and modeling scripts in this repository, under active direction and
  review at each step (real output was inspected and validated against
  actual data at every stage, not generated blind).
- Drafting this README and the accompanying report structure.

AI tools were **not** used for: the underlying analytical and methodological
decisions (spatio-temporal scale, feature selection, model choice,
evaluation design), which were made and are owned by the author, nor for
generating any of the forecasts themselves -- all case-rate predictions are
produced by the statistical/ML models described above, not by a language
model.

## Author

Khushi A Kumar -- IISc Internship Application, 2026
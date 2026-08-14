"""
baseline_models.py

Baseline forecasting models to compare the main model against:
1. Naive persistence: predict next week's case rate = this week's lagged
   case rate. The simplest possible baseline -- if the main model can't
   beat this, it isn't adding real value.
2. Per-state ARIMA(1,1,1): a classic univariate epidemiological
   time-series baseline, fit independently on each state's own case-rate
   history. Some states may fail to converge -- these fall back to naive
   persistence, logged explicitly rather than silently dropped.

NOTE ON FAIRNESS: ARIMA forecasts the full 8-week test horizon in one shot
(true multi-step forecasting, no access to intervening actuals), while
naive persistence and the XGBoost model's lag features use real historical
data at each row (effectively one-step-ahead). This makes ARIMA's error
look worse than a fully fair comparison would -- documented as a known
limitation rather than hidden.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("data/processed")
FEATURES_PATH = PROCESSED_DIR / "features_state_week.csv"
OUT_PATH = PROCESSED_DIR / "baseline_predictions.csv"

TEST_WEEKS = 8
TARGET_COL = "case_rate_per_100k"


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week"])
    return df.sort_values(["State", "week"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_weeks: int = TEST_WEEKS):
    weeks_sorted = sorted(df["week"].unique())
    test_week_set = set(weeks_sorted[-test_weeks:])
    train = df[~df["week"].isin(test_week_set)].copy()
    test = df[df["week"].isin(test_week_set)].copy()
    return train, test


def naive_persistence_predictions(test: pd.DataFrame) -> pd.Series:
    return test["case_rate_lag1"]


def fit_arima_per_state(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    from statsmodels.tsa.arima.model import ARIMA

    predictions = pd.Series(index=test.index, dtype=float)
    n_fallback = 0

    for state in test["State"].unique():
        state_train = train[train["State"] == state].sort_values("week")
        state_test = test[test["State"] == state].sort_values("week")

        series = state_train[TARGET_COL].values
        n_forecast = len(state_test)

        if len(series) < 10:
            predictions.loc[state_test.index] = state_test["case_rate_lag1"].values
            n_fallback += 1
            continue

        try:
            model = ARIMA(series, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=n_forecast)
            forecast = np.clip(forecast, 0, None)
            predictions.loc[state_test.index] = forecast
        except Exception:
            predictions.loc[state_test.index] = state_test["case_rate_lag1"].values
            n_fallback += 1

    print(f"[baseline_models] ARIMA: {n_fallback} of {test['State'].nunique()} states "
          f"fell back to naive persistence.")
    return predictions


def run(path: Path = FEATURES_PATH, out_path: Path = OUT_PATH) -> pd.DataFrame:
    df = load_features(path)
    df = df.dropna(subset=[TARGET_COL, "case_rate_lag1"])

    train, test = time_split(df)
    print(f"[baseline_models] Train: {len(train)} rows, Test: {len(test)} rows "
          f"({TEST_WEEKS} weeks held out)")

    test = test.copy()
    test["pred_naive"] = naive_persistence_predictions(test)
    test["pred_arima"] = fit_arima_per_state(train, test)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    test[["State", "week", TARGET_COL, "pred_naive", "pred_arima"]].to_csv(out_path, index=False)

    print(f"[baseline_models] Predictions written to {out_path}")
    return test


if __name__ == "__main__":
    run()
"""
evaluate.py

Compares the three forecasting approaches on the same held-out test weeks:
1. Naive persistence
2. Per-state ARIMA
3. Pooled XGBoost (main model)

Metrics: MAE, RMSE, MAPE -- overall AND separately for "high-burden" vs
"low-burden" states, since
a model can look good on average while failing during real outbreak spikes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
BASELINE_PATH = PROCESSED_DIR / "baseline_predictions.csv"
XGBOOST_PATH = PROCESSED_DIR / "xgboost_predictions.csv"
OUT_METRICS_PATH = Path("outputs/model_results/model_comparison_metrics.csv")

TARGET_COL = "case_rate_per_100k"


def load_predictions():
    baseline = pd.read_csv(BASELINE_PATH, parse_dates=["week"])
    xgboost_preds = pd.read_csv(XGBOOST_PATH, parse_dates=["week"])
    merged = pd.merge(
        baseline, xgboost_preds[["State", "week", "pred_xgboost"]],
        on=["State", "week"], how="inner"
    )
    return merged


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    nonzero = y_true > 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if nonzero.sum() > 0 else np.nan
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3), "MAPE_%": round(mape, 2)}


def evaluate_by_burden_tier(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    state_avg = df.groupby("State")[TARGET_COL].mean()
    median_rate = state_avg.median()
    high_burden_states = state_avg[state_avg >= median_rate].index

    df = df.copy()
    df["burden_tier"] = np.where(df["State"].isin(high_burden_states), "high_burden", "low_burden")

    rows = []
    for tier in ["high_burden", "low_burden"]:
        subset = df[df["burden_tier"] == tier]
        metrics = compute_metrics(subset[TARGET_COL], subset[model_col])
        metrics["tier"] = tier
        metrics["model"] = model_col
        rows.append(metrics)
    return pd.DataFrame(rows)


def run(out_path: Path = OUT_METRICS_PATH) -> pd.DataFrame:
    df = load_predictions()

    models = {
        "pred_naive": "Naive Persistence",
        "pred_arima": "ARIMA(1,1,1) per state",
        "pred_xgboost": "XGBoost (pooled, full features)",
    }

    overall_rows = []
    tier_rows = []

    for col, label in models.items():
        metrics = compute_metrics(df[TARGET_COL], df[col])
        metrics["model"] = label
        overall_rows.append(metrics)

        tier_metrics = evaluate_by_burden_tier(df, col)
        tier_metrics["model"] = label
        tier_rows.append(tier_metrics)

    overall_df = pd.DataFrame(overall_rows)[["model", "MAE", "RMSE", "MAPE_%"]]
    tier_df = pd.concat(tier_rows)[["model", "tier", "MAE", "RMSE", "MAPE_%"]]

    print("[evaluate] Overall metrics (all states, all test weeks):")
    print(overall_df.to_string(index=False))
    print()
    print("[evaluate] Metrics by burden tier:")
    print(tier_df.to_string(index=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(out_path, index=False)
    tier_df.to_csv(str(out_path).replace(".csv", "_by_tier.csv"), index=False)
    print(f"\n[evaluate] Metrics written to {out_path}")

    return overall_df, tier_df


if __name__ == "__main__":
    run()
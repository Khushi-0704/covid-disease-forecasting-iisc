"""
eda_plots.py

Generates the key visualizations for the report:
1. Predicted vs actual case-rate trajectories for representative states
2. Bar chart comparing MAE across the three models
3. Feature importance chart from the XGBoost model
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("outputs/figures")

TARGET_COL = "case_rate_per_100k"
STATES_TO_PLOT = ["Kerala", "Maharashtra", "Bihar"]


def load_data():
    baseline = pd.read_csv(PROCESSED_DIR / "baseline_predictions.csv", parse_dates=["week"])
    xgboost_preds = pd.read_csv(PROCESSED_DIR / "xgboost_predictions.csv", parse_dates=["week"])
    merged = pd.merge(
        baseline, xgboost_preds[["State", "week", "pred_xgboost"]],
        on=["State", "week"], how="inner"
    )
    return merged


def plot_state_trajectories(df: pd.DataFrame, out_dir: Path = OUT_DIR):
    fig, axes = plt.subplots(len(STATES_TO_PLOT), 1, figsize=(9, 10), sharex=False)

    for ax, state in zip(axes, STATES_TO_PLOT):
        subset = df[df["State"] == state].sort_values("week")
        ax.plot(subset["week"], subset[TARGET_COL], label="Actual", color="black", linewidth=2, marker="o")
        ax.plot(subset["week"], subset["pred_naive"], label="Naive", linestyle="--", alpha=0.6)
        ax.plot(subset["week"], subset["pred_arima"], label="ARIMA", linestyle="--", alpha=0.6)
        ax.plot(subset["week"], subset["pred_xgboost"], label="XGBoost", linewidth=2, color="tab:red")
        ax.set_title(f"{state}: Predicted vs Actual Case Rate (test period)")
        ax.set_ylabel("Cases per 100k")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "predicted_vs_actual_trajectories.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[eda_plots] Saved {out_path}")


def plot_metrics_comparison(out_dir: Path = OUT_DIR):
    metrics_path = Path("outputs/model_results/model_comparison_metrics.csv")
    metrics = pd.read_csv(metrics_path)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(metrics["model"], metrics["MAE"], color=["tab:gray", "tab:orange", "tab:red"])
    ax.set_ylabel("MAE (cases per 100k)")
    ax.set_title("Model Comparison: Mean Absolute Error (lower is better)")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    out_path = out_dir / "model_comparison_mae.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[eda_plots] Saved {out_path}")


def plot_feature_importance(out_dir: Path = OUT_DIR, top_n: int = 15):
    importance_path = Path("outputs/model_results/feature_importance.csv")
    importance = pd.read_csv(importance_path).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"][::-1], importance["importance"][::-1], color="tab:blue")
    ax.set_xlabel("Importance")
    ax.set_title(f"XGBoost: Top {top_n} Feature Importances")
    plt.tight_layout()

    out_path = out_dir / "feature_importance.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[eda_plots] Saved {out_path}")


def run():
    df = load_data()
    plot_state_trajectories(df)
    plot_metrics_comparison()
    plot_feature_importance()


if __name__ == "__main__":
    run()
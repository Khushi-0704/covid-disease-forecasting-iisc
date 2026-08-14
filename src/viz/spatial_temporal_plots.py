"""
spatial_temporal_plots.py

Two exploratory spatio-temporal visualizations, distinct from the model
performance plots in eda_plots.py:

1. State x Week heatmap of case_rate_per_100k -- shows pandemic waves and
   which states rose/fell together, at a glance, across the full timeline.
2. Moran's I over time -- a line chart tracking spatial autocorrelation
   strength week by week across the full dataset, showing how geographic
   clustering of case rates strengthened and weakened across different
   phases of the pandemic.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "features"))
from state_adjacency import STATE_ADJACENCY

PROCESSED_DIR = Path("data/processed")
FEATURES_PATH = PROCESSED_DIR / "features_state_week.csv"
OUT_DIR = Path("outputs/figures")

TARGET_COL = "case_rate_per_100k"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH, parse_dates=["week"])
    return df


def plot_heatmap(df: pd.DataFrame, out_dir: Path = OUT_DIR):
    pivot = df.pivot_table(index="State", columns="week", values=TARGET_COL)

    peak_order = pivot.max(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[peak_order]

    fig, ax = plt.subplots(figsize=(14, 10))
    vmax = np.nanpercentile(pivot.values, 95)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmax=vmax)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    n_weeks = len(pivot.columns)
    tick_positions = np.linspace(0, n_weeks - 1, 15).astype(int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pivot.columns[i].strftime("%b %Y") for i in tick_positions],
                        rotation=45, ha="right", fontsize=8)

    ax.set_title("COVID-19 Case Rate (per 100k) by State and Week\n"
                  "(states sorted by peak case rate; color capped at 95th percentile)")
    fig.colorbar(im, ax=ax, label="Cases per 100k")

    plt.tight_layout()
    out_path = out_dir / "spatiotemporal_heatmap.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[spatial_temporal_plots] Saved {out_path}")


def compute_morans_i_single_week(df: pd.DataFrame, week, value_col: str = TARGET_COL):
    snapshot = df[df["week"] == week].set_index("State")[value_col].dropna()
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

    return (n / W) * (numerator / denominator)


def plot_morans_i_over_time(df: pd.DataFrame, out_dir: Path = OUT_DIR):
    weeks = sorted(df["week"].unique())
    results = []
    for week in weeks:
        i_val = compute_morans_i_single_week(df, week)
        results.append({"week": week, "morans_i": i_val})

    morans_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(morans_df["week"], morans_df["morans_i"], color="tab:purple", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.fill_between(morans_df["week"], morans_df["morans_i"], 0,
                     where=(morans_df["morans_i"] > 0), color="tab:red", alpha=0.15,
                     label="Positive clustering")
    ax.fill_between(morans_df["week"], morans_df["morans_i"], 0,
                     where=(morans_df["morans_i"] <= 0), color="tab:blue", alpha=0.15,
                     label="No / negative clustering")
    ax.set_ylabel("Moran's I")
    ax.set_title("Spatial Autocorrelation of Case Rate Over Time\n"
                  "(positive = neighboring states had similar case rates that week)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    out_path = out_dir / "morans_i_over_time.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[spatial_temporal_plots] Saved {out_path}")

    morans_df.to_csv(Path("outputs/model_results/morans_i_by_week.csv"), index=False)
    return morans_df


def run():
    df = load_data()
    plot_heatmap(df)
    plot_morans_i_over_time(df)


if __name__ == "__main__":
    run()
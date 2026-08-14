"""
shap_analysis.py

Adds interpretability on top of the plain feature importance chart: SHAP
(SHapley Additive exPlanations) values show not just WHICH features
mattered, but HOW -- i.e. whether a high value of a feature pushes the
prediction up or down, and by how much, for each individual state-week.

Reuses the same train/test split and feature set as train_xgboost.py so
the SHAP explanation is for the exact same model already reported in the
results section, not a separate retrain with different settings.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).parent))
from train_xgboost import (
    load_features, time_split, get_feature_cols, XGB_PARAMS,
    FEATURES_PATH, TARGET_COL,
)

OUT_DIR = Path("outputs/figures")


def run():
    df = load_features(FEATURES_PATH)
    df = df.dropna(subset=[TARGET_COL])

    train, test = time_split(df)
    feature_cols = get_feature_cols(df)

    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test = test[feature_cols]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.title("SHAP Summary: Feature Impact on Predicted Case Rate")
    plt.tight_layout()

    out_path = OUT_DIR / "shap_summary.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[shap_analysis] Saved {out_path}")

    mean_abs_shap = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    out_table_path = Path("outputs/model_results/shap_importance.csv")
    out_table_path.parent.mkdir(parents=True, exist_ok=True)
    mean_abs_shap.to_csv(out_table_path, index=False)
    print(f"[shap_analysis] Saved {out_table_path}")
    print("\n[shap_analysis] Top 10 features by mean |SHAP value|:")
    print(mean_abs_shap.head(10).to_string(index=False))


if __name__ == "__main__":
    run()
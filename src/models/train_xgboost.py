"""
train_xgboost.py

Trains the main forecasting model: a single XGBoost regressor pooled across
all states (not one model per state -- with only ~137 weeks of history per
state, per-state models would have too little data; pooling shares learned
relationships across states while lag/spatial/demographic features still
differentiate predictions per state).

Target: case_rate_per_100k. Same time-based train/test split as the
baseline models, for a fair comparison.

Missing values (documented gaps: Ladakh mobility, Telangana census,
socioeconomic coverage gaps) are left as NaN and handled natively by
XGBoost's split-direction learning -- not imputed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb

PROCESSED_DIR = Path("data/processed")
FEATURES_PATH = PROCESSED_DIR / "features_state_week.csv"
OUT_PREDICTIONS_PATH = PROCESSED_DIR / "xgboost_predictions.csv"
OUT_IMPORTANCE_PATH = Path("outputs/model_results/feature_importance.csv")

TEST_WEEKS = 8
TARGET_COL = "case_rate_per_100k"

NON_FEATURE_COLS = ["State", "week", "weekly_cases", "case_rate_per_100k"]

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["week"])
    return df.sort_values(["State", "week"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_weeks: int = TEST_WEEKS):
    weeks_sorted = sorted(df["week"].unique())
    test_week_set = set(weeks_sorted[-test_weeks:])
    train = df[~df["week"].isin(test_week_set)].copy()
    test = df[df["week"].isin(test_week_set)].copy()
    return train, test


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def run(path: Path = FEATURES_PATH,
        out_predictions_path: Path = OUT_PREDICTIONS_PATH,
        out_importance_path: Path = OUT_IMPORTANCE_PATH) -> pd.DataFrame:

    df = load_features(path)
    df = df.dropna(subset=[TARGET_COL])

    train, test = time_split(df)
    feature_cols = get_feature_cols(df)

    print(f"[train_xgboost] Train: {len(train)} rows, Test: {len(test)} rows")
    print(f"[train_xgboost] Using {len(feature_cols)} features")

    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)

    test = test.copy()
    test["pred_xgboost"] = model.predict(X_test)
    test["pred_xgboost"] = test["pred_xgboost"].clip(lower=0)

    out_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test[["State", "week", TARGET_COL, "pred_xgboost"]].to_csv(out_predictions_path, index=False)
    print(f"[train_xgboost] Predictions written to {out_predictions_path}")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    out_importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out_importance_path, index=False)
    print(f"[train_xgboost] Feature importances written to {out_importance_path}")
    print(f"\n[train_xgboost] Top 10 most important features:")
    print(importance.head(10).to_string(index=False))

    return test, importance


if __name__ == "__main__":
    run()
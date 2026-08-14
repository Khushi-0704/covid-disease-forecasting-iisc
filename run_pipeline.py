"""
run_pipeline.py

Runs the entire analytical workflow end-to-end with a single command --
from raw data cleaning through model training, evaluation, and
visualization. Each stage's run() function is called in dependency order.

This exists so the project is genuinely reproducible: anyone (including a
reviewer) can clone the repo, place the raw datasets in data/raw/ per the
README's Data Sources section, and regenerate every processed file, model,
metric, and figure with one command -- rather than manually running each
of the 11 individual scripts in the correct order.

Usage:
    python run_pipeline.py
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

STAGES = [
    ("Data Prep: COVID", "data_prep.load_covid"),
    ("Data Prep: Mobility", "data_prep.load_mobility"),
    ("Data Prep: Socioeconomic", "data_prep.load_socioeconomic"),
    ("Data Prep: Census", "data_prep.load_census"),
    ("Merge Datasets", "data_prep.merge_datasets"),
    ("Feature Engineering", "features.build_features"),
    ("Baseline Models (Naive + ARIMA)", "models.baseline_models"),
    ("Main Model (XGBoost)", "models.train_xgboost"),
    ("Evaluation", "models.evaluate"),
    ("SHAP Interpretability", "models.shap_analysis"),
    ("Visualizations: Model Performance", "viz.eda_plots"),
    ("Visualizations: Spatio-Temporal", "viz.spatial_temporal_plots"),
]


def run_pipeline():
    print("=" * 70)
    print("DISEASE FORECASTING PIPELINE -- FULL RUN")
    print("=" * 70)

    failed_stages = []

    for stage_name, module_path in STAGES:
        print(f"\n{'-' * 70}")
        print(f">>> STAGE: {stage_name}")
        print(f"{'-' * 70}")

        start = time.time()
        try:
            module = __import__(module_path, fromlist=["run"])
            module.run()
            elapsed = time.time() - start
            print(f">>> DONE: {stage_name} ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f">>> FAILED: {stage_name} ({elapsed:.1f}s) -- {type(e).__name__}: {e}")
            failed_stages.append(stage_name)

    print("\n" + "=" * 70)
    if failed_stages:
        print(f"PIPELINE COMPLETED WITH {len(failed_stages)} FAILED STAGE(S):")
        for s in failed_stages:
            print(f"  - {s}")
        print("Check the raw data files in data/raw/ match the filenames")
        print("expected in each script's docstring (see README Data Sources).")
    else:
        print("PIPELINE COMPLETED SUCCESSFULLY -- ALL STAGES PASSED")
    print("=" * 70)

    return len(failed_stages) == 0


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
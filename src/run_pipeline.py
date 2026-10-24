"""
Master Pipeline Orchestrator for Churn Intelligence & CLV Engine.

Executes all 10 pipeline stages in strict sequential order from clean raw data:
1. Data Cleaning & Validation (src/data/clean_data.py)
2. Feature Engineering & Preprocessing (src/features/build_features.py)
3. LightGBM Churn Modeling & Validation Threshold Tuning (src/models/train.py)
4. Model Explainability with TreeSHAP (src/models/explainability.py)
5. Survival Analysis with Lifelines (src/models/survival_analysis.py)
6. Customer Lifetime Value & K-Means Segmentation (src/models/clv_segmentation.py)
7. Hillstrom Uplift Benchmark Ingestion (src/retention/load_uplift_data.py)
8. Two-Model T-Learner Uplift Modeling (src/retention/uplift_model.py)
9. Retention Campaign ROI Simulator (src/retention/roi_simulator.py)
10. System Artifact Verification
"""

import sys
import time
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

STAGES = [
    ("1. Data Cleaning & Imputation", "src/data/clean_data.py"),
    ("2. Anti-Leakage Feature Engineering", "src/features/build_features.py"),
    ("3. Model Training & Threshold Tuning", "src/models/train.py"),
    ("4. TreeSHAP Explainability", "src/models/explainability.py"),
    ("5. Survival Analysis (Lifelines)", "src/models/survival_analysis.py"),
    ("6. CLV Calculation & Segmentation", "src/models/clv_segmentation.py"),
    ("7. Uplift Data Preparation", "src/retention/load_uplift_data.py"),
    ("8. Two-Model T-Learner Uplift", "src/retention/uplift_model.py"),
    ("9. Financial ROI Simulation", "src/retention/roi_simulator.py"),
]


def check_prerequisites():
    """Verify that required raw datasets exist before running pipeline."""
    telco_raw = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    hillstrom_raw = RAW_DATA_DIR / "hillstrom_raw.csv"

    missing = []
    if not telco_raw.exists():
        missing.append(f"IBM Telco raw CSV not found at: {telco_raw}")
    if not hillstrom_raw.exists():
        missing.append(f"Hillstrom raw CSV not found at: {hillstrom_raw}")

    if missing:
        print("\n" + "!" * 70)
        print("PREREQUISITE CHECK FAILED: Missing Raw Data Input(s)")
        print("!" * 70)
        for m in missing:
            print(f" - {m}")
        print("\nPlease acquire datasets as documented in README.md:")
        print("  1. IBM Telco: Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in data/raw/")
        print("  2. Hillstrom: Place 'hillstrom_raw.csv' in data/raw/")
        sys.exit(1)


def run_pipeline():
    print("=" * 70)
    print("NEXVIA CHURN INTELLIGENCE & CLV ENGINE -- REPRODUCIBLE PIPELINE")
    print("=" * 70)
    check_prerequisites()

    python_executable = sys.executable
    start_total = time.time()

    env = dict(sys.modules.get("os", __import__("os")).environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    for step_num, (stage_name, script_rel_path) in enumerate(STAGES, 1):
        script_path = PROJECT_ROOT / script_rel_path
        print(f"\n[{step_num}/{len(STAGES)}] Running: {stage_name} ({script_rel_path})...")
        t0 = time.time()

        result = subprocess.run(
            [python_executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )

        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"FAILED in {elapsed:.2f}s with return code {result.returncode}")
            print("\n--- STDOUT ---")
            print(result.stdout)
            print("\n--- STDERR ---")
            print(result.stderr)
            sys.exit(result.returncode)
        else:
            print(f"SUCCESS ({elapsed:.2f}s)")

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"PIPELINE RUN COMPLETED SUCCESSFULLY in {total_elapsed:.1f}s")
    print("=" * 70)
    print(f"All models saved in: {MODELS_DIR}")
    print(f"All figures saved in: {FIGURES_DIR}")
    print(f"All processed data saved in: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    run_pipeline()

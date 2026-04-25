from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


from airflow import DAG
from airflow.operators.bash import BashOperator


DAG_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = DAG_DIR / "ml_pipeline" / "train_chess_model.py"
PYTHON_BIN = sys.executable

# Local temporary outputs on Composer workers.
LOCAL_ARTIFACT_PATH = "/tmp/chess_win_model_bundle.pkl"
LOCAL_REPORT_PATH = "/tmp/reports/chess_model_report_{{ ds }}.pdf"

# Your GCS bucket and prefix, hardcoded from the browser URL you gave.
GCS_BUCKET = "us-central1-chess-win-predi-8c3082fa-bucket"
GCS_PREFIX = "chess/monthly"

# Persistent destination in GCS.
GCS_ARTIFACT_DATED = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/chess_win_model_bundle_{{{{ ds_nodash }}}}.pkl"
GCS_ARTIFACT_LATEST = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/chess_win_model_bundle_latest.pkl"

GCS_REPORT_DATED = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/reports/chess_model_report_{{{{ ds }}}}.pdf"
GCS_REPORT_LATEST = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/reports/chess_model_report_latest.pdf"


with DAG(
    dag_id="monthly_chess_model_retraining_composer",
    description="Composer-ready monthly retraining with GCS artifact persistence.",
    start_date=datetime(2026, 1, 1),
    schedule="0 3 21 * *",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "monthly", "chess", "composer"],
) as dag:
    run_monthly_retraining = BashOperator(
        task_id="run_monthly_retraining",
        cwd=str(DAG_DIR),
        bash_command=f"""
set -euo pipefail

if [ ! -f "{TRAINING_SCRIPT}" ]; then
  echo "Training script missing at {TRAINING_SCRIPT}"
  exit 1
fi

mkdir -p /tmp/reports

{PYTHON_BIN} {TRAINING_SCRIPT} --run-date {{{{ ds }}}} --artifact-path {LOCAL_ARTIFACT_PATH}

if [ ! -f "{LOCAL_ARTIFACT_PATH}" ]; then
  echo "Expected artifact missing at {LOCAL_ARTIFACT_PATH}"
  exit 1
fi

if [ ! -f "{LOCAL_REPORT_PATH}" ]; then
  echo "Expected report missing at {LOCAL_REPORT_PATH}"
  exit 1
fi

gsutil cp "{LOCAL_ARTIFACT_PATH}" "{GCS_ARTIFACT_DATED}"
gsutil cp "{LOCAL_ARTIFACT_PATH}" "{GCS_ARTIFACT_LATEST}"

gsutil cp "{LOCAL_REPORT_PATH}" "{GCS_REPORT_DATED}"
gsutil cp "{LOCAL_REPORT_PATH}" "{GCS_REPORT_LATEST}"

echo "Training artifact and latest artifact uploaded successfully."
""".strip(),
    )
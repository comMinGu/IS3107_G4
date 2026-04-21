"""Cloud Composer-ready DAG for monthly chess model retraining.

This draft keeps the existing local DAG untouched and adds a Composer-focused
variant that:
1) runs retraining on the Composer worker, writing outputs to /tmp, and
2) uploads the artifact bundle and PDF report to GCS for persistence.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator


DAG_DIR = Path(__file__).resolve().parent
TRAINING_SCRIPT = DAG_DIR / "ml_pipeline" / "train_chess_model.py"
PYTHON_BIN = sys.executable

# Configure in Airflow Variables (Admin > Variables):
# - GCS_ML_ARTIFACT_BUCKET: e.g. "my-ml-artifacts-bucket" (no gs:// prefix)
# - GCS_ML_ARTIFACT_PREFIX: e.g. "chess/monthly" (optional)
GCS_BUCKET = Variable.get("GCS_ML_ARTIFACT_BUCKET", default_var="")
GCS_PREFIX = Variable.get("GCS_ML_ARTIFACT_PREFIX", default_var="chess/monthly")

# Local temporary outputs on Composer workers.
LOCAL_ARTIFACT_PATH = "/tmp/chess_win_model_bundle.pkl"
LOCAL_REPORT_PATH = "/tmp/reports/chess_model_report_{{ ds }}.pdf"

# Persistent destination in GCS.
GCS_ARTIFACT_URI = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/chess_win_model_bundle_{{{{ ds_nodash }}}}.pkl"
GCS_REPORT_URI = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/reports/chess_model_report_{{{{ ds }}}}.pdf"


with DAG(
    dag_id="monthly_chess_model_retraining_composer",
    description="Composer-ready monthly retraining with GCS artifact persistence.",
    start_date=datetime(2026, 1, 21),
    schedule="0 3 21 * *",
    catchup=False,
    tags=["ml", "monthly", "chess", "composer"],
) as dag:
    run_monthly_retraining = BashOperator(
        task_id="run_monthly_retraining",
        cwd=str(DAG_DIR),
        bash_command=f"""
set -euo pipefail

if [ -z "{GCS_BUCKET}" ]; then
  echo "Airflow Variable GCS_ML_ARTIFACT_BUCKET is required."
  exit 1
fi

if [ ! -f "{TRAINING_SCRIPT}" ]; then
  echo "Training script missing at {TRAINING_SCRIPT}"
  exit 1
fi

{PYTHON_BIN} {TRAINING_SCRIPT} --run-date {{{{ ds }}}} --artifact-path {LOCAL_ARTIFACT_PATH}

gsutil cp {LOCAL_ARTIFACT_PATH} "{GCS_ARTIFACT_URI}"
gsutil cp {LOCAL_REPORT_PATH} "{GCS_REPORT_URI}"
""".strip(),
    )

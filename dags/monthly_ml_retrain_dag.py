"""Airflow DAG for monthly chess model retraining."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PROJECT_ROOT / "ml_pipeline" / "train_chess_model.py"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "chess_win_model_bundle.pkl"


with DAG(
    dag_id="monthly_chess_model_retraining",
    description="Retrain the chess win model once per month and save the latest artifact.",
    start_date=datetime(2026, 1, 21),
    schedule="0 3 21 * *",
    catchup=False,
    tags=["ml", "monthly", "chess"],
) as dag:
    run_monthly_retraining = BashOperator(
        task_id="run_monthly_retraining",
        cwd=str(PROJECT_ROOT),
        bash_command=(
            f"python {TRAINING_SCRIPT} "
            "--run-date {{ ds }} "
            f"--artifact-path {ARTIFACT_PATH}"
        ),
    )

    validate_artifact_exists = BashOperator(
        task_id="validate_artifact_exists",
        cwd=str(PROJECT_ROOT),
        bash_command=f"test -f {ARTIFACT_PATH}",
    )

    run_monthly_retraining >> validate_artifact_exists

This folder contains Apache Airflow DAG definitions for the project.

Current DAGs:

- `monthly_ml_retrain_dag.py`: monthly chess model retraining pipeline

`monthly_ml_retrain_dag.py` is the operational DAG for the ML retraining path. It:

1. Runs `ml_pipeline/train_chess_model.py` with the Airflow logical date (`{{ ds }}`).

Implementation notes:

- The DAG resolves the repository root relative to the DAG file, so it does not depend on one developer's absolute path.
- The DAG uses `sys.executable`, which means it will run the retraining script with the same Python interpreter that started Airflow.
- The expected artifact path is `artifacts/chess_win_model_bundle.pkl`.
- The training script also creates a PDF report under `artifacts/reports/`.

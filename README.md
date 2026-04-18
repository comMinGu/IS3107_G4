# IS3107_G4

IS3107 Project Group 4 data engineering project.

## Current ML pipeline

The operational ML retraining flow lives outside the main ETL pipeline for now.

- Training entry point: `ml_pipeline/train_chess_model.py`
- Airflow DAG: `dags/monthly_ml_retrain_dag.py`
- DAG id: `monthly_chess_model_retraining`
- Artifact output: `artifacts/chess_win_model_bundle.pkl`
- Baseline notebook reference: `ml_model.ipynb`

Current production feature set:

- `eval_cp_clipped`
- `elo_diff`
- `time_left_seconds`
- `ply`
- `side_to_move`

The ablation notebook is not the operational source of truth.

## Team setup notes

The monthly retraining DAG assumes the Airflow runtime and the project code run in the same Python environment. The DAG now uses the interpreter that loaded Airflow, so teammates should install Airflow and the ML dependencies into that environment before enabling the DAG.

Minimum Python packages used by the ML retraining path:

- `apache-airflow`
- `lightgbm`
- `matplotlib`
- `pandas`
- `numpy`
- `scikit-learn`
- `requests`
- `zstandard`
- `python-chess`
- `tqdm`

The retraining script may also need a local `stockfish` binary when missing engine evaluations are filled. If Stockfish is not on `PATH`, set `STOCKFISH_PATH`.

## Local Airflow hookup

Example symlink from a standalone Airflow home:

```bash
ln -s /Users/mingukang/Desktop/IS3107_Git/IS3107_G4/dags/monthly_ml_retrain_dag.py ~/airflow/dags/monthly_ml_retrain_dag.py
```

Once Airflow is running, trigger `monthly_chess_model_retraining` from the UI or CLI.

The DAG runs monthly at `03:00` on day `21` and:

1. Retrain the model bundle for the provided Airflow run date.
2. Generate a basic PDF report with core model metrics and charts.

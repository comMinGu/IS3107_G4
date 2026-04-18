This folder contains the operational ML retraining code used by Airflow.

Current files:

- `train_chess_model.py`: monthly retraining entry point extracted from `ml_model.ipynb`

Operational assumptions:

- `ml_model.ipynb` is the baseline source of truth for the current production logic.
- `ml_model_elo_ablation.ipynb` is not part of the monthly retraining path.
- The current production features are `eval_cp_clipped`, `elo_diff`, `time_left_seconds`, `ply`, and `side_to_move`.

`train_chess_model.py` covers:

1. Run-date-based Lichess broadcast URL generation.
2. Filtered broadcast PGN loading.
3. Feature engineering.
4. Train/validation/test split by game.
5. LightGBM model training and evaluation.
6. PDF report generation with basic model metrics and plots.
7. Artifact save plus reload validation.

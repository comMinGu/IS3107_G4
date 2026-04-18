"""Monthly chess win-model retraining entry point.

This script is the operational counterpart to `ml_model.ipynb`.
It keeps the notebook's current baseline approach, but packages the core
steps into a Python entry point that can later be called by Airflow.
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
import requests
import zstandard as zstd
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    import lightgbm as lgb
except ImportError as exc:
    raise ImportError("LightGBM is required for this script. Run `pip install lightgbm`.") from exc


TARGET_FILTERED_GAMES = 2000
MIN_BLACK_WIN_GAMES = 200
MAX_WHITE_WIN_GAMES = 900
MAX_DRAW_GAMES = 900
MIN_ELO = 2500
MAX_GAME_AGE_DAYS = 730
MIN_TIME_CONTROL_SECONDS = 15 * 60

FILL_MISSING_EVALS = True
STOCKFISH_PATH = None
ENGINE_DEPTH = 1
ENGINE_TIME_LIMIT = 0.01
ENGINE_MAX_ROWS = None

FEATURE_COLS = [
    "eval_cp_clipped",
    "elo_diff",
    "time_left_seconds",
    "ply",
    "side_to_move",
]
LABEL_ORDER = ["white_win", "draw", "black_win"]

DEFAULT_STOCKFISH_CANDIDATES = [
    STOCKFISH_PATH,
    os.environ.get("STOCKFISH_PATH"),
    shutil.which("stockfish"),
    "/opt/homebrew/bin/stockfish",
    "/usr/local/bin/stockfish",
]


@dataclass
class TrainingArtifacts:
    model: lgb.LGBMClassifier
    model_df: pd.DataFrame
    feature_cols: List[str]
    label_order: List[str]
    label_to_id: Dict[str, int]
    metrics: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly chess model retraining pipeline.")
    parser.add_argument(
        "--run-date",
        default=None,
        help="Run date in YYYY-MM-DD format. Defaults to today if omitted.",
    )
    parser.add_argument(
        "--artifact-path",
        default="artifacts/chess_win_model_bundle.pkl",
        help="Output path for the serialized model bundle.",
    )
    return parser.parse_args()


def parse_run_date(run_date_str: Optional[str]) -> date:
    if run_date_str is None:
        return date.today()
    return datetime.strptime(run_date_str, "%Y-%m-%d").date()


def shift_month(year: int, month: int, offset: int) -> tuple[int, int]:
    total_months = year * 12 + (month - 1) + offset
    shifted_year = total_months // 12
    shifted_month = total_months % 12 + 1
    return shifted_year, shifted_month


def build_broadcast_urls(
    run_date: Optional[date] = None,
    months_to_load: int = 8,
    cutoff_day: int = 20,
) -> List[str]:
    if run_date is None:
        run_date = date.today()

    start_offset = -1 if run_date.day >= cutoff_day else -2
    urls = []

    for i in range(months_to_load):
        year, month = shift_month(run_date.year, run_date.month, start_offset - i)
        urls.append(f"https://database.lichess.org/broadcast/lichess_db_broadcast_{year}-{month:02d}.pgn.zst")

    return urls


def stream_text_from_zst_url(url: str) -> io.TextIOBase:
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(response.raw)
    return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")


def parse_clock_to_seconds(clock_str: str) -> Optional[float]:
    parts = clock_str.split(":")
    try:
        numeric_parts = [float(value) for value in parts]
    except ValueError:
        return None

    if len(numeric_parts) == 3:
        hours, minutes, seconds = numeric_parts
        return hours * 3600 + minutes * 60 + seconds
    if len(numeric_parts) == 2:
        minutes, seconds = numeric_parts
        return minutes * 60 + seconds
    if len(numeric_parts) == 1:
        return numeric_parts[0]
    return None


def parse_lichess_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    if not date_str or date_str == "????.??.??":
        return None
    try:
        return pd.to_datetime(date_str, format="%Y.%m.%d", errors="raise").normalize()
    except (TypeError, ValueError):
        return None


def parse_time_control_base_seconds(time_control: Optional[str]) -> Optional[int]:
    if not time_control or time_control in {"-", "?"}:
        return None

    base_str = time_control.split("+", 1)[0]
    if not base_str.isdigit():
        return None
    return int(base_str)


def game_matches_filters(
    headers: chess.pgn.Headers,
    min_elo: int,
    min_time_control_seconds: int,
    earliest_date: pd.Timestamp,
    latest_date: pd.Timestamp,
) -> bool:
    white_elo_raw = headers.get("WhiteElo")
    black_elo_raw = headers.get("BlackElo")
    white_elo = int(white_elo_raw) if white_elo_raw and white_elo_raw.isdigit() else None
    black_elo = int(black_elo_raw) if black_elo_raw and black_elo_raw.isdigit() else None
    has_super_gm = (white_elo is not None and white_elo >= min_elo) or (black_elo is not None and black_elo >= min_elo)

    game_date = parse_lichess_date(headers.get("Date"))
    if game_date is None:
        return False

    base_seconds = parse_time_control_base_seconds(headers.get("TimeControl"))
    if base_seconds is None:
        return False

    is_recent_enough = earliest_date <= game_date <= latest_date
    is_long_time_control = base_seconds >= min_time_control_seconds
    return has_super_gm and is_recent_enough and is_long_time_control


def extract_eval_and_clock(comment: str) -> tuple[Optional[int], Optional[int], Optional[float]]:
    eval_cp = None
    mate_in = None
    clk_seconds = None

    eval_match = re.search(r"\[%eval\s+([^\]]+)\]", comment)
    clk_match = re.search(r"\[%clk\s+([^\]]+)\]", comment)

    if eval_match:
        raw_eval = eval_match.group(1).strip()
        if raw_eval.startswith("#"):
            try:
                mate_in = int(raw_eval[1:])
            except ValueError:
                mate_in = None
        else:
            try:
                eval_cp = int(round(float(raw_eval) * 100))
            except ValueError:
                eval_cp = None

    if clk_match:
        clk_seconds = parse_clock_to_seconds(clk_match.group(1).strip())

    return eval_cp, mate_in, clk_seconds


def result_to_label(result: str) -> str:
    if result == "1-0":
        return "white_win"
    if result == "0-1":
        return "black_win"
    if result == "1/2-1/2":
        return "draw"
    return "unknown"


def find_stockfish_binary(stockfish_path: Optional[str] = None) -> str:
    candidates = [stockfish_path, *DEFAULT_STOCKFISH_CANDIDATES]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Stockfish binary not found. Install Stockfish locally and set STOCKFISH_PATH if needed."
    )


def score_fen_with_stockfish(
    engine: chess.engine.SimpleEngine,
    fen: str,
    depth: int = 1,
    time_limit: Optional[float] = 0.01,
) -> tuple[Optional[int], Optional[int]]:
    board = chess.Board(fen)
    limit = chess.engine.Limit(depth=depth) if time_limit is None else chess.engine.Limit(depth=depth, time=time_limit)
    info = engine.analyse(board, limit)
    score = info["score"].white()

    if score.is_mate():
        return None, score.mate()
    return score.score(mate_score=100000), None


def fill_missing_evals_with_stockfish(
    df: pd.DataFrame,
    stockfish_path: Optional[str] = None,
    depth: int = 1,
    time_limit: Optional[float] = 0.01,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    missing_mask = df["eval_cp"].isna() & df["mate_in"].isna()
    missing_idx = list(df.index[missing_mask])

    if max_rows is not None:
        missing_idx = missing_idx[:max_rows]

    if not missing_idx:
        print("No missing evals to fill.")
        return df

    engine_path = find_stockfish_binary(stockfish_path)
    print(f"Using Stockfish at: {engine_path}")
    print(f"Rows queued for engine eval: {len(missing_idx)}")

    filled_df = df.copy()
    fen_cache: Dict[str, tuple[Optional[int], Optional[int]]] = {}

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        for idx in tqdm(missing_idx, desc="Computing missing evals", unit="row"):
            fen = filled_df.at[idx, "fen_after"]
            if fen not in fen_cache:
                fen_cache[fen] = score_fen_with_stockfish(
                    engine,
                    fen,
                    depth=depth,
                    time_limit=time_limit,
                )
            eval_cp, mate_in = fen_cache[fen]
            filled_df.at[idx, "eval_cp"] = eval_cp
            filled_df.at[idx, "mate_in"] = mate_in

    return filled_df


def load_filtered_broadcast_games(
    urls: List[str],
    reference_date: date,
    target_games: int = TARGET_FILTERED_GAMES,
    min_black_win_games: Optional[int] = MIN_BLACK_WIN_GAMES,
    max_white_win_games: Optional[int] = MAX_WHITE_WIN_GAMES,
    max_draw_games: Optional[int] = MAX_DRAW_GAMES,
    min_elo: int = MIN_ELO,
    max_game_age_days: int = MAX_GAME_AGE_DAYS,
    min_time_control_seconds: int = MIN_TIME_CONTROL_SECONDS,
) -> pd.DataFrame:
    rows: List[Dict] = []
    today = pd.Timestamp(reference_date).normalize()
    earliest_date = today - pd.Timedelta(days=max_game_age_days)
    matched_games = 0
    accepted_game_counts = {"white_win": 0, "draw": 0, "black_win": 0}

    with tqdm(total=target_games, desc="Parsing filtered games", unit="game") as progress:
        for url in urls:
            if matched_games >= target_games and (
                min_black_win_games is None or accepted_game_counts["black_win"] >= min_black_win_games
            ):
                break

            print(f"Reading from: {url}")
            text_stream = stream_text_from_zst_url(url)

            while True:
                if matched_games >= target_games and (
                    min_black_win_games is None or accepted_game_counts["black_win"] >= min_black_win_games
                ):
                    break

                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

                headers = game.headers
                if not game_matches_filters(
                    headers,
                    min_elo=min_elo,
                    min_time_control_seconds=min_time_control_seconds,
                    earliest_date=earliest_date,
                    latest_date=today,
                ):
                    continue

                board = game.board()
                result = headers.get("Result", "")
                game_label = result_to_label(result)

                if game_label == "unknown":
                    continue

                if game_label == "white_win" and max_white_win_games is not None and accepted_game_counts["white_win"] >= max_white_win_games:
                    continue
                if game_label == "draw" and max_draw_games is not None and accepted_game_counts["draw"] >= max_draw_games:
                    continue

                white_elo = headers.get("WhiteElo")
                black_elo = headers.get("BlackElo")
                site = headers.get("Site", "")
                game_id = site.rstrip("/").split("/")[-1] if site else None

                node = game
                ply = 0

                while node.variations:
                    next_node = node.variation(0)
                    move = next_node.move
                    ply += 1

                    fen_before = board.fen()
                    side_to_move = "white" if board.turn == chess.WHITE else "black"
                    san = board.san(move)
                    uci = move.uci()

                    board.push(move)
                    fen_after = board.fen()

                    comment = next_node.comment or ""
                    eval_cp, mate_in, clk_seconds = extract_eval_and_clock(comment)

                    rows.append(
                        {
                            "game_id": game_id,
                            "date": headers.get("Date"),
                            "white_player": headers.get("White"),
                            "black_player": headers.get("Black"),
                            "white_elo": int(white_elo) if white_elo and white_elo.isdigit() else None,
                            "black_elo": int(black_elo) if black_elo and black_elo.isdigit() else None,
                            "result": result,
                            "label": game_label,
                            "time_control": headers.get("TimeControl"),
                            "eco": headers.get("ECO"),
                            "opening": headers.get("Opening"),
                            "ply": ply,
                            "side_to_move": side_to_move,
                            "san": san,
                            "uci": uci,
                            "fen_before": fen_before,
                            "fen_after": fen_after,
                            "eval_cp": eval_cp,
                            "mate_in": mate_in,
                            "clock_seconds_after_move": clk_seconds,
                        }
                    )

                    node = next_node

                matched_games += 1
                accepted_game_counts[game_label] += 1
                progress.update(1)

    return pd.DataFrame(rows)


def print_import_summary(df: pd.DataFrame) -> None:
    game_outcome_counts = (
        df.groupby("game_id")["label"]
        .first()
        .value_counts()
        .reindex(LABEL_ORDER, fill_value=0)
    )
    game_outcome_proportions = (game_outcome_counts / game_outcome_counts.sum()).round(4)

    print("Rows:", len(df))
    print("Unique games:", df["game_id"].nunique())
    print("Estimated rows target:", TARGET_FILTERED_GAMES * 200)
    print("Game-level outcome counts:")
    print(game_outcome_counts)
    print("Game-level outcome proportions:")
    print(game_outcome_proportions)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()

    total_rows = len(model_df)
    eval_cp_missing_before = model_df["eval_cp"].isna().sum()
    mate_in_non_null_before = model_df["mate_in"].notna().sum()
    rows_with_any_eval_before = (model_df["eval_cp"].notna() | model_df["mate_in"].notna()).sum()
    rows_without_any_eval_before = total_rows - rows_with_any_eval_before

    if FILL_MISSING_EVALS:
        model_df = fill_missing_evals_with_stockfish(
            model_df,
            stockfish_path=STOCKFISH_PATH,
            depth=ENGINE_DEPTH,
            time_limit=ENGINE_TIME_LIMIT,
            max_rows=ENGINE_MAX_ROWS,
        )
    else:
        print("Skipping Stockfish fill.")

    eval_cp_missing_after = model_df["eval_cp"].isna().sum()
    mate_in_non_null_after = model_df["mate_in"].notna().sum()
    rows_with_any_eval_after = (model_df["eval_cp"].notna() | model_df["mate_in"].notna()).sum()
    rows_without_any_eval_after = total_rows - rows_with_any_eval_after
    filled_eval_rows = eval_cp_missing_before - eval_cp_missing_after

    print("=== eval_cp fill validation ===")
    print("Total rows:", total_rows)
    print("Rows without eval_cp and mate_in before fill:", rows_without_any_eval_before)
    print("Rows without eval_cp and mate_in after fill:", rows_without_any_eval_after)
    print("Rows without eval_cp before fill:", eval_cp_missing_before)
    print("Rows without eval_cp after fill:", eval_cp_missing_after)
    print("New eval_cp values filled:", filled_eval_rows)
    print("Rows with mate_in before fill:", mate_in_non_null_before)
    print("Rows with mate_in after fill:", mate_in_non_null_after)
    print("Rows with any eval signal before fill:", rows_with_any_eval_before)
    print("Rows with any eval signal after fill:", rows_with_any_eval_after)
    print("Fraction with any eval signal after fill:", rows_with_any_eval_after / total_rows if total_rows else 0)

    eval_series = model_df["eval_cp"].dropna()
    print("=== eval_cp stats after fill ===")
    print("min:", eval_series.min())
    print("max:", eval_series.max())
    print("positive:", (eval_series > 0).sum())
    print("negative:", (eval_series < 0).sum())
    print("zero:", (eval_series == 0).sum())

    model_df["elo_diff"] = model_df["white_elo"] - model_df["black_elo"]
    model_df["time_left_seconds"] = model_df["clock_seconds_after_move"].clip(lower=0)
    model_df["eval_cp_clipped"] = model_df["eval_cp"].clip(-1000, 1000)
    model_df["ply"] = model_df["ply"].astype(int)
    model_df["side_to_move"] = model_df["side_to_move"].astype("category")

    required_columns = [
        "game_id",
        "label",
        "eval_cp_clipped",
        "white_elo",
        "black_elo",
        "elo_diff",
        "time_left_seconds",
        "ply",
        "side_to_move",
    ]

    model_df = model_df.dropna(subset=required_columns).copy()
    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    model_df = model_df[model_df["label"].isin(label_to_id)].copy()
    model_df["target"] = model_df["label"].map(label_to_id).astype(int)

    print("Rows kept for modeling:", len(model_df))
    print("Unique games kept:", model_df["game_id"].nunique())
    print("Feature columns:", FEATURE_COLS)
    print(model_df[["label", "target"]].drop_duplicates().sort_values("target"))

    return model_df


def split_games(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    game_level_df = model_df.groupby("game_id", as_index=False).agg(game_label=("label", "first"))

    train_val_games, test_games = train_test_split(
        game_level_df,
        test_size=0.15,
        random_state=42,
        stratify=game_level_df["game_label"],
    )

    val_size_within_train_val = 0.15 / 0.85
    train_games, val_games = train_test_split(
        train_val_games,
        test_size=val_size_within_train_val,
        random_state=42,
        stratify=train_val_games["game_label"],
    )

    train_ids = set(train_games["game_id"])
    val_ids = set(val_games["game_id"])
    test_ids = set(test_games["game_id"])

    train_df = model_df[model_df["game_id"].isin(train_ids)].copy()
    val_df = model_df[model_df["game_id"].isin(val_ids)].copy()
    test_df = model_df[model_df["game_id"].isin(test_ids)].copy()

    print("Train rows:", len(train_df), "| games:", train_df["game_id"].nunique())
    print("Val rows:", len(val_df), "| games:", val_df["game_id"].nunique())
    print("Test rows:", len(test_df), "| games:", test_df["game_id"].nunique())

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    return train_df, val_df, test_df


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> TrainingArtifacts:
    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df["target"].copy()
    X_val = val_df[FEATURE_COLS].copy()
    y_val = val_df["target"].copy()
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df["target"].copy()

    categorical_features = ["side_to_move"]

    lgbm_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(LABEL_ORDER),
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    lgbm_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=25),
        ],
    )

    val_proba = lgbm_model.predict_proba(X_val)
    test_proba = lgbm_model.predict_proba(X_test)
    val_pred = val_proba.argmax(axis=1)
    test_pred = test_proba.argmax(axis=1)

    metrics = {
        "validation_log_loss": float(log_loss(y_val, val_proba, labels=[0, 1, 2])),
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_log_loss": float(log_loss(y_test, test_proba, labels=[0, 1, 2])),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
    }

    print("Validation log loss:", metrics["validation_log_loss"])
    print("Validation accuracy:", metrics["validation_accuracy"])
    print("Test log loss:", metrics["test_log_loss"])
    print("Test accuracy:", metrics["test_accuracy"])
    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, target_names=LABEL_ORDER))

    feature_importance_df = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": lgbm_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nFeature importance:")
    print(feature_importance_df.to_string(index=False))

    return TrainingArtifacts(
        model=lgbm_model,
        model_df=pd.concat([train_df, val_df, test_df]).sort_index(),
        feature_cols=FEATURE_COLS,
        label_order=LABEL_ORDER,
        label_to_id={label: idx for idx, label in enumerate(LABEL_ORDER)},
        metrics=metrics,
    )


def save_model_bundle(
    artifacts: TrainingArtifacts,
    artifact_path: str,
    run_date: date,
    broadcast_urls: List[str],
) -> None:
    artifact_file = Path(artifact_path)
    artifact_file.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": artifacts.model,
        "feature_cols": artifacts.feature_cols,
        "label_order": artifacts.label_order,
        "label_to_id": artifacts.label_to_id,
        "metrics": artifacts.metrics,
        "run_date": run_date.isoformat(),
        "broadcast_urls": broadcast_urls,
    }

    with artifact_file.open("wb") as file_obj:
        pickle.dump(model_bundle, file_obj)

    print("Saved model bundle to:", artifact_file.as_posix())

    with artifact_file.open("rb") as file_obj:
        loaded_bundle = pickle.load(file_obj)

    loaded_model = loaded_bundle["model"]
    loaded_feature_cols = loaded_bundle["feature_cols"]
    loaded_label_order = loaded_bundle["label_order"]

    sample_input = artifacts.model_df[loaded_feature_cols].head(5).copy()
    original_proba = artifacts.model.predict_proba(sample_input)
    loaded_proba = loaded_model.predict_proba(sample_input)

    print("Loaded feature columns:", loaded_feature_cols)
    print("Loaded label order:", loaded_label_order)
    print("Predictions match after reload:", np.allclose(original_proba, loaded_proba))
    print(pd.DataFrame(loaded_proba, columns=loaded_label_order))


def main(run_date: Optional[str] = None, artifact_path: str = "artifacts/chess_win_model_bundle.pkl") -> None:
    parsed_run_date = parse_run_date(run_date)
    broadcast_urls = build_broadcast_urls(parsed_run_date)

    print("Monthly chess model retraining pipeline")
    print("Run date:", parsed_run_date.isoformat())
    print("Broadcast URLs:")
    for url in broadcast_urls:
        print("-", url)

    df = load_filtered_broadcast_games(
        urls=broadcast_urls,
        reference_date=parsed_run_date,
    )
    print_import_summary(df)

    model_df = engineer_features(df)
    train_df, val_df, test_df = split_games(model_df)
    training_artifacts = train_model(train_df, val_df, test_df)
    save_model_bundle(
        training_artifacts,
        artifact_path=artifact_path,
        run_date=parsed_run_date,
        broadcast_urls=broadcast_urls,
    )


if __name__ == "__main__":
    args = parse_args()
    main(run_date=args.run_date, artifact_path=args.artifact_path)

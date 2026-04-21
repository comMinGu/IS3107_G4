from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.providers.google.cloud.hooks.gcs import GCSHook

log = logging.getLogger(__name__)

LICHESS_TV_CHANNELS_URL = "https://lichess.org/api/tv/channels"
LICHESS_GAME_STREAM_URL = "https://lichess.org/api/stream/game/{game_id}"
CLOUD_EVAL_URL = "https://lichess.org/api/cloud-eval"
ML_ARTIFACT_PATH = os.environ.get("ML_ARTIFACT_PATH", "artifacts/chess_win_model_bundle.pkl")
ML_ARTIFACT_URI = Variable.get("ML_ARTIFACT_URI", default_var=ML_ARTIFACT_PATH)
GCP_CONN_ID = Variable.get("GCP_CONN_ID", default_var="google_cloud_default")
MODEL_BUNDLE_CACHE: dict[str, Any] | None = None

default_args = {
    "owner": "is3107",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def get_tv_channels() -> dict[str, Any]:
    req = Request(
        LICHESS_TV_CHANNELS_URL,
        headers={"Accept": "application/json", "User-Agent": "IS3107_G4-airflow/1.0 (education)"},
        method="GET",
    )
    with urlopen(req, timeout=60) as resp:
        payload = resp.read().decode("utf-8")
    channels = json.loads(payload)
    if not isinstance(channels, dict):
        raise TypeError(f"Expected dict from TV channels, got {type(channels)}")
    return channels


def get_classical_game_id(channels: dict[str, Any]) -> str:
    classical = channels.get("classical")
    game_id = classical.get("gameId") if isinstance(classical, dict) else None
    if not game_id:
        raise ValueError("No gameId found for channels.classical")
    return game_id


def get_cloud_eval_score(fen: str | None) -> int | float | None:
    if not fen:
        return None
    req = Request(
        f"{CLOUD_EVAL_URL}?{urlencode({'fen': fen})}",
        headers={"Accept": "application/json", "User-Agent": "IS3107_G4-airflow/1.0 (education)"},
        method="GET",
    )
    try:
        with urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8")
        data = json.loads(payload)
    except (HTTPError, URLError):
        return None

    pvs = data.get("pvs") or []
    if not pvs:
        return None
    best = pvs[0]
    if "cp" in best:
        return best["cp"]
    if "mate" in best:
        return best["mate"]
    return None


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    bucket_and_object = uri[5:]
    if "/" not in bucket_and_object:
        raise ValueError(f"GCS URI must include object path, got: {uri}")
    bucket, object_name = bucket_and_object.split("/", 1)
    return bucket, object_name


def resolve_local_artifact_path(artifact_uri: str) -> Path:
    if artifact_uri.startswith("gs://"):
        bucket, object_name = _parse_gcs_uri(artifact_uri)
        local_path = Path("/tmp") / Path(object_name).name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        GCSHook(gcp_conn_id=GCP_CONN_ID).download(
            bucket_name=bucket,
            object_name=object_name,
            filename=local_path.as_posix(),
        )
        log.info("Downloaded ML artifact from %s to %s", artifact_uri, local_path.as_posix())
        return local_path

    artifact_file = Path(artifact_uri)
    if not artifact_file.exists():
        raise FileNotFoundError(
            f"ML artifact not found at '{artifact_file.as_posix()}'. "
            "Set Airflow Variable ML_ARTIFACT_URI or env ML_ARTIFACT_PATH."
        )
    return artifact_file


def load_model_bundle() -> dict[str, Any]:
    global MODEL_BUNDLE_CACHE
    if MODEL_BUNDLE_CACHE is not None:
        return MODEL_BUNDLE_CACHE

    artifact_file = resolve_local_artifact_path(ML_ARTIFACT_URI)
    with artifact_file.open("rb") as file_obj:
        bundle = pickle.load(file_obj)
    for required_key in ["model", "feature_cols", "label_order"]:
        if required_key not in bundle:
            raise KeyError(f"ML artifact missing required key: {required_key}")
    MODEL_BUNDLE_CACHE = bundle
    return bundle


def _safe_numeric(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_side_to_move_from_fen(fen: str) -> str:
    fields = fen.split()
    if len(fields) < 2:
        raise ValueError(f"Invalid FEN format: {fen}")
    if fields[1] == "w":
        return "white"
    if fields[1] == "b":
        return "black"
    raise ValueError(f"Unexpected active color in FEN: {fen}")


def build_model_input(game: dict[str, Any], move: dict[str, Any], move_number: int) -> dict[str, Any]:
    fen = move.get("fen")
    if not fen:
        raise ValueError("Missing fen in move event.")
    side_to_move = parse_side_to_move_from_fen(fen)
    mover_clock_raw = move.get("wc") if side_to_move == "white" else move.get("bc")
    time_left_seconds = max(_safe_numeric(mover_clock_raw) or 0.0, 0.0)
    cloud_eval_raw = _safe_numeric(get_cloud_eval_score(fen))
    eval_cp_clipped = max(min(cloud_eval_raw or 0.0, 1000.0), -1000.0)
    elo_diff = int(game["players"]["white"]["rating"]) - int(game["players"]["black"]["rating"])
    return {
        "eval_cp_clipped": eval_cp_clipped,
        "elo_diff": float(elo_diff),
        "time_left_seconds": float(time_left_seconds),
        "ply": int(move_number),
        "side_to_move": side_to_move,
    }


def invoke_ml_artifact(model_input: dict[str, Any]) -> dict[str, Any]:
    bundle = load_model_bundle()
    feature_cols: list[str] = bundle["feature_cols"]
    label_order: list[str] = bundle["label_order"]
    frame = pd.DataFrame([model_input], columns=feature_cols)
    probabilities = bundle["model"].predict_proba(frame)[0]
    return {
        "predicted_label": label_order[int(probabilities.argmax())],
        "probabilities": {label_order[idx]: float(probabilities[idx]) for idx in range(len(label_order))},
    }


@dag(
    dag_id="lichess_live_data_simple_composer",
    description="Composer-friendly Lichess stream -> ML inference from local/GCS artifact.",
    schedule="@continuous",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["lichess", "ml", "live", "composer"],
    default_args=default_args,
)
def live_data_composer_dag():
    @task(task_id="fetch_tv_channels")
    def fetch_tv_channels() -> dict[str, Any]:
        channels = get_tv_channels()
        log.info("Fetched tv channels keys=%s", list(channels.keys()))
        return channels

    @task(task_id="extract_classical_game_id")
    def extract_classical_game_id(channels: dict[str, Any]) -> str:
        game_id = get_classical_game_id(channels)
        log.info("Selected classical game_id=%s", game_id)
        return game_id

    @task(task_id="stream_and_infer")
    def stream_and_infer(game_id: str) -> dict[str, Any]:
        req = Request(
            LICHESS_GAME_STREAM_URL.format(game_id=game_id),
            headers={"Accept": "application/x-ndjson", "User-Agent": "IS3107_G4-airflow/1.0 (education)"},
            method="GET",
        )
        processed_moves = 0
        latest_turn = None
        line_num = 0
        with urlopen(req, timeout=60) as resp:
            game = None
            for raw_line in resp:
                line_num += 1
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                if game is None:
                    game = event
                    log.info("Game header loaded game_id=%s", game.get("id"))
                    continue
                if "fen" not in event or "lm" not in event:
                    continue
                move_number = processed_moves + 1
                turn = (move_number + 1) // 2
                model_input = build_model_input(game, event, move_number=move_number)
                probs = invoke_ml_artifact(model_input)["probabilities"]
                log.info(
                    "turn=%s side_to_move=%s ml=[%.6f, %.6f, %.6f]",
                    turn,
                    model_input["side_to_move"],
                    probs["white_win"],
                    probs["draw"],
                    probs["black_win"],
                )
                processed_moves += 1
                latest_turn = turn
        return {"game_id": game_id, "processed_moves": processed_moves, "latest_turn": latest_turn, "lines_seen": line_num}

    channels = fetch_tv_channels()
    game_id = extract_classical_game_id(channels)
    stream_and_infer(game_id)


dag = live_data_composer_dag()

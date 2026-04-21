from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pendulum
from airflow.decorators import dag, task

log = logging.getLogger(__name__)

LICHESS_TV_CHANNELS_URL = "https://lichess.org/api/tv/channels"
LICHESS_GAME_STREAM_URL = "https://lichess.org/api/stream/game/{game_id}"
CLOUD_EVAL_URL = "https://lichess.org/api/cloud-eval"

default_args = {
    "owner": "is3107",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def get_tv_channels() -> dict[str, Any]:
    req = Request(
        LICHESS_TV_CHANNELS_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "IS3107_G4-airflow/1.0 (education)",
        },
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

    query = urlencode({"fen": fen})
    req = Request(
        f"{CLOUD_EVAL_URL}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "IS3107_G4-airflow/1.0 (education)",
        },
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


def build_move_record(game: dict[str, Any], move: dict[str, Any], move_number: int) -> dict[str, Any]:
    side = "w" if move_number % 2 == 1 else "b"
    turn = (move_number + 1) // 2
    return {
        "game_id": game["id"],
        "channel": "classical",
        "rating_white": game["players"]["white"]["rating"],
        "rating_black": game["players"]["black"]["rating"],
        "move_number": move_number,
        "turn": turn,
        "last_move": move.get("lm"),
        "last_move_color": side,
        "time_white": move.get("wc"),
        "time_black": move.get("bc"),
        "fen": move.get("fen"),
        "cloud_eval_score": get_cloud_eval_score(move.get("fen")),
    }


def invoke_ml_artifact(record: dict[str, Any]) -> dict[str, Any]:
    """
    Placeholder for your ML artifact call.
    Replace this implementation with your real inference call.
    """
    # Keep simple shape so downstream logging has stable fields.
    return {
        "value_1": 0.12,
        "value_2": 0.73,
        "value_3": 0.15,
    }


@dag(
    dag_id="lichess_live_data_simple",
    description="Simple Lichess classical stream -> record -> ML inference -> logs",
    schedule="@continuous",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["lichess", "ml", "live", "simple"],
    default_args=default_args,
)
def live_data_simple_dag():
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
            headers={
                "Accept": "application/x-ndjson",
                "User-Agent": "IS3107_G4-airflow/1.0 (education)",
            },
            method="GET",
        )

        processed_moves = 0
        latest_turn = None
        line_num = 0

        try:
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

                    if "fen" not in event:
                        continue

                    move_number = processed_moves + 1
                    record = build_move_record(game, event, move_number=move_number)
                    ml_result = invoke_ml_artifact(record)
                    log.info(
                        "turn=%s move=%s side=%s ml=%s",
                        record.get("turn"),
                        record.get("last_move"),
                        record.get("last_move_color"),
                        ml_result,
                    )
                    processed_moves += 1
                    latest_turn = record.get("turn")
        except HTTPError as exc:
            log.error("Stream HTTPError game_id=%s code=%s", game_id, exc.code)
            raise
        except URLError as exc:
            log.error("Stream URLError game_id=%s reason=%s", game_id, exc.reason)
            raise

        return {
            "game_id": game_id,
            "processed_moves": processed_moves,
            "latest_turn": latest_turn,
            "lines_seen": line_num,
        }

    channels = fetch_tv_channels()
    game_id = extract_classical_game_id(channels)
    stream_and_infer(game_id)


dag = live_data_simple_dag()

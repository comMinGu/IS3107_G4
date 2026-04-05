from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any
from urllib.request import Request, urlopen

import pendulum
from airflow.decorators import dag, task
from airflow.providers.redis.hooks.redis import RedisHook

log = logging.getLogger(__name__)

LICHESS_TV_CHANNELS_URL = "https://lichess.org/api/tv/channels"
LICHESS_GAME_STREAM_URL = "https://lichess.org/api/stream/game/{game_id}"
CLOUD_EVAL_URL = "https://lichess.org/api/cloud-eval?fen={fen}"
REDIS_CONN_ID = "redis_default"
CURRENT_GAME_KEY = "lichess:classical:current_game"

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
        data = json.loads(resp.read().decode("utf-8"))

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object from TV channels, got {type(data)}")
    return data


def get_classical_game_id(channels: dict[str, Any]) -> str:
    game_id = channels.get("classical", {}).get("gameId")
    if not game_id:
        raise ValueError("No gameId found for classical channel")
    return game_id


def get_cloud_eval_score(fen: str) -> int | float | None:
    req = Request(
        CLOUD_EVAL_URL.format(fen=fen),
        headers={
            "Accept": "application/json",
            "User-Agent": "IS3107_G4-airflow/1.0 (education)",
        },
        method="GET",
    )
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    pvs = data.get("pvs") or []
    if not pvs:
        return None

    best = pvs[0]
    if "cp" in best:
        return best["cp"]
    if "mate" in best:
        return best["mate"]
    return None


def build_move_record(game: dict[str, Any], move: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_id": game["id"],
        "channel": "classical",
        "rating_white": game["players"]["white"]["rating"],
        "rating_black": game["players"]["black"]["rating"],
        "turn": move.get("turns"),
        "last_move": move.get("lm"),
        "last_move_color": "white" if move.get("turns", 0) % 2 == 1 else "black",
        "eval": get_cloud_eval_score(move["fen"]),
        "time_white": move.get("wc"),
        "time_black": move.get("bc"),
        "fen": move.get("fen"),
    }


def run_ml_model(record: dict[str, Any]) -> dict[str, Any]:
    result = {
        "value_1": 0.12,
        "value_2": 0.73,
        "value_3": 0.15,
    }
    if not isinstance(result, dict) or len(result) != 3:
        raise ValueError("ML model must return a dict of exactly three values")
    return result


def get_redis():
    return RedisHook(redis_conn_id=REDIS_CONN_ID).get_conn()


def set_current_game_pointer(game_id: str, stream_key: str) -> None:
    redis_client = get_redis()
    payload = {
        "game_id": game_id,
        "stream_key": stream_key,
        "channel": "classical",
    }
    redis_client.set(CURRENT_GAME_KEY, json.dumps(payload))


def publish_to_redis_stream(
    stream_key: str,
    record: dict[str, Any],
    ml_result: dict[str, Any],
) -> str:
    redis_client = get_redis()
    entry_id = redis_client.xadd(
        stream_key,
        {
            "game_id": str(record["game_id"]),
            "turn": str(record["turn"]),
            "last_move": str(record["last_move"]),
            "record_json": json.dumps(record),
            "ml_result_json": json.dumps(ml_result),
        },
        maxlen=5000,
        approximate=True,
    )

    latest_payload = {
        "record": record,
        "ml_result": ml_result,
    }
    redis_client.set(f"{stream_key}:latest", json.dumps(latest_payload))

    if isinstance(entry_id, bytes):
        entry_id = entry_id.decode("utf-8")
    return entry_id


@dag(
    dag_id="lichess_live_data",
    description="Continuously process current Lichess classical TV game into Redis",
    schedule="@continuous",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["lichess", "redis", "ml", "live"],
    default_args=default_args,
)
def lichess_live_data_dag():
    @task(task_id="fetch_tv_channels")
    def fetch_tv_channels() -> dict[str, Any]:
        return get_tv_channels()

    @task(task_id="extract_classical_game_id")
    def extract_classical_game_id(channels: dict[str, Any]) -> str:
        return get_classical_game_id(channels)

    @task(task_id="stream_game_infer_and_publish")
    def stream_game_infer_and_publish(game_id: str) -> dict[str, Any]:
        req = Request(
            LICHESS_GAME_STREAM_URL.format(game_id=game_id),
            headers={
                "Accept": "application/x-ndjson",
                "User-Agent": "IS3107_G4-airflow/1.0 (education)",
            },
            method="GET",
        )

        stream_key = f"lichess:game:{game_id}:live"
        set_current_game_pointer(game_id=game_id, stream_key=stream_key)

        processed_moves = 0
        latest_turn = None
        latest_entry_id = None

        with urlopen(req, timeout=60) as resp:
            game = None

            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                event = json.loads(line)

                if game is None:
                    game = event
                    continue

                if "fen" not in event:
                    continue

                record = build_move_record(game, event)
                ml_result = run_ml_model(record)

                latest_entry_id = publish_to_redis_stream(
                    stream_key=stream_key,
                    record=record,
                    ml_result=ml_result,
                )

                processed_moves += 1
                latest_turn = record["turn"]

                log.info(
                    "Published turn=%s game_id=%s entry_id=%s",
                    latest_turn,
                    game_id,
                    latest_entry_id,
                )

        log.info("Game stream ended for game_id=%s", game_id)

        return {
            "game_id": game_id,
            "stream_key": stream_key,
            "processed_moves": processed_moves,
            "latest_turn": latest_turn,
            "latest_entry_id": latest_entry_id,
            "status": "game_finished",
        }

    channels = fetch_tv_channels()
    classical_game_id = extract_classical_game_id(channels)
    stream_game_infer_and_publish(classical_game_id)


dag = lichess_live_data_dag()
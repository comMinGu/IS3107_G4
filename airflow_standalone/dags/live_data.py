from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import pendulum
from airflow.decorators import dag, task
from google.auth import default as google_auth_default
from google.auth.transport.requests import AuthorizedSession

log = logging.getLogger(__name__)

LICHESS_TV_CHANNELS_URL = "https://lichess.org/api/tv/channels"
LICHESS_GAME_STREAM_URL = "https://lichess.org/api/stream/game/{game_id}"
CLOUD_EVAL_URL = "https://lichess.org/api/cloud-eval?fen={fen}"

# Same values as the web client (firebase-config). Firestore writes use Cloud Composer / ADC.
FIREBASE_CONFIG: dict[str, str] = {
    "apiKey": "AIzaSyAf0R2DOaL1K4mUBe8LnUpniSD-2Q96pao",
    "authDomain": "chess-win-predictor.firebaseapp.com",
    "projectId": "chess-win-predictor",
    "storageBucket": "chess-win-predictor.firebasestorage.app",
    "messagingSenderId": "708542432330",
    "appId": "1:708542432330:web:1736c9c2146332b9de92b3",
    "measurementId": "G-0XK6XQLCPH",
}

PROJECT_ID = FIREBASE_CONFIG["projectId"]
_FIRESTORE_SCOPES = ("https://www.googleapis.com/auth/datastore",)

PREDICTIONS_COLLECTION = "predictions"
CURRENT_CLASSICAL_DOC_ID = "_current_classical"

_authorized_session: AuthorizedSession | None = None


def _firestore_documents_root() -> str:
    return (
        f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/databases/(default)/documents"
    )


def _authorized_firestore_session() -> AuthorizedSession:
    global _authorized_session
    if _authorized_session is None:
        creds, _ = google_auth_default(scopes=_FIRESTORE_SCOPES)
        _authorized_session = AuthorizedSession(creds)
    return _authorized_session


def _timestamp_value() -> dict[str, Any]:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return {"timestampValue": ts}


def _to_firestore_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, dict):
        return {
            "mapValue": {
                "fields": {k: _to_firestore_value(value[k]) for k in value},
            },
        }
    if isinstance(value, list):
        return {"arrayValue": {"values": [_to_firestore_value(v) for v in value]}}
    return {"stringValue": str(value)}


def _patch_document(doc_rel_path: str, fields: dict[str, Any]) -> None:
    field_paths = list(fields.keys())
    mask = "&".join(f"updateMask.fieldPaths={quote(fp, safe='')}" for fp in field_paths)
    url = f"{_firestore_documents_root()}/{quote(doc_rel_path, safe='')}?{mask}&allowMissing=true"
    body = {"fields": {k: _to_firestore_value(fields[k]) for k in fields}}
    resp = _authorized_firestore_session().patch(url, json=body, timeout=60)
    resp.raise_for_status()


def _create_turn_document(game_id: str, turn_doc_id: str, fields: dict[str, Any]) -> str:
    parent_enc = quote(f"{PREDICTIONS_COLLECTION}/{game_id}", safe="")
    url = (
        f"{_firestore_documents_root()}/{parent_enc}/turns"
        f"?documentId={quote(turn_doc_id, safe='')}"
    )
    body = {"fields": {k: _to_firestore_value(fields[k]) for k in fields}}
    resp = _authorized_firestore_session().post(url, json=body, timeout=60)
    resp.raise_for_status()
    return turn_doc_id


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


def set_current_game_pointer(game_id: str, stream_key: str) -> None:
    now = _timestamp_value()
    _patch_document(
        f"{PREDICTIONS_COLLECTION}/{CURRENT_CLASSICAL_DOC_ID}",
        {
            "game_id": game_id,
            "stream_key": stream_key,
            "channel": "classical",
            "updated_at": now,
        },
    )
    _patch_document(
        f"{PREDICTIONS_COLLECTION}/{game_id}",
        {
            "game_id": game_id,
            "stream_key": stream_key,
            "channel": "classical",
            "status": "live",
            "created_at": now,
            "updated_at": now,
        },
    )


def publish_prediction_to_firestore(
    game_id: str,
    stream_key: str,
    record: dict[str, Any],
    ml_result: dict[str, Any],
) -> str:
    now = _timestamp_value()
    _patch_document(
        f"{PREDICTIONS_COLLECTION}/{game_id}",
        {
            "game_id": game_id,
            "stream_key": stream_key,
            "channel": "classical",
            "status": "live",
            "turn": record.get("turn"),
            "last_move": record.get("last_move"),
            "record": record,
            "ml_result": ml_result,
            "updated_at": now,
        },
    )
    turn_doc_id = str(uuid.uuid4())
    _create_turn_document(
        game_id,
        turn_doc_id,
        {
            "game_id": str(record["game_id"]),
            "turn": record.get("turn"),
            "last_move": record.get("last_move"),
            "record": record,
            "ml_result": ml_result,
            "created_at": now,
        },
    )
    return turn_doc_id


@dag(
    dag_id="lichess_live_data",
    description="Continuously process current Lichess classical TV game into Firestore",
    schedule="@continuous",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["lichess", "firestore", "ml", "live"],
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

                latest_entry_id = publish_prediction_to_firestore(
                    game_id=game_id,
                    stream_key=stream_key,
                    record=record,
                    ml_result=ml_result,
                )

                processed_moves += 1
                latest_turn = record["turn"]

                log.info(
                    "Published turn=%s game_id=%s turn_doc_id=%s",
                    latest_turn,
                    game_id,
                    latest_entry_id,
                )

        log.info("Game stream ended for game_id=%s", game_id)
        _patch_document(
            f"{PREDICTIONS_COLLECTION}/{game_id}",
            {"status": "finished", "updated_at": _timestamp_value()},
        )

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
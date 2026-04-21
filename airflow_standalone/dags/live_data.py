from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
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

# Same values as the web client (firebase-config). Firestore writes use ADC / Composer SA.
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

# Global pointer so the website can find the active classical game / Firestore collection name.
META_COLLECTION = "predictions"
CURRENT_CLASSICAL_DOC_ID = "_current_classical"
# Summary row inside each per-game collection (collection id == Lichess game_id).
GAME_SESSION_DOC_ID = "game_session"

_authorized_session: AuthorizedSession | None = None
_firebase_initialized: bool = False


def _firestore_documents_root() -> str:
    return (
        f"https://firestore.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/databases/(default)/documents"
    )


def initialize_firebase_connection() -> None:
    """
    Load credentials and build an authorized Firestore REST session once per process.
    Firestore creates a collection when the first document is written under that path.
    """
    global _authorized_session, _firebase_initialized

    if _firebase_initialized and _authorized_session is not None:
        log.debug("Firestore already initialized; skipping re-init")
        return

    log.info(
        "Initializing Firebase/Firestore connection project_id=%s auth_domain=%s",
        PROJECT_ID,
        FIREBASE_CONFIG.get("authDomain"),
    )
    try:
        creds, project = google_auth_default(scopes=_FIRESTORE_SCOPES)
        cred_type = type(creds).__name__
        sa_email = getattr(creds, "service_account_email", None)
        log.info(
            "google.auth.default OK cred_type=%s adc_project=%s service_account_email=%s",
            cred_type,
            project,
            sa_email or "(none)",
        )
        _authorized_session = AuthorizedSession(creds)
        _firebase_initialized = True
        log.info("Firestore REST AuthorizedSession ready base=%s", _firestore_documents_root())
    except Exception:
        _authorized_session = None
        _firebase_initialized = False
        log.exception("initialize_firebase_connection failed during credential or session setup")
        raise


def _authorized_firestore_session() -> AuthorizedSession:
    if _authorized_session is None or not _firebase_initialized:
        initialize_firebase_connection()
    assert _authorized_session is not None
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


def _log_firestore_http_error(operation: str, url: str, exc: Exception) -> None:
    if hasattr(exc, "response") and exc.response is not None:
        resp = exc.response
        log.error(
            "%s HTTP error url=%s status=%s body=%s",
            operation,
            url,
            getattr(resp, "status_code", None),
            getattr(resp, "text", None),
        )
    else:
        log.error("%s failed url=%s err=%s", operation, url, exc)


def _patch_document(doc_rel_path: str, fields: dict[str, Any]) -> None:
    field_paths = list(fields.keys())
    mask = "&".join(f"updateMask.fieldPaths={quote(fp, safe='')}" for fp in field_paths)
    url = f"{_firestore_documents_root()}/{quote(doc_rel_path, safe='')}?{mask}&allowMissing=true"
    body = {"fields": {k: _to_firestore_value(fields[k]) for k in fields}}
    log.debug("Firestore PATCH doc=%s fields=%s", doc_rel_path, list(fields.keys()))
    try:
        resp = _authorized_firestore_session().patch(url, json=body, timeout=120)
        resp.raise_for_status()
        log.info("Firestore PATCH OK doc=%s", doc_rel_path)
    except Exception as exc:
        _log_firestore_http_error("PATCH", url, exc)
        raise


def move_document_id(record: dict[str, Any]) -> str:
    """Document id: halfmove number + side to move after the move (White / Black)."""
    turn = record.get("turn")
    color_raw = (record.get("last_move_color") or "unknown").strip().lower()
    if color_raw == "white":
        side = "White"
    elif color_raw == "black":
        side = "Black"
    else:
        side = "Unknown"
    if turn is None:
        return f"unknown_{side}"
    return f"{turn}_{side}"


default_args = {
    "owner": "is3107",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def get_tv_channels() -> dict[str, Any]:
    log.info("Fetching Lichess TV channels from %s", LICHESS_TV_CHANNELS_URL)
    req = Request(
        LICHESS_TV_CHANNELS_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": "IS3107_G4-airflow/1.0 (education)",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            log.info("TV channels HTTP status=%s bytes=%s", getattr(resp, "status", "?"), len(raw))
            data = json.loads(raw)
    except HTTPError as e:
        log.error("TV channels HTTPError code=%s reason=%s", e.code, e.reason)
        raise
    except URLError as e:
        log.error("TV channels URLError reason=%s", e.reason)
        raise

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object from TV channels, got {type(data)}")
    log.info("TV channels parsed top-level keys=%s", list(data.keys()))
    return data


def get_classical_game_id(channels: dict[str, Any]) -> str:
    classical = channels.get("classical")
    log.info("Extracting classical game id classical_block=%s", type(classical).__name__)
    game_id = classical.get("gameId") if isinstance(classical, dict) else None
    if not game_id:
        log.error("No gameId in channels.classical payload=%s", classical)
        raise ValueError("No gameId found for classical channel")
    log.info("Classical game_id=%s", game_id)
    return game_id


def get_cloud_eval_score(fen: str) -> int | float | None:
    short_fen = fen.split()[0] if fen else ""
    log.debug("Cloud eval request fen_prefix=%s", short_fen[:40])
    req = Request(
        CLOUD_EVAL_URL.format(fen=fen),
        headers={
            "Accept": "application/json",
            "User-Agent": "IS3107_G4-airflow/1.0 (education)",
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        log.warning("Cloud eval HTTPError code=%s for fen_prefix=%s", e.code, short_fen[:20])
        return None
    except URLError as e:
        log.warning("Cloud eval URLError reason=%s", e.reason)
        return None

    pvs = data.get("pvs") or []
    if not pvs:
        log.debug("Cloud eval no pvs for fen_prefix=%s", short_fen[:20])
        return None

    best = pvs[0]
    if "cp" in best:
        log.debug("Cloud eval cp=%s", best["cp"])
        return best["cp"]
    if "mate" in best:
        log.debug("Cloud eval mate=%s", best["mate"])
        return best["mate"]
    return None


def build_move_record(game: dict[str, Any], move: dict[str, Any]) -> dict[str, Any]:
    log.debug("build_move_record move_keys=%s", list(move.keys()))
    rec = {
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
    log.info(
        "Built move record turn=%s lm=%s side=%s eval=%s",
        rec["turn"],
        rec["last_move"],
        rec["last_move_color"],
        rec["eval"],
    )
    return rec


def run_ml_model(record: dict[str, Any]) -> dict[str, Any]:
    log.info("run_ml_model input turn=%s", record.get("turn"))
    result = {
        "value_1": 0.12,
        "value_2": 0.73,
        "value_3": 0.15,
    }
    if not isinstance(result, dict) or len(result) != 3:
        raise ValueError("ML model must return a dict of exactly three values")
    log.info("run_ml_model output keys=%s", list(result.keys()))
    return result


def set_current_game_pointer(game_id: str, stream_key: str) -> None:
    initialize_firebase_connection()
    now = _timestamp_value()
    log.info(
        "Writing meta pointer collection=%s doc=%s game_id=%s",
        META_COLLECTION,
        CURRENT_CLASSICAL_DOC_ID,
        game_id,
    )
    _patch_document(
        f"{META_COLLECTION}/{CURRENT_CLASSICAL_DOC_ID}",
        {
            "game_id": game_id,
            "stream_key": stream_key,
            "channel": "classical",
            "firestore_collection": game_id,
            "updated_at": now,
        },
    )
    # Per-game collection is created implicitly; session doc holds live status.
    log.info("Upserting game session doc collection=%s doc=%s", game_id, GAME_SESSION_DOC_ID)
    _patch_document(
        f"{game_id}/{GAME_SESSION_DOC_ID}",
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
    if not _firebase_initialized:
        initialize_firebase_connection()
    now = _timestamp_value()
    move_doc = move_document_id(record)
    log.info(
        "Publishing move to Firestore collection=%s document=%s turn=%s",
        game_id,
        move_doc,
        record.get("turn"),
    )

    _patch_document(
        f"{game_id}/{GAME_SESSION_DOC_ID}",
        {
            "game_id": game_id,
            "stream_key": stream_key,
            "channel": "classical",
            "status": "live",
            "turn": record.get("turn"),
            "last_move": record.get("last_move"),
            "last_move_color": record.get("last_move_color"),
            "record": record,
            "ml_result": ml_result,
            "updated_at": now,
        },
    )

    _patch_document(
        f"{game_id}/{move_doc}",
        {
            "game_id": str(record["game_id"]),
            "turn": record.get("turn"),
            "last_move": record.get("last_move"),
            "last_move_color": record.get("last_move_color"),
            "record": record,
            "ml_result": ml_result,
            "created_at": now,
        },
    )
    log.info("Firestore move write OK collection=%s document=%s", game_id, move_doc)
    return move_doc


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
        log.info("Task fetch_tv_channels start")
        out = get_tv_channels()
        log.info("Task fetch_tv_channels done")
        return out

    @task(task_id="extract_classical_game_id")
    def extract_classical_game_id(channels: dict[str, Any]) -> str:
        log.info("Task extract_classical_game_id start")
        gid = get_classical_game_id(channels)
        log.info("Task extract_classical_game_id done game_id=%s", gid)
        return gid

    @task(task_id="stream_game_infer_and_publish")
    def stream_game_infer_and_publish(game_id: str) -> dict[str, Any]:
        log.info("Task stream_game_infer_and_publish start game_id=%s", game_id)
        stream_key = f"lichess:game:{game_id}:live"
        try:
            initialize_firebase_connection()
            set_current_game_pointer(game_id=game_id, stream_key=stream_key)
        except Exception:
            log.exception("Failed during Firebase init or set_current_game_pointer game_id=%s", game_id)
            raise

        req = Request(
            LICHESS_GAME_STREAM_URL.format(game_id=game_id),
            headers={
                "Accept": "application/x-ndjson",
                "User-Agent": "IS3107_G4-airflow/1.0 (education)",
            },
            method="GET",
        )
        log.info("Opening Lichess game stream url=%s", req.get_full_url())

        processed_moves = 0
        latest_turn = None
        latest_entry_id = None

        try:
            with urlopen(req, timeout=60) as resp:
                log.info("Game stream HTTP connected status=%s", getattr(resp, "status", "?"))
                game = None
                line_num = 0

                for raw_line in resp:
                    line_num += 1
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        log.warning("Line %s invalid JSON skipped prefix=%s", line_num, line[:80])
                        continue

                    if game is None:
                        game = event
                        log.info(
                            "Captured game header line=%s game_id=%s keys=%s",
                            line_num,
                            game.get("id"),
                            list(game.keys())[:20],
                        )
                        continue

                    if "fen" not in event:
                        log.debug("Line %s skipped (no fen) keys=%s", line_num, list(event.keys())[:15])
                        continue

                    log.info("Processing move line=%s turns=%s", line_num, event.get("turns"))
                    try:
                        record = build_move_record(game, event)
                        ml_result = run_ml_model(record)
                        latest_entry_id = publish_prediction_to_firestore(
                            game_id=game_id,
                            stream_key=stream_key,
                            record=record,
                            ml_result=ml_result,
                        )
                    except Exception:
                        log.exception(
                            "Failed infer/publish at stream line=%s game_id=%s event_keys=%s",
                            line_num,
                            game_id,
                            list(event.keys()),
                        )
                        raise

                    processed_moves += 1
                    latest_turn = record["turn"]
                    log.info(
                        "Published turn=%s game_id=%s move_doc_id=%s processed_moves=%s",
                        latest_turn,
                        game_id,
                        latest_entry_id,
                        processed_moves,
                    )

        except HTTPError as e:
            log.error(
                "Game stream HTTPError code=%s reason=%s game_id=%s",
                e.code,
                e.reason,
                game_id,
            )
            raise
        except URLError as e:
            log.error("Game stream URLError reason=%s game_id=%s", e.reason, game_id)
            raise

        log.info("Game stream ended for game_id=%s lines_seen=%s moves=%s", game_id, line_num, processed_moves)
        try:
            _patch_document(
                f"{game_id}/{GAME_SESSION_DOC_ID}",
                {"status": "finished", "updated_at": _timestamp_value()},
            )
        except Exception:
            log.exception("Failed to mark game_session finished game_id=%s", game_id)

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

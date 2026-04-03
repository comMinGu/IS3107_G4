import json
import threading
import time
from typing import Dict, Optional, Tuple

import chess
import numpy as np
import requests
from flask import Flask, render_template
from flask_socketio import SocketIO

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - allows app startup without torch installed
    torch = None
    nn = None


LICHESS_TV_FEED_URL = "https://lichess.org/api/tv/feed"

PIECE_TO_LAYER = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


class ChessWinCNN(nn.Module):
    """Simple CNN placeholder returning 3 logits: white/draw/black."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class HeuristicFallbackModel:
    """Used when torch isn't available; keeps API shape identical."""

    @staticmethod
    def predict(bitboard: np.ndarray, fen: str) -> np.ndarray:
        board = chess.Board(fen)
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }
        score = 0
        for piece_type, value in values.items():
            score += value * (
                len(board.pieces(piece_type, chess.WHITE))
                - len(board.pieces(piece_type, chess.BLACK))
            )

        # Convert material score into rough probabilities.
        white = 0.34 + max(min(score / 40.0, 0.35), -0.35)
        black = 0.34 - max(min(score / 40.0, 0.35), -0.35)
        draw = max(0.1, 1.0 - (white + black))
        probs = np.array([white, draw, black], dtype=np.float32)
        probs /= probs.sum()
        return probs


def fen_to_bitboard_tensor(fen: str) -> np.ndarray:
    """Converts FEN into [12, 8, 8] one-hot bitboard layers."""
    board = chess.Board(fen)
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        layer = PIECE_TO_LAYER[piece.symbol()]
        rank = 7 - chess.square_rank(square)
        file_ = chess.square_file(square)
        planes[layer, rank, file_] = 1.0

    return planes


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def extract_players(payload: Dict) -> Tuple[str, str]:
    white = "White"
    black = "Black"

    def format_player(player_obj: Dict) -> Optional[str]:
        if not isinstance(player_obj, dict):
            return None
        user = player_obj.get("user", {})
        title = ""
        name = ""
        rating = player_obj.get("rating")

        if isinstance(user, dict):
            title = user.get("title") or ""
            name = user.get("name") or user.get("username") or ""

        if not name:
            name = player_obj.get("name") or player_obj.get("username") or ""
        if not name:
            return None

        display = f"{title} {name}".strip()
        if isinstance(rating, int):
            display = f"{display} ({rating})"
        return display

    if isinstance(payload.get("white"), dict):
        white = payload["white"].get("name") or payload["white"].get("username") or white
    elif isinstance(payload.get("white"), str):
        white = payload.get("white", white)

    if isinstance(payload.get("black"), dict):
        black = payload["black"].get("name") or payload["black"].get("username") or black
    elif isinstance(payload.get("black"), str):
        black = payload.get("black", black)

    players = payload.get("players", {})
    if isinstance(players, dict):
        if isinstance(players.get("white"), dict):
            white = (
                players["white"].get("name")
                or players["white"].get("username")
                or white
            )
        if isinstance(players.get("black"), dict):
            black = (
                players["black"].get("name")
                or players["black"].get("username")
                or black
            )
    elif isinstance(players, list):
        for player in players:
            if not isinstance(player, dict):
                continue
            color = player.get("color")
            formatted = format_player(player)
            if color == "white" and formatted:
                white = formatted
            elif color == "black" and formatted:
                black = formatted

    if isinstance(payload.get("d"), dict):
        nested_white, nested_black = extract_players(payload["d"])
        white = nested_white or white
        black = nested_black or black

    return white, black


def extract_fen(payload: Dict) -> Optional[str]:
    possible_keys = ["fen", "lastFen", "currentFen"]
    for key in possible_keys:
        if key in payload and isinstance(payload[key], str) and "/" in payload[key]:
            return payload[key]

    for key in ["d", "data", "move", "state"]:
        if isinstance(payload.get(key), dict):
            nested = extract_fen(payload[key])
            if nested:
                return nested

    return None


def extract_game_id(payload: Dict) -> Optional[str]:
    possible_keys = ["id", "gameId", "game_id"]
    for key in possible_keys:
        if key in payload and isinstance(payload[key], str) and payload[key]:
            return payload[key]

    for key in ["d", "data", "game", "state", "move"]:
        if isinstance(payload.get(key), dict):
            nested = extract_game_id(payload[key])
            if nested:
                return nested

    return None


def extract_clocks(payload: Dict) -> Tuple[Optional[int], Optional[int]]:
    white_clock = None
    black_clock = None

    if isinstance(payload.get("wc"), int):
        white_clock = payload["wc"]
    if isinstance(payload.get("bc"), int):
        black_clock = payload["bc"]

    players = payload.get("players")
    if isinstance(players, list):
        for player in players:
            if not isinstance(player, dict):
                continue
            color = player.get("color")
            seconds = player.get("seconds")
            if not isinstance(seconds, int):
                continue
            if color == "white":
                white_clock = seconds
            elif color == "black":
                black_clock = seconds

    for key in ["d", "data", "game", "state", "move"]:
        if isinstance(payload.get(key), dict):
            nested_white, nested_black = extract_clocks(payload[key])
            if nested_white is not None:
                white_clock = nested_white
            if nested_black is not None:
                black_clock = nested_black

    return white_clock, black_clock


class PredictorService:
    def __init__(self) -> None:
        if torch is not None and nn is not None:
            self.model = ChessWinCNN()
            self.model.eval()
            self.use_torch = True
        else:
            self.model = HeuristicFallbackModel()
            self.use_torch = False

    def predict(self, fen: str) -> Dict[str, float]:
        bitboard = fen_to_bitboard_tensor(fen)

        if self.use_torch:
            with torch.no_grad():
                x = torch.from_numpy(bitboard).unsqueeze(0)  # [1, 12, 8, 8]
                logits = self.model(x).squeeze(0).numpy()
                probs = softmax(logits)
        else:
            probs = self.model.predict(bitboard, fen)

        return {
            "white": float(probs[0]),
            "draw": float(probs[1]),
            "black": float(probs[2]),
        }


app = Flask(__name__)
app.config["SECRET_KEY"] = "chess-dashboard-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

predictor = PredictorService()
stream_lock = threading.Lock()
stream_started = False


@app.route("/")
def index():
    return render_template("index.html")


def emit_position(
    fen: str,
    white_name: str,
    black_name: str,
    game_id: Optional[str] = None,
    white_clock: Optional[int] = None,
    black_clock: Optional[int] = None,
) -> None:
    probabilities = predictor.predict(fen)
    socketio.emit(
        "position_update",
        {
            "fen": fen,
            "game_id": game_id,
            "white_player": white_name,
            "black_player": black_name,
            "white_clock": white_clock,
            "black_clock": black_clock,
            "probabilities": probabilities,
        },
    )


def stream_lichess_tv_feed() -> None:
    """Background task: streams Lichess TV and emits updates each move."""
    headers = {"Accept": "application/x-ndjson"}
    backoff_seconds = 2
    current_game_id: Optional[str] = None
    current_white_name = "Player 1"
    current_black_name = "Player 2"
    current_white_clock: Optional[int] = None
    current_black_clock: Optional[int] = None

    while True:
        try:
            with requests.get(
                LICHESS_TV_FEED_URL,
                headers=headers,
                stream=True,
                timeout=30,
            ) as response:
                response.raise_for_status()
                backoff_seconds = 2

                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    payload = json.loads(raw_line)
                    fen = extract_fen(payload)
                    if not fen:
                        continue

                    game_id = extract_game_id(payload)
                    if game_id and game_id != current_game_id:
                        current_game_id = game_id
                        current_white_name = "Player 1"
                        current_black_name = "Player 2"
                        current_white_clock = None
                        current_black_clock = None

                    white_name, black_name = extract_players(payload)
                    if white_name and white_name not in ("White", "Player 1"):
                        current_white_name = white_name
                    if black_name and black_name not in ("Black", "Player 2"):
                        current_black_name = black_name

                    white_clock, black_clock = extract_clocks(payload)
                    if white_clock is not None:
                        current_white_clock = white_clock
                    if black_clock is not None:
                        current_black_clock = black_clock

                    emit_position(
                        fen,
                        current_white_name,
                        current_black_name,
                        game_id or current_game_id,
                        current_white_clock,
                        current_black_clock,
                    )

        except Exception as exc:
            print(f"[stream] reconnecting after error: {exc}")
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)


@socketio.on("connect")
def on_connect():
    global stream_started
    with stream_lock:
        if not stream_started:
            socketio.start_background_task(stream_lichess_tv_feed)
            stream_started = True


if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=True,
        allow_unsafe_werkzeug=True,
    )

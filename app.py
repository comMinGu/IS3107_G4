import json
import math
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
except Exception:
    torch = None
    nn = None


LICHESS_TV_FEED_URL = "https://lichess.org/api/tv/feed"
STOCKFISH_API_URL   = "https://stockfish.online/api/s/v2.php"
STOCKFISH_DEPTH     = 12

PIECE_TO_LAYER = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
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


def eval_cp_to_probs(eval_cp: float) -> np.ndarray:
    """Convert centipawn eval to win/draw/loss probabilities (Lichess formula)."""
    win_white = 1.0 / (1.0 + math.exp(-0.004 * eval_cp))
    win_black = 1.0 - win_white
    draw = max(0.05, 0.3 * math.exp(-0.000003 * eval_cp ** 2))
    total = win_white + draw + win_black
    return np.array([win_white / total, draw / total, win_black / total], dtype=np.float32)


def mate_to_probs(mate_in: int) -> np.ndarray:
    if mate_in > 0:
        return np.array([0.97, 0.02, 0.01], dtype=np.float32)
    else:
        return np.array([0.01, 0.02, 0.97], dtype=np.float32)


def stockfish_online_eval(fen: str, session: requests.Session) -> Optional[np.ndarray]:
    """Call Stockfish Online API and return probabilities, or None on failure."""
    try:
        resp = session.get(
            STOCKFISH_API_URL,
            params={"fen": fen, "depth": STOCKFISH_DEPTH},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success", True):
            return None

        mate = data.get("mate")
        if mate is not None and mate != 0:
            return mate_to_probs(int(mate))

        evaluation = data.get("evaluation")
        if evaluation is not None:
            eval_float = float(evaluation)
            eval_cp = eval_float * 100 if abs(eval_float) < 100 else eval_float
            return eval_cp_to_probs(eval_cp)

    except Exception as e:
        print(f"[stockfish] API error: {e}")

    return None


class HeuristicFallbackModel:
    """Fast material-based fallback used only when Stockfish API fails."""

    @staticmethod
    def predict(fen: str) -> np.ndarray:
        board = chess.Board(fen)
        values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
        }
        material = 0
        for piece_type, value in values.items():
            material += value * (
                len(board.pieces(piece_type, chess.WHITE))
                - len(board.pieces(piece_type, chess.BLACK))
            )
        mobility = len(list(board.legal_moves)) * (1 if board.turn == chess.WHITE else -1)
        score = material + mobility * 0.05
        eval_cp = score * 30
        return eval_cp_to_probs(eval_cp)


def fen_to_bitboard_tensor(fen: str) -> np.ndarray:
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


def extract_players_from_d(d: Dict) -> Tuple[str, str]:
    white, black = "White", "Black"
    for player in d.get("players", []):
        if not isinstance(player, dict):
            continue
        color  = player.get("color")
        user   = player.get("user", {})
        title  = user.get("title", "") if isinstance(user, dict) else ""
        name   = user.get("name", "")  if isinstance(user, dict) else ""
        rating = player.get("rating")
        if not name:
            continue
        display = f"{title} {name}".strip() if title else name
        if isinstance(rating, int):
            display = f"{display} ({rating})"
        if color == "white":
            white = display
        elif color == "black":
            black = display
    return white, black


class PredictorService:
    def __init__(self) -> None:
        self.http = requests.Session()
        self.fallback = HeuristicFallbackModel()
        self._last_call = 0.0          # timestamp of last Stockfish API call
        self._rate_limit = 1.1         # minimum seconds between API calls

        if torch is not None and nn is not None:
            self.cnn = ChessWinCNN()
            self.cnn.eval()
            self.use_torch = True
        else:
            self.cnn = None
            self.use_torch = False

    def predict(self, fen: str) -> Dict[str, float]:
        # Enforce rate limit: wait until 1.1s since last call
        elapsed = time.time() - self._last_call
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

        self._last_call = time.time()

        # 1. Try Stockfish Online API (most accurate)
        probs = stockfish_online_eval(fen, self.http)

        # 2. Fall back to CNN if available
        if probs is None and self.use_torch:
            bitboard = fen_to_bitboard_tensor(fen)
            with torch.no_grad():
                x = torch.from_numpy(bitboard).unsqueeze(0)
                logits = self.cnn(x).squeeze(0).numpy()
                probs = softmax(logits)

        # 3. Last resort: heuristic
        if probs is None:
            probs = self.fallback.predict(fen)

        return {
            "white": float(probs[0]),
            "draw":  float(probs[1]),
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
    print(f"[predict] W:{probabilities['white']:.1%} D:{probabilities['draw']:.1%} B:{probabilities['black']:.1%}")
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
    headers = {"Accept": "application/x-ndjson"}
    backoff_seconds = 2
    current_game_id: Optional[str] = None
    current_white_name = "White"
    current_black_name = "Black"
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
                print("[stream] Connected to Lichess TV feed.")

                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    try:
                        payload = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    msg_type = payload.get("t")
                    d = payload.get("d", {})
                    if not isinstance(d, dict):
                        continue

                    if msg_type == "featured":
                        current_game_id = d.get("id")
                        current_white_name, current_black_name = extract_players_from_d(d)
                        for player in d.get("players", []):
                            color = player.get("color")
                            secs  = player.get("seconds")
                            if color == "white" and isinstance(secs, int):
                                current_white_clock = secs
                            elif color == "black" and isinstance(secs, int):
                                current_black_clock = secs
                        fen = d.get("fen")
                        if not fen:
                            continue
                        print(f"[stream] New game: {current_white_name} vs {current_black_name} | {current_game_id}")

                    elif msg_type == "fen":
                        fen = d.get("fen")
                        if not fen:
                            continue
                        wc = d.get("wc")
                        bc = d.get("bc")
                        if isinstance(wc, int):
                            current_white_clock = wc
                        if isinstance(bc, int):
                            current_black_clock = bc
                        print(f"[stream] Move | W:{current_white_clock}s B:{current_black_clock}s | {fen[:40]}...")
                    else:
                        continue

                    emit_position(
                        fen,
                        current_white_name,
                        current_black_name,
                        current_game_id,
                        current_white_clock,
                        current_black_clock,
                    )

        except Exception as exc:
            print(f"[stream] Reconnecting after error: {exc}")
            time.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 30)


@socketio.on("connect")
def on_connect():
    global stream_started
    with stream_lock:
        if not stream_started:
            socketio.start_background_task(stream_lichess_tv_feed)
            stream_started = True
    print("[socket] Client connected.")


if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=True,
        allow_unsafe_werkzeug=True,
    )

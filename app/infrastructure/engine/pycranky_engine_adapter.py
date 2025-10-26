from __future__ import annotations

from threading import Lock
from typing import Optional

import chess

from app.application.ports.engine_port import EnginePort
from engines.pycranky_engine import Engine


class PyCrankyEngineAdapter(EnginePort):
    """Adapter that bridges the legacy PyCranky engine to the EnginePort."""

    def __init__(self, max_depth: int = 3) -> None:
        self._engine = Engine()
        self._lock = Lock()
        self._max_depth = max_depth

    def select_move(self, fen: str) -> Optional[str]:
        with self._lock:
            self._engine.board = chess.Board(fen)
            move = self._engine.search(max_depth=self._max_depth)
            return move.uci() if move else None


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import chess

from app.domain.errors import GameFinishedError, IllegalMoveError, WrongTurnError
from app.domain.id_generator import generate_game_id


class PlayerSide(str, Enum):
    WHITE = "white"
    BLACK = "black"


class GameStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    CHECKMATE_WHITE = "Checkmate white"
    CHECKMATE_BLACK = "Checkmate black"
    STALEMATE = "stalemate"
    DRAW = "draw"


@dataclass
class Game:
    user_side: PlayerSide
    id: str = field(default_factory=generate_game_id)
    board: chess.Board = field(default_factory=chess.Board)
    moves: List[str] = field(default_factory=list)
    status: GameStatus = GameStatus.IN_PROGRESS

    @classmethod
    def create(cls, user_side: PlayerSide, game_id: Optional[str] = None) -> "Game":
        return cls(user_side=user_side, id=game_id or generate_game_id())

    @property
    def user_color(self) -> chess.Color:
        return chess.WHITE if self.user_side == PlayerSide.WHITE else chess.BLACK

    @property
    def engine_color(self) -> chess.Color:
        return chess.BLACK if self.user_color == chess.WHITE else chess.WHITE

    def apply_player_move(self, move_uci: str) -> None:
        self._ensure_active()
        self._ensure_turn(self.user_color)
        self._push_move(move_uci)

    def apply_engine_move(self, move_uci: str) -> None:
        self._ensure_active()
        self._ensure_turn(self.engine_color)
        self._push_move(move_uci)

    def _push_move(self, move_uci: str) -> None:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError as exc:
            raise IllegalMoveError(f"Invalid move notation: {move_uci}") from exc
        if not self.board.is_legal(move):
            raise IllegalMoveError(f"Illegal move for current position: {move_uci}")
        self.board.push(move)
        self.moves.append(move_uci)
        self.status = self._derive_status()

    def _ensure_turn(self, expected: chess.Color) -> None:
        if self.board.turn != expected:
            raise WrongTurnError("It is not the requested side's turn.")

    def _ensure_active(self) -> None:
        if self.status is not GameStatus.IN_PROGRESS:
            raise GameFinishedError("Game has already finished.")

    def _derive_status(self) -> GameStatus:
        if not self.board.is_game_over(claim_draw=True):
            return GameStatus.IN_PROGRESS
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            if self.board.is_stalemate():
                return GameStatus.STALEMATE
            return GameStatus.DRAW
        return GameStatus.CHECKMATE_WHITE if outcome.winner else GameStatus.CHECKMATE_BLACK

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from app.application.ports.engine_port import EnginePort
from app.application.ports.game_repository import GameRepository
from app.domain.entities.game import GameStatus
from app.domain.errors import GameFinishedError, IllegalMoveError, WrongTurnError


@dataclass
class MoveResult:
    game_id: str
    engine_move: Optional[str]
    status: GameStatus


class MakeMoveService:
    def __init__(self, repository: GameRepository, engine: EnginePort) -> None:
        self._repository = repository
        self._engine = engine

    async def execute(self, game_id: str, player_move: str) -> MoveResult:
        game = self._repository.get(game_id)
        try:
            game.apply_player_move(player_move)
        except (IllegalMoveError, WrongTurnError, GameFinishedError):
            raise

        self._repository.save(game)

        if game.status is not GameStatus.IN_PROGRESS:
            return MoveResult(game.id, None, game.status)

        engine_move = await asyncio.to_thread(self._engine.select_move, game.board.fen())
        if engine_move:
            game.apply_engine_move(engine_move)
            self._repository.save(game)

        return MoveResult(game.id, engine_move, game.status)

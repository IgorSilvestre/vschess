from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from app.application.ports.engine_port import EnginePort
from app.application.ports.game_repository import GameRepository
from app.domain.entities.game import Game, GameStatus, PlayerSide
from app.domain.id_generator import generate_game_id


@dataclass
class CreateGameResult:
    game_id: str
    engine_move: Optional[str]
    status: GameStatus


class CreateGameService:
    def __init__(self, repository: GameRepository, engine: EnginePort) -> None:
        self._repository = repository
        self._engine = engine

    async def execute(self, side: PlayerSide) -> CreateGameResult:
        game_id = self._generate_unique_id()
        game = Game.create(user_side=side, game_id=game_id)
        self._repository.add(game)

        engine_move: Optional[str] = None
        if side == PlayerSide.BLACK:
            engine_move = await self._maybe_play_engine_move(game)

        return CreateGameResult(game_id=game.id, engine_move=engine_move, status=game.status)

    async def _maybe_play_engine_move(self, game: Game) -> Optional[str]:
        move = await asyncio.to_thread(self._engine.select_move, game.board.fen())
        if not move:
            return None
        game.apply_engine_move(move)
        self._repository.save(game)
        return move

    def _generate_unique_id(self) -> str:
        for _ in range(32):
            candidate = generate_game_id()
            if not self._repository.exists(candidate):
                return candidate
        raise RuntimeError("Unable to allocate unique game id.")

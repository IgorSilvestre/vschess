from __future__ import annotations

from threading import Lock
from typing import Dict

from app.application.ports.game_repository import GameRepository
from app.domain.entities.game import Game
from app.domain.errors import GameNotFoundError


class InMemoryGameRepository(GameRepository):
    """Thread-safe in-memory storage for games."""

    def __init__(self) -> None:
        self._games: Dict[str, Game] = {}
        self._lock = Lock()

    def add(self, game: Game) -> None:
        with self._lock:
            self._games[game.id] = game

    def get(self, game_id: str) -> Game:
        with self._lock:
            try:
                return self._games[game_id]
            except KeyError as exc:
                raise GameNotFoundError(f"Game {game_id} not found.") from exc

    def save(self, game: Game) -> None:
        with self._lock:
            if game.id not in self._games:
                raise GameNotFoundError(f"Game {game.id} not found.")
            self._games[game.id] = game

    def exists(self, game_id: str) -> bool:
        with self._lock:
            return game_id in self._games

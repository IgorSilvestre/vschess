from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities.game import Game


class GameRepository(ABC):
    """Storage boundary for chess games."""

    @abstractmethod
    def add(self, game: Game) -> None:
        ...

    @abstractmethod
    def get(self, game_id: str) -> Game:
        ...

    @abstractmethod
    def save(self, game: Game) -> None:
        ...

    @abstractmethod
    def exists(self, game_id: str) -> bool:
        ...

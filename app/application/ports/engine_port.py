from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class EnginePort(ABC):
    """Abstraction for a chess engine capable of choosing moves."""

    @abstractmethod
    def select_move(self, fen: str) -> Optional[str]:
        """Return the engine's chosen move in UCI notation for the given FEN."""


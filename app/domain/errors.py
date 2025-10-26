from __future__ import annotations


class GameError(Exception):
    """Base class for game-related domain errors."""


class IllegalMoveError(GameError):
    """Raised when a move cannot be played on the current board."""


class WrongTurnError(GameError):
    """Raised when a move is attempted while it is not that side's turn."""


class GameFinishedError(GameError):
    """Raised when trying to play moves on a finished game."""


class GameNotFoundError(GameError):
    """Raised when the requested game does not exist."""


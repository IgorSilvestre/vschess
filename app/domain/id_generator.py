from __future__ import annotations

import secrets
import string

_ADJECTIVES = [
    "bravo",
    "calmo",
    "esperto",
    "agil",
    "forte",
    "sutil",
    "firme",
    "leve",
    "rapido",
    "sagaz",
    "sereno",
    "vasto",
]

_NOUNS = [
    "torre",
    "bispo",
    "cavalo",
    "rainha",
    "rei",
    "peao",
    "castelo",
    "bico",
    "capivara",
    "leao",
    "pantera",
    "tigre",
]


def generate_game_id() -> str:
    """Return a memorable slug composed of two words and a short number."""
    adjective = secrets.choice(_ADJECTIVES)
    noun = secrets.choice(_NOUNS)
    suffix = "".join(secrets.choice(string.digits) for _ in range(2))
    return f"{adjective}-{noun}-{suffix}"

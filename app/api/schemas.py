from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class CreateGameRequest(BaseModel):
    side: Literal["white", "black"]


class CreateGameResponse(BaseModel):
    gameId: str = Field(..., alias="gameId")
    move: Optional[str]
    status: str

    class Config:
        populate_by_name = True


class MoveRequest(BaseModel):
    move: str


class MoveResponse(BaseModel):
    gameId: str = Field(..., alias="gameId")
    move: Optional[str]
    status: str

    class Config:
        populate_by_name = True

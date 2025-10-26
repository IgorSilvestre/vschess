from __future__ import annotations

from fastapi import APIRouter, FastAPI, HTTPException

from app.api.schemas import (
    CreateGameRequest,
    CreateGameResponse,
    MoveRequest,
    MoveResponse,
)
from app.application.services.create_game_service import CreateGameService
from app.application.services.make_move_service import MakeMoveService
from app.domain.entities.game import PlayerSide
from app.domain.errors import (
    GameFinishedError,
    GameNotFoundError,
    IllegalMoveError,
    WrongTurnError,
)
from app.infrastructure.engine.pycranky_engine_adapter import PyCrankyEngineAdapter
from app.infrastructure.persistence.memory_game_repository import (
    InMemoryGameRepository,
)

app = FastAPI(title="VS Chess API")

_repository = InMemoryGameRepository()
_engine_adapter = PyCrankyEngineAdapter()
_create_game = CreateGameService(_repository, _engine_adapter)
_make_move = MakeMoveService(_repository, _engine_adapter)

router = APIRouter(prefix="/api/v1")


@router.post("/games", response_model=CreateGameResponse)
async def create_game(request: CreateGameRequest) -> CreateGameResponse:
    result = await _create_game.execute(PlayerSide(request.side))
    return CreateGameResponse(
        gameId=result.game_id,
        move=result.engine_move,
        status=result.status.value,
    )


@router.post("/games/{game_id}/move", response_model=MoveResponse)
async def make_move(game_id: str, request: MoveRequest) -> MoveResponse:
    try:
        result = await _make_move.execute(game_id, request.move)
    except GameNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IllegalMoveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except WrongTurnError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except GameFinishedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return MoveResponse(
        gameId=result.game_id,
        move=result.engine_move,
        status=result.status.value,
    )


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

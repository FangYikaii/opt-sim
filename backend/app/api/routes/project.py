from __future__ import annotations

from fastapi import APIRouter

from ...schemas import WorkspaceProject
from ...service import get_project

router = APIRouter(prefix="/api", tags=["project"])


@router.get("/project", response_model=WorkspaceProject)
def read_project() -> WorkspaceProject:
    return get_project()

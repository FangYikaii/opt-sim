from __future__ import annotations

from pydantic import BaseModel


class WorkspaceProject(BaseModel):
    id: str
    name: str
    description: str

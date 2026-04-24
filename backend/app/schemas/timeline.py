from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .runs import RunStatus


TimelineEventType = Literal["agent", "tool", "warning", "approval", "result"]


class TimelineEvent(BaseModel):
    id: str
    type: TimelineEventType
    label: str
    title: str
    body: str
    meta: str | None = None
    status: RunStatus | Literal["Running"] | None = None
    actionLabel: str | None = None

from __future__ import annotations

from pydantic import BaseModel


class ErrorBody(BaseModel):
    code: str
    message: str
    details: object | None = None


class ApiErrorPayload(BaseModel):
    error: ErrorBody

from __future__ import annotations

from fastapi import HTTPException


def not_found(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )

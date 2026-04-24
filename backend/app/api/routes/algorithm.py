from __future__ import annotations

from fastapi import APIRouter

from ...schemas import AlgorithmOverview
from ...service import get_algorithm_overview

router = APIRouter(prefix="/api", tags=["algorithm"])


@router.get("/algorithm-overview", response_model=AlgorithmOverview)
def read_algorithm_overview() -> AlgorithmOverview:
    return get_algorithm_overview()

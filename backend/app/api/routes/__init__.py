from .algorithm import router as algorithm_router
from .agent import router as agent_router
from .artifacts import router as artifacts_router
from .project import router as project_router
from .review import router as review_router
from .runs import router as runs_router
from .workspace import router as workspace_router

__all__ = [
    "algorithm_router",
    "artifacts_router",
    "agent_router",
    "project_router",
    "review_router",
    "runs_router",
    "workspace_router",
]

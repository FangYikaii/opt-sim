from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import (
    algorithm_router,
    agent_router,
    artifacts_router,
    project_router,
    review_router,
    runs_router,
    workspace_router,
)
from .config import load_project_env

load_project_env()

app = FastAPI(
    title="Opt-Sim API",
    version="0.1.0",
    description="Prototype API for the Opt-Sim microstructure design workspace.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:9002",
        "http://127.0.0.1:9002",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(project_router)
app.include_router(algorithm_router)
app.include_router(runs_router)
app.include_router(review_router)
app.include_router(artifacts_router)
app.include_router(workspace_router)
app.include_router(agent_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

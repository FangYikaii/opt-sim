from __future__ import annotations

import importlib
import json

from fastapi.testclient import TestClient

from backend.app.main import app
import backend.app.runtime_store as runtime_store


client = TestClient(app)


def test_workspace_detail_includes_model_and_decision_support() -> None:
    create_response = client.post(
        "/api/agent/design-run",
        json={
            "requirementText": "Reproduce a warm copper structural color with the Ag-SiO2-Ag paper route.",
            "targetHex": "#bf6f4f",
            "topK": 3,
            "thetaDeg": 30.0,
            "polarization": "unpolarized",
        },
    )

    assert create_response.status_code == 200
    run_id = create_response.json()["activeRun"]["id"]

    workspace_response = client.get(f"/api/runs/{run_id}/workspace")

    assert workspace_response.status_code == 200
    payload = workspace_response.json()
    assert payload["activeRun"]["id"] == run_id
    assert payload["activeModel"]["status"] in {"ready", "fallback"}
    assert payload["agentConfiguration"]["mode"] in {"live", "fallback", "disabled"}
    assert payload["decisionSupport"]["headline"]


def test_workspace_missing_run_returns_404_payload() -> None:
    response = client.get("/api/runs/run-does-not-exist/workspace")

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"]["error"]["code"] == "RUN_NOT_FOUND"
    assert "run-does-not-exist" in payload["detail"]["error"]["message"]


def test_runtime_store_persists_runs_to_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runtime_store, "RUNTIME_RUNS_DIR", tmp_path / "runs")
    runtime_store._runtime_runs.clear()

    create_response = client.post(
        "/api/agent/design-run",
        json={
            "requirementText": "Reproduce a warm copper structural color with the Ag-SiO2-Ag paper route.",
            "targetHex": "#bf6f4f",
            "topK": 3,
            "thetaDeg": 15.0,
            "polarization": "unpolarized",
        },
    )

    assert create_response.status_code == 200
    run_id = create_response.json()["activeRun"]["id"]
    persisted_path = runtime_store.RUNTIME_RUNS_DIR / f"{run_id}.json"
    assert persisted_path.exists()

    runtime_store._runtime_runs.clear()
    reloaded_module = importlib.reload(runtime_store)
    monkeypatch.setattr(reloaded_module, "RUNTIME_RUNS_DIR", tmp_path / "runs")
    reloaded_module._runtime_runs.clear()
    reloaded_module._runtime_runs.update(reloaded_module._load_persisted_runtime_runs())
    reloaded_module._refresh_run_history()

    workspace = reloaded_module.get_runtime_workspace(run_id)
    assert workspace is not None
    assert workspace.activeRun.id == run_id

    payload = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert payload["workspace"]["activeRun"]["id"] == run_id

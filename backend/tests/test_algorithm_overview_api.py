from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_algorithm_overview_endpoint_returns_artifact_backed_summary() -> None:
    response = client.get("/api/algorithm-overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["algorithmName"] == "Ag-SiO2-Ag cGAN plus thin-film inverse design"
    assert payload["artifactRootPath"].endswith("/backend/artifacts")
    assert len(payload["experiments"]) >= 1
    assert len(payload["headlineMetrics"]) >= 1
    assert len(payload["operationSteps"]) >= 1
    assert payload["activeModel"] is not None
    assert payload["agentConfiguration"] is not None

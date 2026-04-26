from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from backend.app.algorithms import inverse_design
from backend.app.main import app


client = TestClient(app)


def test_inverse_design_returns_refined_or_gan_backed_candidates_with_composite_metrics() -> None:
    result = inverse_design.run_inverse_design("#bf6f4f", top_k=3)

    assert len(result.candidates) == 3
    first_candidate_metrics = {metric.label: metric.value for metric in result.candidates[0].metrics}
    assert "Composite score" in first_candidate_metrics
    assert "Source" in first_candidate_metrics
    assert any(
        {metric.label: metric.value for metric in candidate.metrics}["Source"] in {"refined", "cGAN+refined", "cGAN+TMM"}
        for candidate in result.candidates
    )


def test_inverse_design_changes_with_theta_and_reports_angle_constraint() -> None:
    normal_result = inverse_design.run_inverse_design("#bf6f4f", top_k=3, theta_deg=0.0)
    oblique_result = inverse_design.run_inverse_design("#bf6f4f", top_k=3, theta_deg=45.0)

    assert normal_result.candidates[0].simulatedColorHex != oblique_result.candidates[0].simulatedColorHex
    angle_constraint = next(
        constraint for constraint in oblique_result.constraints if constraint.label == "Incidence angle"
    )
    assert angle_constraint.detail == "Candidates were simulated at an incident angle of 45.0 degrees."


def test_inverse_design_reports_polarization_condition_in_metrics_and_constraints() -> None:
    result = inverse_design.run_inverse_design("#bf6f4f", top_k=3, theta_deg=45.0, polarization="te")

    first_metrics = {metric.label: metric.value for metric in result.candidates[0].metrics}
    polarization_constraint = next(
        constraint for constraint in result.constraints if constraint.label == "Polarization"
    )

    assert first_metrics["Incidence angle"] == "45.0 deg"
    assert first_metrics["Polarization"] == "TE"
    assert polarization_constraint.detail == "Candidates were ranked using TE light response."


def test_inverse_design_reports_selected_retrieval_metric() -> None:
    result = inverse_design.run_inverse_design(
        "#bf6f4f",
        top_k=3,
        retrieval_metric="delta_e_2000",
    )

    first_metrics = {metric.label: metric.value for metric in result.candidates[0].metrics}
    retrieval_constraint = next(
        constraint for constraint in result.constraints if constraint.label == "Retrieval metric"
    )

    assert first_metrics["Retrieval metric"] == "DeltaE 2000"
    assert retrieval_constraint.detail == "Retrieval seeds were selected using DeltaE 2000."


def test_load_saved_cgan_bundle_uses_best_experiment_checkpoint(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "exp-best" / "generator_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"checkpoint")
    sentinel = object()

    monkeypatch.setattr(
        inverse_design,
        "get_algorithm_overview",
        lambda: SimpleNamespace(bestExperimentId="exp-best"),
    )
    monkeypatch.setattr(
        inverse_design,
        "resolve_selected_checkpoint_path",
        lambda experiment_id: checkpoint_path if experiment_id == "exp-best" else None,
    )

    def fake_load_model_bundle(path):
        assert path == checkpoint_path
        return sentinel

    monkeypatch.setattr(inverse_design, "load_model_bundle", fake_load_model_bundle)

    assert inverse_design._load_saved_cgan_bundle() is sentinel


def test_load_saved_cgan_bundle_prefers_selected_checkpoint_metadata(monkeypatch, tmp_path) -> None:
    experiment_dir = tmp_path / "exp-best"
    checkpoint_path = experiment_dir / "generator_checkpoint.pt"
    selected_checkpoint_path = experiment_dir / "generator_checkpoint_best.pt"
    experiment_dir.mkdir(parents=True)
    checkpoint_path.write_bytes(b"final-checkpoint")
    selected_checkpoint_path.write_bytes(b"best-checkpoint")
    (experiment_dir / "metrics.json").write_text(
        '{"artifacts": {"selected_checkpoint": "generator_checkpoint_best.pt"}}',
        encoding="utf-8",
    )
    sentinel = object()

    monkeypatch.setattr(
        inverse_design,
        "get_algorithm_overview",
        lambda: SimpleNamespace(bestExperimentId="exp-best"),
    )
    monkeypatch.setattr(
        inverse_design,
        "resolve_selected_checkpoint_path",
        lambda experiment_id: selected_checkpoint_path if experiment_id == "exp-best" else None,
    )

    def fake_load_model_bundle(path):
        assert path == selected_checkpoint_path
        return sentinel

    monkeypatch.setattr(inverse_design, "load_model_bundle", fake_load_model_bundle)

    assert inverse_design._load_saved_cgan_bundle() is sentinel


def test_sample_gan_designs_prefers_saved_bundle(monkeypatch) -> None:
    target_lab = np.array([45.0, -3.0, 8.0], dtype=np.float64)
    lab_array = np.array([[45.0, -3.0, 8.0]], dtype=np.float64)
    design_array = np.array([[10.0, 60.0, 10.0], [30.0, 180.0, 30.0]], dtype=np.float64)
    sampled = np.array([[12.0, 80.0, 14.0], [48.0, 420.0, 39.0]], dtype=np.float64)
    expected = np.array([[12.0, 80.0, 14.0], [30.0, 180.0, 30.0]], dtype=np.float64)
    sentinel = object()

    monkeypatch.setattr(inverse_design, "_load_saved_cgan_bundle", lambda: sentinel)
    monkeypatch.setattr(
        inverse_design,
        "sample_designs_from_bundle",
        lambda bundle, target, sample_count: sampled,
    )
    monkeypatch.setattr(
        inverse_design,
        "sample_designs_from_cgan",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not retrain when a checkpoint is available")),
    )

    result = inverse_design._sample_gan_designs(
        target_lab=target_lab,
        lab_array=lab_array,
        design_array=design_array,
        sample_count=6,
    )

    assert np.array_equal(result, expected)


def test_sample_gan_designs_falls_back_to_online_training(monkeypatch) -> None:
    target_lab = np.array([45.0, -3.0, 8.0], dtype=np.float64)
    lab_array = np.array([[45.0, -3.0, 8.0]], dtype=np.float64)
    design_array = np.array([[10.0, 60.0, 10.0], [30.0, 180.0, 30.0]], dtype=np.float64)
    expected = np.array([[16.0, 92.0, 18.0]], dtype=np.float64)

    monkeypatch.setattr(inverse_design, "_load_saved_cgan_bundle", lambda: None)
    monkeypatch.setattr(
        inverse_design,
        "sample_designs_from_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("saved bundle path should not run")),
    )
    monkeypatch.setattr(
        inverse_design,
        "sample_designs_from_cgan",
        lambda **kwargs: expected,
    )

    result = inverse_design._sample_gan_designs(
        target_lab=target_lab,
        lab_array=lab_array,
        design_array=design_array,
        sample_count=6,
    )

    assert np.array_equal(result, expected)


def test_design_run_api_exposes_refined_candidate_metrics() -> None:
    response = client.post(
        "/api/agent/design-run",
        json={
            "requirementText": "Reproduce a warm copper structural color with the Ag-SiO2-Ag paper route.",
            "targetHex": "#bf6f4f",
            "topK": 3,
            "thetaDeg": 45.0,
            "polarization": "tm",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["candidates"]) == 3
    metric_labels = {metric["label"] for metric in payload["candidates"][0]["metrics"]}
    assert "Composite score" in metric_labels
    assert "Polarization" in metric_labels
    assert payload["draft"]["incidenceAngleValue"] == "45.0 deg"
    assert payload["draft"]["polarizationValue"] == "TM"
    assert payload["activeModel"]["status"] in {"ready", "fallback"}
    assert payload["agentConfiguration"]["mode"] in {"live", "fallback", "disabled"}
    assert payload["decisionSupport"]["recommendedCandidateId"] == payload["candidates"][0]["id"]


def test_artifact_detail_records_ranking_condition_and_export_metadata() -> None:
    response = client.post(
        "/api/agent/design-run",
        json={
            "requirementText": "Reproduce a warm copper structural color with the Ag-SiO2-Ag paper route.",
            "targetHex": "#bf6f4f",
            "topK": 3,
            "thetaDeg": 45.0,
            "polarization": "tm",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    run_id = payload["activeRun"]["id"]

    report = client.get(f"/api/artifacts/{run_id}-report")
    export_plan = client.get(f"/api/artifacts/{run_id}-export-plan")

    assert report.status_code == 200
    assert export_plan.status_code == 200

    report_payload = report.json()
    export_payload = export_plan.json()
    report_metadata = {item["label"]: item["value"] for item in report_payload["metadata"]}
    export_metadata = {item["label"]: item["value"] for item in export_payload["metadata"]}

    assert "ranked under 45.0 deg, TM" in report_payload["description"]
    assert report_metadata["ranked_under"] == "45.0 deg, TM"
    assert report_metadata["polarization"] == "TM"
    assert report_metadata["target_hex"] == "#BF6F4F"

    assert "ranked under 45.0 deg, TM" in export_payload["description"]
    assert export_metadata["ranked_under"] == "45.0 deg, TM"
    assert export_metadata["polarization"] == "TM"
    assert export_metadata["delivery_format"] == payload["exportEstimate"]["format"]
    assert report_metadata["decision_confidence"] in {"high", "medium", "low"}
    assert "active_model" in export_metadata

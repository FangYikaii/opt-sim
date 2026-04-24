from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from backend.app.algorithms.cgan import fit_lightweight_cgan, load_model_bundle, sample_designs_from_bundle
from backend.app.algorithms.optics import (
    reflectance_spectrum_ag_sio2_ag,
    transmittance_spectrum_ag_sio2_ag,
)
from backend.scripts.train_cgan_reproduction import (
    build_ag_sio2_ag_dataset,
    compact_reproduction_metrics_for_json,
    collect_candidate_records,
    compute_jensen_shannon_distance,
    count_solution_groups_with_dbscan,
    load_paper_dataset_csv,
    load_runtime_dependencies,
    save_model_bundle,
    summarize_target,
    write_artifact_manifest,
)

load_runtime_dependencies()


def test_lightweight_cgan_records_losses_and_samples_in_bounds() -> None:
    lab_samples = np.array(
        [
            [20.0, -2.0, 4.0],
            [25.0, 3.0, -1.0],
            [30.0, 7.0, 6.0],
            [35.0, -4.0, 9.0],
        ],
        dtype=np.float64,
    )
    design_samples = np.array(
        [
            [10.0, 60.0, 10.0],
            [15.0, 90.0, 14.0],
            [22.0, 130.0, 20.0],
            [30.0, 180.0, 30.0],
        ],
        dtype=np.float64,
    )

    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=2,
        batch_size=4,
        seed=11,
        record_losses=True,
    )
    samples = sample_designs_from_bundle(bundle, lab_samples[0], sample_count=6, seed=12)

    assert len(bundle.losses) == 2
    assert samples.shape == (6, 3)
    assert np.all((samples[:, 0] >= 10.0) & (samples[:, 0] <= 30.0))
    assert np.all((samples[:, 1] >= 60.0) & (samples[:, 1] <= 180.0))
    assert np.all((samples[:, 2] >= 10.0) & (samples[:, 2] <= 30.0))


def test_reproduction_helpers_return_candidate_metrics() -> None:
    design_samples, lab_samples, hex_samples = build_ag_sio2_ag_dataset(
        bottom_points=3,
        sio2_points=4,
        top_points=3,
    )
    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=1,
        batch_size=8,
        seed=13,
        record_losses=True,
    )
    records = collect_candidate_records(
        bundle=bundle,
        lab_samples=lab_samples,
        design_samples=design_samples,
        hex_samples=hex_samples,
        targets=["#4f86c6"],
        sample_count=5,
        top_generated=3,
        seed=13,
    )
    summary = summarize_target(records, "#4f86c6")

    assert len(records) == 4
    assert {record.source for record in records} == {"retrieval", "cgan"}
    assert summary["target_hex"] == "#4f86c6"
    assert summary["cgan_unique_rounded_0p1nm"] >= 1
    assert summary["cgan_mean_pairwise_distance_nm"] >= 0.0


def test_compute_jensen_shannon_distance_matches_identical_histograms() -> None:
    reference = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    identical = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    shifted = np.array([40.0, 50.0, 60.0, 70.0], dtype=np.float64)

    same_distance = compute_jensen_shannon_distance(
        reference,
        identical,
        bins=4,
        value_range=(10.0, 70.0),
    )
    shifted_distance = compute_jensen_shannon_distance(
        reference,
        shifted,
        bins=4,
        value_range=(10.0, 70.0),
    )

    assert same_distance == pytest.approx(0.0)
    assert shifted_distance > same_distance


def test_dbscan_solution_group_counter_detects_separated_modes() -> None:
    grouped_designs = np.array(
        [
            [0.010, 0.100, 0.010],
            [0.011, 0.101, 0.011],
            [0.012, 0.102, 0.012],
            [0.250, 0.450, 0.250],
            [0.251, 0.451, 0.251],
            [0.252, 0.452, 0.252],
        ],
        dtype=np.float64,
    )

    group_count = count_solution_groups_with_dbscan(
        grouped_designs,
        eps=0.02,
        min_samples=2,
    )

    assert group_count == 2


def test_load_paper_dataset_csv_reads_normalized_columns(tmp_path) -> None:
    dataset_path = tmp_path / "paper.csv"
    dataset_path.write_text(
        "d1,d2,d3,L,a,b\n"
        "0.01,0.20,0.03,40,-10,5\n"
        "0.02,0.30,0.04,45,-12,6\n",
        encoding="utf-8",
    )

    design_samples, lab_samples = load_paper_dataset_csv(dataset_path)

    assert design_samples.shape == (2, 3)
    assert lab_samples.shape == (2, 3)
    assert np.allclose(design_samples[0], [10.0, 200.0, 30.0])
    assert np.allclose(lab_samples[1], [45.0, -12.0, 6.0])


def test_transmissive_fabry_perot_spectra_are_energy_bounded() -> None:
    reflectance = reflectance_spectrum_ag_sio2_ag(30.0, 150.0, 30.0)
    transmittance = transmittance_spectrum_ag_sio2_ag(30.0, 150.0, 30.0)

    assert reflectance.shape == transmittance.shape
    assert np.all((reflectance >= 0.0) & (reflectance <= 1.0))
    assert np.all((transmittance >= 0.0) & (transmittance <= 1.0))
    assert np.all(reflectance + transmittance <= 1.000001)


def test_save_model_bundle_persists_regressor_state_and_metadata(tmp_path) -> None:
    lab_samples = np.array(
        [
            [20.0, -2.0, 4.0],
            [25.0, 3.0, -1.0],
            [30.0, 7.0, 6.0],
            [35.0, -4.0, 9.0],
        ],
        dtype=np.float64,
    )
    design_samples = np.array(
        [
            [10.0, 60.0, 10.0],
            [15.0, 90.0, 14.0],
            [22.0, 130.0, 20.0],
            [30.0, 180.0, 30.0],
        ],
        dtype=np.float64,
    )

    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=1,
        batch_size=4,
        regressor_epochs=1,
        seed=17,
        record_losses=True,
    )
    output_path = tmp_path / "generator_checkpoint.pt"

    save_model_bundle(bundle, output_path)
    checkpoint = torch.load(output_path, map_location="cpu")

    assert output_path.exists()
    assert checkpoint["noise_dim"] == bundle.noise_dim
    assert "generator_state_dict" in checkpoint
    assert "lab_regressor_state_dict" in checkpoint
    assert "selected_checkpoint_epoch" in checkpoint
    assert "selected_checkpoint_metric_name" in checkpoint
    assert "selected_checkpoint_metric_value" in checkpoint


def test_compact_reproduction_metrics_for_json_drops_large_arrays() -> None:
    metrics = {
        "samples_per_lab": 8,
        "mean_best_delta_e": 1.23,
        "generated_designs_nm": [[[1.0, 2.0, 3.0]]],
        "best_delta_e_values": [1.0, 2.0],
        "solution_group_counts": [2, 3],
        "abs_d2_errors_nm": [0.5, 1.5],
        "jsd": {"d1": 0.1, "d2": 0.2, "d3": 0.3},
    }

    compact = compact_reproduction_metrics_for_json(metrics)

    assert compact is not None
    assert "generated_designs_nm" not in compact
    assert "best_delta_e_values" not in compact
    assert compact["generated_designs_shape"] == [1, 1, 3]
    assert compact["best_delta_e_count"] == 2
    assert compact["details_file"] == "paper_reproduction_details.npz"


def test_write_artifact_manifest_reports_selected_and_available_checkpoints(tmp_path) -> None:
    (tmp_path / "generator_checkpoint.pt").write_bytes(b"final")
    (tmp_path / "generator_checkpoint_best.pt").write_bytes(b"best")

    manifest = write_artifact_manifest(
        output_dir=tmp_path,
        selected_checkpoint_name="generator_checkpoint_best.pt",
        final_checkpoint_name="generator_checkpoint.pt",
    )

    assert manifest["selected_checkpoint"] == "generator_checkpoint_best.pt"
    assert manifest["final_checkpoint"] == "generator_checkpoint.pt"
    assert manifest["best_available_checkpoint"] == "generator_checkpoint_best.pt"
    assert manifest["available_checkpoints"] == [
        "generator_checkpoint_best.pt",
        "generator_checkpoint.pt",
    ]


def test_load_model_bundle_restores_sampling_ready_bundle(tmp_path) -> None:
    lab_samples = np.array(
        [
            [20.0, -2.0, 4.0],
            [25.0, 3.0, -1.0],
            [30.0, 7.0, 6.0],
            [35.0, -4.0, 9.0],
        ],
        dtype=np.float64,
    )
    design_samples = np.array(
        [
            [10.0, 60.0, 10.0],
            [15.0, 90.0, 14.0],
            [22.0, 130.0, 20.0],
            [30.0, 180.0, 30.0],
        ],
        dtype=np.float64,
    )

    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=1,
        batch_size=4,
        regressor_epochs=1,
        seed=19,
        record_losses=True,
    )
    output_path = tmp_path / "generator_checkpoint.pt"

    save_model_bundle(bundle, output_path)
    loaded_bundle = load_model_bundle(output_path, device="cpu")
    samples = sample_designs_from_bundle(loaded_bundle, lab_samples[0], sample_count=5, seed=23)

    assert loaded_bundle.noise_dim == bundle.noise_dim
    assert loaded_bundle.device == "cpu"
    assert np.allclose(loaded_bundle.lab_min, bundle.lab_min)
    assert np.allclose(loaded_bundle.design_max, bundle.design_max)
    assert samples.shape == (5, 3)


def test_save_best_checkpoint_uses_selected_generator_state(tmp_path) -> None:
    lab_samples = np.array(
        [
            [20.0, -2.0, 4.0],
            [25.0, 3.0, -1.0],
            [30.0, 7.0, 6.0],
            [35.0, -4.0, 9.0],
        ],
        dtype=np.float64,
    )
    design_samples = np.array(
        [
            [10.0, 60.0, 10.0],
            [15.0, 90.0, 14.0],
            [22.0, 130.0, 20.0],
            [30.0, 180.0, 30.0],
        ],
        dtype=np.float64,
    )

    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=1,
        batch_size=4,
        regressor_epochs=1,
        seed=29,
        record_losses=True,
    )
    current_state = {
        key: value.detach().cpu().clone()
        for key, value in bundle.generator.state_dict().items()
    }
    best_state = {
        key: torch.zeros_like(value)
        for key, value in current_state.items()
    }
    bundle.best_generator_state_dict = best_state
    output_path = tmp_path / "generator_checkpoint_best.pt"

    save_model_bundle(bundle, output_path)
    checkpoint = torch.load(output_path, map_location="cpu")

    for key, value in checkpoint["generator_state_dict"].items():
        assert torch.equal(value, best_state[key])

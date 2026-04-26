from __future__ import annotations

from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from backend.app.algorithms.cgan import (
    Discriminator,
    Generator,
    fit_lightweight_cgan,
    load_model_bundle,
    sample_designs_for_labs_from_bundle,
    sample_designs_from_bundle,
)
from backend.app.algorithms.optics import (
    reflectance_spectrum_ag_sio2_ag,
    transmittance_spectrum_ag_sio2_ag,
)
from backend.scripts.train_cgan_reproduction import (
    DEFAULT_LOW_MODE_SEEKING_WEIGHT,
    _stream_logs_to_file,
    build_ag_sio2_ag_dataset,
    compute_checkpoint_score,
    compact_reproduction_metrics_for_json,
    collect_candidate_records,
    compute_jensen_shannon_distance,
    count_solution_groups_with_dbscan,
    evaluate_saved_checkpoint,
    evaluate_testing_set_distribution,
    load_paper_dataset_csv,
    load_runtime_dependencies,
    nearest_retrieval,
    nearest_retrieval_delta_e_2000,
    nearest_retrieval_with_metric,
    parse_args,
    get_experiment_preset,
    resolve_checkpoint_evaluation_budgets,
    resolve_artifact_bundle,
    save_best_model_bundle,
    save_model_bundle,
    summarize_target,
    write_metrics,
    write_artifact_manifest,
    write_candidate_csv,
    write_retrieval_comparison_json,
)
from backend.scripts.analyze_cgan_run import parse_checkpoint_log, parse_train_progress

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
    assert "mode_seeking_loss" in bundle.losses[0]
    assert samples.shape == (6, 3)
    assert np.all((samples[:, 0] >= 10.0) & (samples[:, 0] <= 30.0))
    assert np.all((samples[:, 1] >= 60.0) & (samples[:, 1] <= 180.0))
    assert np.all((samples[:, 2] >= 10.0) & (samples[:, 2] <= 30.0))


def test_lightweight_cgan_accepts_explicit_generator_and_discriminator_hyperparameters() -> None:
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
        generator_learning_rate=3e-4,
        discriminator_learning_rate=8e-5,
        steps_per_batch=2,
        alpha_start=0.05,
        alpha_ramp_epochs=4000,
        max_alpha=0.35,
        lab_delta_e_weight=0.1,
        mode_seeking_weight=0.3,
        discriminator_conditioning="target_lab",
        seed=41,
        record_losses=True,
    )

    assert bundle.generator_learning_rate == pytest.approx(3e-4)
    assert bundle.discriminator_learning_rate == pytest.approx(8e-5)
    assert bundle.steps_per_batch == 2
    assert bundle.alpha_start == pytest.approx(0.05)
    assert bundle.alpha_ramp_epochs == 4000
    assert bundle.max_alpha == pytest.approx(0.35)
    assert bundle.lab_delta_e_weight == pytest.approx(0.1)
    assert bundle.mode_seeking_weight == pytest.approx(0.3)
    assert bundle.discriminator_conditioning == "target_lab"


def test_generator_and_discriminator_use_configurable_sn_architectures() -> None:
    generator = Generator(
        noise_dim=2,
        hidden_dim=96,
        trunk_depth=3,
    )
    discriminator = Discriminator(
        hidden_dim=80,
        trunk_depth=2,
        conditioning="target_lab",
    )
    scores = discriminator(torch.randn(3, 3), torch.randn(3, 3))

    generator_linears = [module for module in generator.modules() if isinstance(module, torch.nn.Linear)]
    discriminator_linears = [
        module for module in discriminator.modules() if isinstance(module, torch.nn.Linear)
    ]
    discriminator_batch_norms = [
        module for module in discriminator.modules() if isinstance(module, torch.nn.BatchNorm1d)
    ]

    assert generator.hidden_dim == 96
    assert generator.trunk_depth == 3
    assert discriminator.hidden_dim == 80
    assert discriminator.trunk_depth == 2
    assert discriminator.conditioning == "target_lab"
    assert discriminator_batch_norms == []
    assert discriminator_linears[0].in_features == 6
    assert scores.shape == (3, 1)
    assert all(hasattr(module, "weight_u") for module in generator_linears)
    assert all(hasattr(module, "weight_u") for module in discriminator_linears)


def test_lightweight_cgan_persists_architecture_metadata() -> None:
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
        generator_hidden_dim=144,
        generator_depth=5,
        discriminator_hidden_dim=112,
        discriminator_depth=4,
        seed=101,
        record_losses=True,
    )

    assert bundle.generator_hidden_dim == 144
    assert bundle.generator_depth == 5
    assert bundle.discriminator_hidden_dim == 112
    assert bundle.discriminator_depth == 4


def test_lightweight_cgan_records_mode_seeking_loss_when_enabled() -> None:
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
        regressor_epochs=1,
        mode_seeking_weight=0.4,
        seed=57,
        record_losses=True,
    )

    assert bundle.mode_seeking_weight == pytest.approx(0.4)
    assert all("mode_seeking_loss" in row for row in bundle.losses)
    assert all(float(row["mode_seeking_loss"]) >= 0.0 for row in bundle.losses)


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
    assert all(record.retrieval_metric == "euclidean_lab" for record in records)


def test_nearest_retrieval_delta_e_2000_supports_perceptual_path() -> None:
    lab_samples = np.array(
        [
            [55.0, 0.0, 0.0],
            [55.0, 12.0, -12.0],
            [55.0, 4.0, -4.0],
        ],
        dtype=np.float64,
    )
    design_samples = np.array(
        [
            [10.0, 60.0, 10.0],
            [15.0, 90.0, 15.0],
            [20.0, 120.0, 20.0],
        ],
        dtype=np.float64,
    )
    target_lab = np.array([55.0, 3.0, -3.0], dtype=np.float64)

    euclidean_distance, euclidean_design = nearest_retrieval(lab_samples, design_samples, target_lab)
    delta_e_distance, delta_e_design = nearest_retrieval_delta_e_2000(lab_samples, design_samples, target_lab)

    assert euclidean_distance >= 0.0
    assert delta_e_distance >= 0.0
    assert euclidean_design.shape == (3,)
    assert delta_e_design.shape == (3,)


def test_nearest_retrieval_with_metric_returns_delta_e_scored_candidate() -> None:
    design_samples, lab_samples, hex_samples = build_ag_sio2_ag_dataset(
        bottom_points=2,
        sio2_points=2,
        top_points=2,
    )
    target_lab = lab_samples[0]

    retrieval_delta_e, simulated_hex, retrieval_design = nearest_retrieval_with_metric(
        lab_samples,
        design_samples,
        hex_samples,
        target_lab,
        retrieval_metric="delta_e_2000",
    )

    assert retrieval_delta_e == pytest.approx(0.0)
    assert simulated_hex == hex_samples[0]
    assert np.allclose(retrieval_design, design_samples[0])


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
    assert checkpoint["checkpoint_format_version"] == 3
    assert checkpoint["lab_scaling_type"] == "standardization"
    assert checkpoint["design_scaling_type"] == "normalization"
    assert "lab_mean" in checkpoint
    assert "lab_std" in checkpoint
    assert checkpoint["generator_learning_rate"] == pytest.approx(bundle.generator_learning_rate)
    assert checkpoint["discriminator_learning_rate"] == pytest.approx(bundle.discriminator_learning_rate)
    assert checkpoint["steps_per_batch"] == bundle.steps_per_batch
    assert checkpoint["alpha_start"] == pytest.approx(bundle.alpha_start)
    assert checkpoint["alpha_ramp_epochs"] == bundle.alpha_ramp_epochs
    assert checkpoint["max_alpha"] == pytest.approx(bundle.max_alpha)
    assert checkpoint["lab_delta_e_weight"] == pytest.approx(bundle.lab_delta_e_weight)
    assert checkpoint["mode_seeking_weight"] == pytest.approx(bundle.mode_seeking_weight)
    assert checkpoint["generator_hidden_dim"] == bundle.generator_hidden_dim
    assert checkpoint["generator_depth"] == bundle.generator_depth
    assert checkpoint["discriminator_hidden_dim"] == bundle.discriminator_hidden_dim
    assert checkpoint["discriminator_depth"] == bundle.discriminator_depth
    assert checkpoint["discriminator_conditioning"] == bundle.discriminator_conditioning
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
    assert loaded_bundle.checkpoint_format_version == 3
    assert loaded_bundle.lab_scaling_type == "standardization"
    assert loaded_bundle.design_scaling_type == "normalization"
    assert np.allclose(loaded_bundle.lab_mean, bundle.lab_mean)
    assert np.allclose(loaded_bundle.lab_std, bundle.lab_std)
    assert np.allclose(loaded_bundle.design_max, bundle.design_max)
    assert loaded_bundle.generator_learning_rate == pytest.approx(bundle.generator_learning_rate)
    assert loaded_bundle.discriminator_learning_rate == pytest.approx(bundle.discriminator_learning_rate)
    assert loaded_bundle.steps_per_batch == bundle.steps_per_batch
    assert loaded_bundle.alpha_start == pytest.approx(bundle.alpha_start)
    assert loaded_bundle.alpha_ramp_epochs == bundle.alpha_ramp_epochs
    assert loaded_bundle.max_alpha == pytest.approx(bundle.max_alpha)
    assert loaded_bundle.lab_delta_e_weight == pytest.approx(bundle.lab_delta_e_weight)
    assert loaded_bundle.mode_seeking_weight == pytest.approx(bundle.mode_seeking_weight)
    assert loaded_bundle.generator_hidden_dim == bundle.generator_hidden_dim
    assert loaded_bundle.generator_depth == bundle.generator_depth
    assert loaded_bundle.discriminator_hidden_dim == bundle.discriminator_hidden_dim
    assert loaded_bundle.discriminator_depth == bundle.discriminator_depth
    assert loaded_bundle.discriminator_conditioning == bundle.discriminator_conditioning
    assert samples.shape == (5, 3)


def test_load_model_bundle_supports_legacy_min_max_lab_checkpoint(tmp_path) -> None:
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
        seed=71,
        record_losses=True,
    )
    output_path = tmp_path / "legacy_generator_checkpoint.pt"
    save_model_bundle(bundle, output_path)
    checkpoint = torch.load(output_path, map_location="cpu")
    del checkpoint["checkpoint_format_version"]
    del checkpoint["lab_scaling_type"]
    del checkpoint["design_scaling_type"]
    checkpoint["lab_min"] = bundle.lab_min.tolist()
    checkpoint["lab_max"] = bundle.lab_max.tolist()
    torch.save(checkpoint, output_path)

    loaded_bundle = load_model_bundle(output_path, device="cpu")
    samples = sample_designs_from_bundle(loaded_bundle, lab_samples[0], sample_count=3, seed=73)

    assert loaded_bundle.checkpoint_format_version == 1
    assert loaded_bundle.lab_scaling_type == "min_max"
    assert np.allclose(loaded_bundle.lab_min, bundle.lab_min)
    assert np.allclose(loaded_bundle.lab_max, bundle.lab_max)
    assert samples.shape == (3, 3)


def test_load_model_bundle_infers_legacy_generator_architecture_metadata(tmp_path) -> None:
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
        generator_hidden_dim=128,
        generator_depth=9,
        seed=79,
        record_losses=True,
    )
    output_path = tmp_path / "legacy_arch_checkpoint.pt"
    save_model_bundle(bundle, output_path)
    checkpoint = torch.load(output_path, map_location="cpu")
    del checkpoint["generator_hidden_dim"]
    del checkpoint["generator_depth"]
    torch.save(checkpoint, output_path)

    loaded_bundle = load_model_bundle(output_path, device="cpu")

    assert loaded_bundle.generator_hidden_dim == 128
    assert loaded_bundle.generator_depth == 9


def test_load_model_bundle_infers_legacy_variable_width_generator_architecture(tmp_path) -> None:
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
    generator = Generator(
        noise_dim=2,
        hidden_dim=128,
        noise_hidden_dim=128,
        lab_hidden_dim=128,
        regressor_hidden_dims=[256, 256, 256, 256, 256, 256, 256, 256, 128],
    )
    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=1,
        batch_size=4,
        regressor_epochs=1,
        noise_dim=2,
        generator_hidden_dim=128,
        generator_depth=9,
        seed=83,
        record_losses=True,
    )
    bundle.generator = generator
    output_path = tmp_path / "legacy_variable_width_checkpoint.pt"
    save_model_bundle(bundle, output_path)
    checkpoint = torch.load(output_path, map_location="cpu")
    del checkpoint["generator_hidden_dim"]
    del checkpoint["generator_depth"]
    torch.save(checkpoint, output_path)

    loaded_bundle = load_model_bundle(output_path, device="cpu")

    assert loaded_bundle.noise_dim == 2
    assert loaded_bundle.generator_hidden_dim == 128
    assert loaded_bundle.generator_depth == 9
    assert loaded_bundle.generator.noise_hidden_dim == 128
    assert loaded_bundle.generator.lab_hidden_dim == 128
    assert loaded_bundle.generator.regressor_hidden_dims == (
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        256,
        128,
    )


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

    save_best_model_bundle(bundle, output_path)
    checkpoint = torch.load(output_path, map_location="cpu")

    for key, value in checkpoint["generator_state_dict"].items():
        assert torch.equal(value, best_state[key])


def test_save_model_bundle_keeps_current_generator_state_by_default(tmp_path) -> None:
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
        seed=211,
        record_losses=True,
    )
    current_state = {
        key: value.detach().cpu().clone()
        for key, value in bundle.generator.state_dict().items()
    }
    bundle.best_generator_state_dict = {
        key: torch.zeros_like(value)
        for key, value in current_state.items()
    }

    output_path = tmp_path / "generator_checkpoint.pt"
    save_model_bundle(bundle, output_path)
    checkpoint = torch.load(output_path, map_location="cpu")

    for key, value in checkpoint["generator_state_dict"].items():
        assert torch.equal(value.cpu(), current_state[key].cpu())


def test_sample_designs_for_labs_from_bundle_chunks_large_batches() -> None:
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
        seed=53,
        record_losses=True,
        device="cpu",
    )
    original_forward = bundle.generator.forward
    batch_sizes: list[int] = []

    def recording_forward(lab_norm: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        batch_sizes.append(int(lab_norm.shape[0]))
        return original_forward(lab_norm, noise)

    bundle.generator.forward = recording_forward  # type: ignore[method-assign]
    samples = sample_designs_for_labs_from_bundle(
        bundle,
        lab_samples[:3],
        sample_count=5,
        seed=59,
        device="cpu",
        max_forward_batch_size=7,
    )

    assert samples.shape == (3, 5, 3)
    assert batch_sizes == [7, 7, 1]
    assert max(batch_sizes) <= 7


def test_lightweight_cgan_emits_progress_events() -> None:
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
    progress_events: list[dict[str, object]] = []

    def progress_callback(event: dict[str, object]) -> None:
        progress_events.append(event)

    fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=2,
        batch_size=4,
        regressor_epochs=2,
        seed=31,
        record_losses=True,
        progress_callback=progress_callback,
        progress_interval=1,
        regressor_progress_interval=1,
    )

    assert progress_events
    assert progress_events[0]["stage"] == "regressor"
    assert progress_events[0]["event"] == "start"
    assert any(
        event["stage"] == "regressor"
        and event["event"] == "progress"
        and event["epoch"] == 1
        for event in progress_events
    )
    assert any(
        event["stage"] == "training"
        and event["event"] == "progress"
        and event["epoch"] == 2
        for event in progress_events
    )
    training_start = next(event for event in progress_events if event["stage"] == "training" and event["event"] == "start")
    assert training_start["generator_learning_rate"] == pytest.approx(1e-3)
    assert training_start["discriminator_learning_rate"] == pytest.approx(2e-4)
    assert training_start["steps_per_batch"] == 1
    assert training_start["alpha_start"] == pytest.approx(0.0)
    assert training_start["alpha_ramp_epochs"] == 2000
    assert training_start["max_alpha"] == pytest.approx(1.0)
    assert training_start["lab_delta_e_weight"] == pytest.approx(0.0)
    assert training_start["mode_seeking_weight"] == pytest.approx(0.0)
    assert training_start["discriminator_conditioning"] == "target_lab"
    assert progress_events[-1]["stage"] == "training"
    assert progress_events[-1]["event"] == "complete"


def test_parse_args_exposes_explicit_hyperparameters_and_retrieval_metric(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_cgan_reproduction.py",
            "--generator-learning-rate",
            "0.0007",
            "--discriminator-learning-rate",
            "0.0002",
            "--steps-per-batch",
            "3",
            "--alpha-start",
            "0.08",
            "--mode-seeking-weight",
            "0.12",
            "--generator-hidden-dim",
            "192",
            "--generator-depth",
            "5",
            "--discriminator-hidden-dim",
            "160",
            "--discriminator-depth",
            "4",
            "--alpha-ramp-epochs",
            "6000",
            "--max-alpha",
            "0.4",
            "--lab-delta-e-weight",
            "0.06",
            "--discriminator-conditioning",
            "target_lab",
            "--checkpoint-budget-mode",
            "recheck_best",
            "--checkpoint-recheck-samples-per-lab",
            "256",
            "--retrieval-metric",
            "delta_e_2000",
            "--checkpoint-score-weight-mean-best-delta-e",
            "1.1",
            "--checkpoint-score-weight-d2-within-5nm",
            "9.0",
            "--checkpoint-score-weight-d3-jsd",
            "1.5",
        ],
    )

    args = parse_args()

    assert args.generator_learning_rate == pytest.approx(0.0007)
    assert args.discriminator_learning_rate == pytest.approx(0.0002)
    assert args.steps_per_batch == 3
    assert args.alpha_start == pytest.approx(0.08)
    assert args.mode_seeking_weight == pytest.approx(0.12)
    assert args.generator_hidden_dim == 192
    assert args.generator_depth == 5
    assert args.discriminator_hidden_dim == 160
    assert args.discriminator_depth == 4
    assert args.alpha_ramp_epochs == 6000
    assert args.max_alpha == pytest.approx(0.4)
    assert args.lab_delta_e_weight == pytest.approx(0.06)
    assert args.discriminator_conditioning == "target_lab"
    assert args.checkpoint_budget_mode == "recheck_best"
    assert args.checkpoint_recheck_samples_per_lab == 256
    assert args.retrieval_metric == "delta_e_2000"
    assert args.checkpoint_score_weight_mean_best_delta_e == pytest.approx(1.1)
    assert args.checkpoint_score_weight_d2_within_5nm == pytest.approx(9.0)
    assert args.checkpoint_score_weight_d3_jsd == pytest.approx(1.5)


def test_write_metrics_records_explicit_hyperparameters_and_retrieval_metric(tmp_path) -> None:
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
        generator_learning_rate=5e-4,
        discriminator_learning_rate=1e-4,
        steps_per_batch=2,
        alpha_start=0.1,
        generator_hidden_dim=160,
        generator_depth=5,
        discriminator_hidden_dim=128,
        discriminator_depth=4,
        alpha_ramp_epochs=5000,
        max_alpha=0.25,
        lab_delta_e_weight=0.05,
        mode_seeking_weight=0.2,
        discriminator_conditioning="target_lab",
        seed=91,
        record_losses=True,
    )
    records = collect_candidate_records(
        bundle=bundle,
        lab_samples=lab_samples,
        design_samples=design_samples,
        hex_samples=None,
        targets=["#4f86c6"],
        sample_count=2,
        top_generated=1,
        seed=91,
        retrieval_metric="delta_e_2000",
    )
    output_path = tmp_path / "metrics.json"
    args = SimpleNamespace(
        epochs=3,
        batch_size=4,
        noise_dim=2,
        generator_learning_rate=5e-4,
        discriminator_learning_rate=1e-4,
        steps_per_batch=2,
        alpha_start=0.1,
        mode_seeking_weight=0.2,
        generator_hidden_dim=160,
        generator_depth=5,
        discriminator_hidden_dim=128,
        discriminator_depth=4,
        alpha_ramp_epochs=5000,
        max_alpha=0.25,
        lab_delta_e_weight=0.05,
        discriminator_conditioning="target_lab",
        seed=91,
        device="cpu",
        regressor_epochs=1,
        checkpoint_eval_interval=10,
        checkpoint_budget_mode="recheck_best",
        checkpoint_recheck_samples_per_lab=256,
        checkpoint_samples_per_lab=64,
        paper_samples_per_lab=1000,
        retrieval_metric="delta_e_2000",
        checkpoint_score_weight_mean_best_delta_e=1.0,
        checkpoint_score_weight_median_best_delta_e=0.02,
        checkpoint_score_weight_d2_within_5nm=8.0,
        checkpoint_score_weight_mean_jsd=0.5,
        checkpoint_score_weight_d3_jsd=2.0,
    )

    write_metrics(
        bundle=bundle,
        records=records,
        targets=["#4f86c6"],
        dataset_size=len(design_samples),
        args=args,
        output_path=output_path,
    )
    payload = output_path.read_text(encoding="utf-8")

    assert '"generator_learning_rate": 0.0005' in payload
    assert '"discriminator_learning_rate": 0.0001' in payload
    assert '"steps_per_batch": 2' in payload
    assert '"alpha_start": 0.1' in payload
    assert '"generator_hidden_dim": 160' in payload
    assert '"generator_depth": 5' in payload
    assert '"discriminator_hidden_dim": 128' in payload
    assert '"discriminator_depth": 4' in payload
    assert '"alpha_ramp_epochs": 5000' in payload
    assert '"max_alpha": 0.25' in payload
    assert '"lab_delta_e_weight": 0.05' in payload
    assert '"mode_seeking_weight": 0.2' in payload
    assert '"discriminator_conditioning": "target_lab"' in payload
    assert '"retrieval_metric": "delta_e_2000"' in payload
    assert '"checkpoint_score_weights"' in payload
    assert '"checkpoint_budget_mode": "recheck_best"' in payload
    assert '"checkpoint_samples_per_lab": 64' in payload
    assert '"checkpoint_recheck_samples_per_lab": 256' in payload
    assert '"final_samples_per_lab": 1000' in payload
    assert '"mean_best_delta_e": 1.0' in payload
    assert '"d2_within_5nm": 8.0' in payload
    assert '"d3_jsd": 2.0' in payload
    assert '"colorimetry"' in payload
    assert '"illuminant_source": "refer_data/D65.csv"' in payload
    assert '"tristimulus_source": "refer_data/tristimulus.csv"' in payload
    assert '"conversion_backend": "colour-science"' in payload
    assert '"lab": "standardization"' in payload
    assert '"design": "normalization"' in payload


def test_write_candidate_csv_includes_retrieval_metric_column(tmp_path) -> None:
    records = [
        SimpleNamespace(
            target_hex="#4f86c6",
            source="retrieval",
            rank=1,
            d_ag_bottom_nm=10.0,
            d_sio2_nm=60.0,
            d_ag_top_nm=10.0,
            delta_e=1.23,
            simulated_hex="#123456",
            retrieval_metric="delta_e_2000",
        )
    ]
    output_path = tmp_path / "candidate_samples.csv"

    write_candidate_csv(records, output_path)
    payload = output_path.read_text(encoding="utf-8")

    assert "retrieval_metric" in payload.splitlines()[0]
    assert "delta_e_2000" in payload.splitlines()[1]


def test_write_retrieval_comparison_json_records_ab_metrics(tmp_path) -> None:
    euclidean_records = [
        SimpleNamespace(
            target_hex="#4f86c6",
            source="retrieval",
            rank=1,
            d_ag_bottom_nm=10.0,
            d_sio2_nm=60.0,
            d_ag_top_nm=10.0,
            delta_e=2.0,
            simulated_hex="#111111",
            retrieval_metric="euclidean_lab",
        ),
        SimpleNamespace(
            target_hex="#4f86c6",
            source="cgan",
            rank=1,
            d_ag_bottom_nm=12.0,
            d_sio2_nm=80.0,
            d_ag_top_nm=12.0,
            delta_e=1.5,
            simulated_hex="#222222",
            retrieval_metric="euclidean_lab",
        ),
    ]
    delta_e_records = [
        SimpleNamespace(
            target_hex="#4f86c6",
            source="retrieval",
            rank=1,
            d_ag_bottom_nm=11.0,
            d_sio2_nm=62.0,
            d_ag_top_nm=11.0,
            delta_e=1.0,
            simulated_hex="#333333",
            retrieval_metric="delta_e_2000",
        ),
        SimpleNamespace(
            target_hex="#4f86c6",
            source="cgan",
            rank=1,
            d_ag_bottom_nm=12.0,
            d_sio2_nm=80.0,
            d_ag_top_nm=12.0,
            delta_e=1.5,
            simulated_hex="#222222",
            retrieval_metric="delta_e_2000",
        ),
    ]
    output_path = tmp_path / "retrieval_metric_comparison.json"

    write_retrieval_comparison_json(
        comparison_records_by_metric={
            "euclidean_lab": euclidean_records,
            "delta_e_2000": delta_e_records,
        },
        output_path=output_path,
        elapsed_seconds_by_metric={
            "euclidean_lab": 0.5,
            "delta_e_2000": 0.8,
        },
    )
    payload = output_path.read_text(encoding="utf-8")

    assert '"metrics": {' in payload
    assert '"euclidean_lab"' in payload
    assert '"delta_e_2000"' in payload
    assert '"mean_retrieval_best_delta_e": 2.0' in payload
    assert '"mean_retrieval_best_delta_e": 1.0' in payload
    assert '"retrieval_wins_vs_cgan": 1' in payload
    assert '"elapsed_seconds": 0.8' in payload


def test_stream_logs_to_file_writes_console_output(tmp_path, capsys) -> None:
    log_path = tmp_path / "train.log"

    with _stream_logs_to_file(log_path):
        print("hello-log")

    captured = capsys.readouterr()

    assert "hello-log" in captured.out
    assert "hello-log" in log_path.read_text(encoding="utf-8")


def test_resolve_artifact_bundle_prefers_best_checkpoint_when_available(tmp_path) -> None:
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
    training_bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=1,
        batch_size=4,
        regressor_epochs=1,
        seed=97,
        record_losses=True,
    )
    zero_state = {
        key: torch.zeros_like(value)
        for key, value in training_bundle.generator.state_dict().items()
    }
    training_bundle.best_generator_state_dict = zero_state
    best_checkpoint_path = tmp_path / "generator_checkpoint_best.pt"
    save_best_model_bundle(training_bundle, best_checkpoint_path)

    artifact_bundle = resolve_artifact_bundle(
        training_bundle,
        best_checkpoint_path=best_checkpoint_path,
        device="cpu",
    )

    for key, value in artifact_bundle.generator.state_dict().items():
        assert torch.equal(value.cpu(), zero_state[key].cpu())


def test_resolve_checkpoint_evaluation_budgets_supports_recheck_mode() -> None:
    args = SimpleNamespace(
        paper_samples_per_lab=1000,
        checkpoint_samples_per_lab=64,
        checkpoint_budget_mode="recheck_best",
        checkpoint_recheck_samples_per_lab=256,
    )

    budgets = resolve_checkpoint_evaluation_budgets(args)

    assert budgets["checkpoint_samples_per_lab"] == 64
    assert budgets["checkpoint_recheck_samples_per_lab"] == 256
    assert budgets["final_samples_per_lab"] == 1000


def test_parse_args_defaults_to_new_training_baseline(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["train_cgan_reproduction.py"])

    args = parse_args()

    assert args.batch_size == 2048
    assert args.noise_dim == 8
    assert args.alpha_ramp_epochs == 2000
    assert args.max_alpha == pytest.approx(1.0)
    assert args.mode_seeking_weight == pytest.approx(0.1)
    assert args.experiment_preset is None


def test_fit_lightweight_cgan_defaults_to_new_training_baseline() -> None:
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
        regressor_epochs=1,
        seed=113,
        record_losses=True,
    )

    assert bundle.noise_dim == 8
    assert bundle.alpha_ramp_epochs == 2000


def test_get_experiment_preset_returns_legacy_tune4_alpha_ablation_config() -> None:
    preset = get_experiment_preset("legacy_tune4_alpha_conditional_d_noise8_mode_seeking_low")

    assert preset is not None
    assert preset.overrides["noise_dim"] == 8
    assert preset.overrides["discriminator_conditioning"] == "target_lab"
    assert preset.overrides["alpha_ramp_epochs"] == 40000
    assert preset.overrides["max_alpha"] == pytest.approx(0.3)
    assert preset.overrides["mode_seeking_weight"] == pytest.approx(DEFAULT_LOW_MODE_SEEKING_WEIGHT)


def test_parse_args_applies_legacy_tune4_alpha_preset(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_cgan_reproduction.py",
            "--experiment-preset",
            "legacy_tune4_alpha",
        ],
    )

    args = parse_args()

    assert args.experiment_preset == "legacy_tune4_alpha"
    assert args.batch_size == 16384
    assert args.noise_dim == 2
    assert args.generator_hidden_dim == 128
    assert args.generator_depth == 9
    assert args.discriminator_hidden_dim == 128
    assert args.discriminator_depth == 4
    assert args.discriminator_conditioning == "none"
    assert args.alpha_ramp_epochs == 40000
    assert args.max_alpha == pytest.approx(0.3)
    assert args.mode_seeking_weight == pytest.approx(0.0)
    assert args.regressor_epochs == 10000


def test_parse_args_applies_conditional_discriminator_only_ablation(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_cgan_reproduction.py",
            "--experiment-preset",
            "legacy_tune4_alpha_conditional_d",
        ],
    )

    args = parse_args()

    assert args.noise_dim == 2
    assert args.discriminator_conditioning == "target_lab"
    assert args.alpha_ramp_epochs == 40000
    assert args.max_alpha == pytest.approx(0.3)
    assert args.mode_seeking_weight == pytest.approx(0.0)


def test_parse_args_applies_noise8_ablation(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_cgan_reproduction.py",
            "--experiment-preset",
            "legacy_tune4_alpha_conditional_d_noise8",
        ],
    )

    args = parse_args()

    assert args.noise_dim == 8
    assert args.discriminator_conditioning == "target_lab"
    assert args.alpha_ramp_epochs == 40000
    assert args.max_alpha == pytest.approx(0.3)
    assert args.mode_seeking_weight == pytest.approx(0.0)


def test_parse_args_applies_low_mode_seeking_ablation(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_cgan_reproduction.py",
            "--experiment-preset",
            "legacy_tune4_alpha_conditional_d_noise8_mode_seeking_low",
        ],
    )

    args = parse_args()

    assert args.noise_dim == 8
    assert args.discriminator_conditioning == "target_lab"
    assert args.alpha_ramp_epochs == 40000
    assert args.max_alpha == pytest.approx(0.3)
    assert args.mode_seeking_weight == pytest.approx(DEFAULT_LOW_MODE_SEEKING_WEIGHT)


def test_parse_args_allows_explicit_cli_override_after_preset(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_cgan_reproduction.py",
            "--experiment-preset",
            "legacy_tune4_alpha_conditional_d_noise8",
            "--mode-seeking-weight",
            "0.05",
            "--batch-size",
            "4096",
        ],
    )

    args = parse_args()

    assert args.noise_dim == 8
    assert args.discriminator_conditioning == "target_lab"
    assert args.batch_size == 4096
    assert args.mode_seeking_weight == pytest.approx(0.05)


def test_evaluate_testing_set_distribution_emits_progress_events() -> None:
    design_samples, lab_samples, _ = build_ag_sio2_ag_dataset(
        bottom_points=2,
        sio2_points=2,
        top_points=2,
    )
    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=1,
        batch_size=8,
        regressor_epochs=1,
        seed=37,
        record_losses=True,
    )
    progress_events: list[dict[str, object]] = []

    metrics = evaluate_testing_set_distribution(
        bundle=bundle,
        test_designs=design_samples[:3],
        test_labs=lab_samples[:3],
        samples_per_lab=2,
        seed=41,
        progress_callback=progress_events.append,
        progress_interval=1,
        progress_phase="checkpoint",
    )

    assert metrics["samples_per_lab"] == 2
    assert progress_events[0] == {
        "stage": "paper_eval",
        "event": "start",
        "phase": "checkpoint",
        "processed": 0,
        "total": 3,
    }
    assert any(
        event["stage"] == "paper_eval"
        and event["event"] == "progress"
        and event["processed"] == 2
        and event["total"] == 3
        for event in progress_events
    )
    assert progress_events[-1]["stage"] == "paper_eval"
    assert progress_events[-1]["event"] == "complete"
    assert progress_events[-1]["phase"] == "checkpoint"


def test_evaluate_saved_checkpoint_reuses_paper_eval_pipeline(tmp_path) -> None:
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
        seed=59,
        record_losses=True,
    )
    checkpoint_path = tmp_path / "generator_checkpoint.pt"
    save_model_bundle(bundle, checkpoint_path)
    paper_test_csv = tmp_path / "paper_test.csv"
    paper_test_csv.write_text(
        "d1,d2,d3,L,a,b\n"
        "0.01,0.06,0.01,20,-2,4\n"
        "0.03,0.18,0.03,35,-4,9\n",
        encoding="utf-8",
    )
    progress_events: list[dict[str, object]] = []

    loaded_bundle, metrics = evaluate_saved_checkpoint(
        checkpoint_path=checkpoint_path,
        paper_test_csv=paper_test_csv,
        samples_per_lab=3,
        seed=73,
        device="cpu",
        progress_callback=progress_events.append,
        progress_interval=1,
        progress_phase="eval_only",
    )

    assert loaded_bundle.device == "cpu"
    assert metrics["samples_per_lab"] == 3
    assert np.asarray(metrics["generated_designs_nm"]).shape == (2, 3, 3)
    assert progress_events[0]["stage"] == "paper_eval"
    assert progress_events[0]["phase"] == "eval_only"
    assert progress_events[-1]["event"] == "complete"


def test_checkpoint_score_prefers_lower_error_and_higher_coverage() -> None:
    stronger_metrics = {
        "mean_best_delta_e": 1.0,
        "median_best_delta_e": 0.8,
        "d2_ground_truth_within_5nm_ratio": 0.95,
        "jsd": {"d1": 0.05, "d2": 0.04, "d3": 0.03},
    }
    weaker_metrics = {
        "mean_best_delta_e": 2.0,
        "median_best_delta_e": 1.6,
        "d2_ground_truth_within_5nm_ratio": 0.70,
        "jsd": {"d1": 0.10, "d2": 0.11, "d3": 0.12},
    }

    stronger_score = compute_checkpoint_score(stronger_metrics)
    weaker_score = compute_checkpoint_score(weaker_metrics)

    assert stronger_score < weaker_score


def test_checkpoint_score_prioritizes_mean_best_delta_e_and_d2_before_jsd() -> None:
    primary_objective_metrics = {
        "mean_best_delta_e": 7.0,
        "median_best_delta_e": 6.8,
        "d2_ground_truth_within_5nm_ratio": 0.90,
        "jsd": {"d1": 0.20, "d2": 0.22, "d3": 0.24},
    }
    better_jsd_but_worse_primary_metrics = {
        "mean_best_delta_e": 8.0,
        "median_best_delta_e": 7.8,
        "d2_ground_truth_within_5nm_ratio": 0.78,
        "jsd": {"d1": 0.04, "d2": 0.05, "d3": 0.03},
    }

    primary_score = compute_checkpoint_score(primary_objective_metrics)
    better_jsd_score = compute_checkpoint_score(better_jsd_but_worse_primary_metrics)

    assert primary_score < better_jsd_score


def test_checkpoint_score_uses_d3_as_secondary_tiebreaker() -> None:
    lower_d3_metrics = {
        "mean_best_delta_e": 7.5,
        "median_best_delta_e": 7.4,
        "d2_ground_truth_within_5nm_ratio": 0.88,
        "jsd": {"d1": 0.10, "d2": 0.12, "d3": 0.08},
    }
    higher_d3_metrics = {
        "mean_best_delta_e": 7.5,
        "median_best_delta_e": 7.4,
        "d2_ground_truth_within_5nm_ratio": 0.88,
        "jsd": {"d1": 0.10, "d2": 0.12, "d3": 0.22},
    }

    assert compute_checkpoint_score(lower_d3_metrics) < compute_checkpoint_score(higher_d3_metrics)


def test_parse_checkpoint_log_extracts_best_checkpoint_progression(tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "[2026-04-26 11:41:29] Persisted best checkpoint (epoch=500, paper_reproduction.checkpoint_score=17.640356, mean_best_delta_e=12.418426, median_best_delta_e=11.336943, d2_within_5nm=0.402200)",
                "[2026-04-26 11:41:29] Checkpoint evaluation complete at epoch 500 | paper_reproduction.checkpoint_score=17.640356 | new best checkpoint",
                "[2026-04-26 11:56:55] Checkpoint evaluation complete at epoch 1000 | paper_reproduction.checkpoint_score=18.294030",
                "[2026-04-26 12:12:17] Persisted best checkpoint (epoch=1500, paper_reproduction.checkpoint_score=16.716167, mean_best_delta_e=11.965445, median_best_delta_e=11.643469, d2_within_5nm=0.462600)",
                "[2026-04-26 12:12:17] Checkpoint evaluation complete at epoch 1500 | paper_reproduction.checkpoint_score=16.716167 | new best checkpoint",
                "[2026-04-26 13:13:11] Training complete in 1h47m10s | best_checkpoint_epoch=1500 | paper_reproduction.checkpoint_score=16.716167",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_checkpoint_log(log_path)
    checkpoints = parsed["checkpoints"]

    assert len(checkpoints) == 3
    assert checkpoints[0]["epoch"] == 500
    assert checkpoints[0]["is_best"] is True
    assert checkpoints[0]["mean_best_delta_e"] == pytest.approx(12.418426)
    assert checkpoints[1]["epoch"] == 1000
    assert checkpoints[1]["is_best"] is False
    assert checkpoints[2]["epoch"] == 1500
    assert checkpoints[2]["d2_within_5nm"] == pytest.approx(0.462600)
    assert parsed["training_complete"]["best_epoch"] == 1500


def test_parse_train_progress_extracts_alpha_and_losses(tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "[2026-04-26 11:26:07] Train 80/4000 | d_loss=2.123573 | g_loss=0.002031 | lab_mse=1.995021 | alpha=0.0002 | elapsed=6s | eta=5m15s",
                "[2026-04-26 11:26:14] Train 160/4000 | d_loss=2.065916 | g_loss=-0.000700 | lab_mse=1.864487 | alpha=0.0004 | elapsed=13s | eta=5m08s",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = parse_train_progress(log_path)

    assert len(rows) == 2
    assert rows[0]["epoch"] == pytest.approx(80.0)
    assert rows[0]["alpha"] == pytest.approx(0.0002)
    assert rows[1]["g_loss"] == pytest.approx(-0.000700)


def test_evaluate_testing_set_distribution_is_repeatable_for_fixed_seed() -> None:
    design_samples, lab_samples, _ = build_ag_sio2_ag_dataset(
        bottom_points=2,
        sio2_points=2,
        top_points=2,
    )
    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=1,
        batch_size=8,
        regressor_epochs=1,
        seed=47,
        record_losses=True,
    )

    metrics_a = evaluate_testing_set_distribution(
        bundle=bundle,
        test_designs=design_samples[:3],
        test_labs=lab_samples[:3],
        samples_per_lab=4,
        seed=101,
    )
    metrics_b = evaluate_testing_set_distribution(
        bundle=bundle,
        test_designs=design_samples[:3],
        test_labs=lab_samples[:3],
        samples_per_lab=4,
        seed=101,
    )

    assert metrics_a["mean_best_delta_e"] == pytest.approx(metrics_b["mean_best_delta_e"])
    assert metrics_a["median_best_delta_e"] == pytest.approx(metrics_b["median_best_delta_e"])
    assert metrics_a["d2_ground_truth_within_5nm_ratio"] == pytest.approx(
        metrics_b["d2_ground_truth_within_5nm_ratio"]
    )
    assert metrics_a["jsd"]["d1"] == pytest.approx(metrics_b["jsd"]["d1"])
    assert metrics_a["jsd"]["d2"] == pytest.approx(metrics_b["jsd"]["d2"])
    assert metrics_a["jsd"]["d3"] == pytest.approx(metrics_b["jsd"]["d3"])


def test_lightweight_cgan_stops_early_after_checkpoint_patience() -> None:
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
    checkpoint_epochs: list[int] = []
    progress_events: list[dict[str, object]] = []

    def checkpoint_metric_fn(bundle, epoch: int) -> float:
        checkpoint_epochs.append(epoch)
        return float(epoch)

    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=10,
        batch_size=4,
        regressor_epochs=1,
        seed=43,
        record_losses=True,
        checkpoint_metric_fn=checkpoint_metric_fn,
        checkpoint_metric_name="test.metric",
        checkpoint_metric_interval=2,
        checkpoint_patience=2,
        progress_callback=progress_events.append,
        progress_interval=10,
        regressor_progress_interval=1,
    )

    assert checkpoint_epochs == [2, 4, 6]
    assert len(bundle.losses) == 6
    assert bundle.selected_checkpoint_epoch == 2
    assert bundle.selected_checkpoint_metric_value == pytest.approx(2.0)
    assert any(
        event["stage"] == "training"
        and event["event"] == "early_stop"
        and event["epoch"] == 6
        for event in progress_events
    )

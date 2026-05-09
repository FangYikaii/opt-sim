from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import contextmanager
import csv
import json
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
import time
import zipfile
from typing import TYPE_CHECKING, Any, Literal, TextIO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    import numpy as np
    import torch
    from backend.app.algorithms.cgan import CganBundle

np: Any = None
torch: Any = None
CganBundle: Any = Any
fit_lightweight_cgan: Any = None
get_torch_runtime_info: Any = None
sample_designs_from_bundle: Any = None
sample_designs_for_labs_from_bundle: Any = None
load_model_bundle: Any = None
DBSCAN: Any = None
delta_e_76: Any = None
delta_e_2000: Any = None
hex_to_lab: Any = None
reflectance_spectrum_ag_sio2_ag: Any = None
spectrum_mode_name: Any = None
spectrum_to_xyz: Any = None
transmittance_spectrum_ag_sio2_ag: Any = None
xyz_to_lab: Any = None
xyz_to_srgb_hex: Any = None


DEFAULT_TARGETS = ["#4f86c6", "#d15f3f", "#66aa55"]
DEFAULT_PAPER_DATASET_ZIP = ROOT / "backend" / "data" / "paper-reproduction" / "dataset.zip"
DEFAULT_PAPER_TRAINING_CSV = ROOT / "backend" / "data" / "paper-reproduction" / "training set.csv"
DEFAULT_PAPER_TESTING_CSV = ROOT / "backend" / "data" / "paper-reproduction" / "testing set.csv"
RetrievalMetric = Literal["euclidean_lab", "delta_e_2000"]
ExperimentPresetName = Literal[
    "legacy_tune4_alpha",
    "legacy_tune4_alpha_conditional_d",
    "legacy_tune4_alpha_conditional_d_noise8",
    "legacy_tune4_alpha_conditional_d_noise8_mode_seeking_low",
]

DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEAN_BEST_DELTA_E = 1.0
DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEDIAN_BEST_DELTA_E = 0.02
DEFAULT_CHECKPOINT_SCORE_WEIGHT_D2_WITHIN_5NM = 8.0
DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEAN_JSD = 0.5
DEFAULT_CHECKPOINT_SCORE_WEIGHT_D3_JSD = 2.0
DEFAULT_CHECKPOINT_BUDGET_MODE = "match_final"
DEFAULT_LOW_MODE_SEEKING_WEIGHT = 0.02


@dataclass(frozen=True)
class ExperimentPreset:
    description: str
    overrides: dict[str, object]


EXPERIMENT_PRESETS: dict[str, ExperimentPreset] = {
    "legacy_tune4_alpha": ExperimentPreset(
        description="旧 tune4 alpha 基线，不带 conditional discriminator / noise_dim=8 / mode seeking。",
        overrides={
            "batch_size": 16384,
            "noise_dim": 2,
            "generator_hidden_dim": 128,
            "generator_depth": 9,
            "discriminator_hidden_dim": 128,
            "discriminator_depth": 4,
            "discriminator_conditioning": "none",
            "alpha_start": 0.0,
            "alpha_ramp_epochs": 40000,
            "max_alpha": 0.3,
            "lab_delta_e_weight": 0.0,
            "mode_seeking_weight": 0.0,
            "regressor_epochs": 10000,
            "generator_learning_rate": 1e-3,
            "discriminator_learning_rate": 2e-4,
            "steps_per_batch": 1,
        },
    ),
    "legacy_tune4_alpha_conditional_d": ExperimentPreset(
        description="只在旧 tune4 alpha 基线上打开 conditional discriminator。",
        overrides={
            "batch_size": 16384,
            "noise_dim": 2,
            "generator_hidden_dim": 128,
            "generator_depth": 9,
            "discriminator_hidden_dim": 128,
            "discriminator_depth": 4,
            "discriminator_conditioning": "target_lab",
            "alpha_start": 0.0,
            "alpha_ramp_epochs": 40000,
            "max_alpha": 0.3,
            "lab_delta_e_weight": 0.0,
            "mode_seeking_weight": 0.0,
            "regressor_epochs": 10000,
            "generator_learning_rate": 1e-3,
            "discriminator_learning_rate": 2e-4,
            "steps_per_batch": 1,
        },
    ),
    "legacy_tune4_alpha_conditional_d_noise8": ExperimentPreset(
        description="在 conditional discriminator + 旧 tune4 alpha 基线上单独把 noise_dim 提到 8。",
        overrides={
            "batch_size": 16384,
            "noise_dim": 8,
            "generator_hidden_dim": 128,
            "generator_depth": 9,
            "discriminator_hidden_dim": 128,
            "discriminator_depth": 4,
            "discriminator_conditioning": "target_lab",
            "alpha_start": 0.0,
            "alpha_ramp_epochs": 40000,
            "max_alpha": 0.3,
            "lab_delta_e_weight": 0.0,
            "mode_seeking_weight": 0.0,
            "regressor_epochs": 10000,
            "generator_learning_rate": 1e-3,
            "discriminator_learning_rate": 2e-4,
            "steps_per_batch": 1,
        },
    ),
    "legacy_tune4_alpha_conditional_d_noise8_mode_seeking_low": ExperimentPreset(
        description=(
            "在 conditional discriminator + 旧 tune4 alpha + noise_dim=8 基线上，"
            f"再低权重加入 mode seeking（默认 {DEFAULT_LOW_MODE_SEEKING_WEIGHT}）。"
        ),
        overrides={
            "batch_size": 16384,
            "noise_dim": 8,
            "generator_hidden_dim": 128,
            "generator_depth": 9,
            "discriminator_hidden_dim": 128,
            "discriminator_depth": 4,
            "discriminator_conditioning": "target_lab",
            "alpha_start": 0.0,
            "alpha_ramp_epochs": 40000,
            "max_alpha": 0.3,
            "lab_delta_e_weight": 0.0,
            "mode_seeking_weight": DEFAULT_LOW_MODE_SEEKING_WEIGHT,
            "regressor_epochs": 10000,
            "generator_learning_rate": 1e-3,
            "discriminator_learning_rate": 2e-4,
            "steps_per_batch": 1,
        },
    ),
}


def get_experiment_preset(name: str | None) -> ExperimentPreset | None:
    if name is None:
        return None
    return EXPERIMENT_PRESETS[name]


@dataclass(frozen=True)
class CandidateRecord:
    target_hex: str
    source: str
    rank: int
    d_ag_bottom_nm: float
    d_sio2_nm: float
    d_ag_top_nm: float
    delta_e: float
    simulated_hex: str
    retrieval_metric: str


def load_runtime_dependencies() -> None:
    global np
    global torch
    global CganBundle
    global fit_lightweight_cgan
    global get_torch_runtime_info
    global sample_designs_from_bundle
    global sample_designs_for_labs_from_bundle
    global load_model_bundle
    global DBSCAN
    global delta_e_76
    global delta_e_2000
    global hex_to_lab
    global reflectance_spectrum_ag_sio2_ag
    global spectrum_mode_name
    global spectrum_to_xyz
    global transmittance_spectrum_ag_sio2_ag
    global xyz_to_lab
    global xyz_to_srgb_hex

    try:
        import numpy as imported_np
        import torch as imported_torch
        from sklearn.cluster import DBSCAN as ImportedDBSCAN
        from backend.app.algorithms.cgan import (
            CganBundle as ImportedCganBundle,
            fit_lightweight_cgan as imported_fit_lightweight_cgan,
            get_torch_runtime_info as imported_get_torch_runtime_info,
            load_model_bundle as imported_load_model_bundle,
            sample_designs_for_labs_from_bundle as imported_sample_designs_for_labs_from_bundle,
            sample_designs_from_bundle as imported_sample_designs_from_bundle,
        )
        from backend.app.algorithms.optics import (
            delta_e_76 as imported_delta_e_76,
            delta_e_2000 as imported_delta_e_2000,
            hex_to_lab as imported_hex_to_lab,
            reflectance_spectrum_ag_sio2_ag as imported_reflectance_spectrum_ag_sio2_ag,
            transmittance_spectrum_ag_sio2_ag as imported_transmittance_spectrum_ag_sio2_ag,
            spectrum_to_xyz as imported_spectrum_to_xyz,
            xyz_to_lab as imported_xyz_to_lab,
            xyz_to_srgb_hex as imported_xyz_to_srgb_hex,
        )
    except ModuleNotFoundError as exc:
        missing_name = exc.name or "required dependency"
        raise SystemExit(
            f"Missing Python dependency '{missing_name}'. "
            "Install them with: python3 -m pip install -r backend/requirements.txt"
        ) from exc

    np = imported_np
    torch = imported_torch
    CganBundle = ImportedCganBundle
    fit_lightweight_cgan = imported_fit_lightweight_cgan
    get_torch_runtime_info = imported_get_torch_runtime_info
    sample_designs_for_labs_from_bundle = imported_sample_designs_for_labs_from_bundle
    sample_designs_from_bundle = imported_sample_designs_from_bundle
    load_model_bundle = imported_load_model_bundle
    DBSCAN = ImportedDBSCAN
    delta_e_76 = imported_delta_e_76
    delta_e_2000 = imported_delta_e_2000
    hex_to_lab = imported_hex_to_lab
    reflectance_spectrum_ag_sio2_ag = imported_reflectance_spectrum_ag_sio2_ag
    transmittance_spectrum_ag_sio2_ag = imported_transmittance_spectrum_ag_sio2_ag
    spectrum_mode_name = "transmittance"
    spectrum_to_xyz = imported_spectrum_to_xyz
    xyz_to_lab = imported_xyz_to_lab
    xyz_to_srgb_hex = imported_xyz_to_srgb_hex


def load_paper_dataset_csv(dataset_path: Path) -> tuple[np.ndarray, np.ndarray]:
    design_rows: list[list[float]] = []
    lab_rows: list[list[float]] = []

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"d1", "d2", "d3", "L", "a", "b"}
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Dataset file {dataset_path} is missing columns: {missing_str}")

        for row in reader:
            design_rows.append(
                [
                    float(row["d1"]) * 1000.0,
                    float(row["d2"]) * 1000.0,
                    float(row["d3"]) * 1000.0,
                ]
            )
            lab_rows.append([float(row["L"]), float(row["a"]), float(row["b"])])

    return np.array(design_rows, dtype=np.float64), np.array(lab_rows, dtype=np.float64)


def ensure_paper_dataset_files(
    *,
    dataset_zip: Path = DEFAULT_PAPER_DATASET_ZIP,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    train_csv = (output_dir or dataset_zip.parent) / "training set.csv"
    test_csv = (output_dir or dataset_zip.parent) / "testing set.csv"
    if train_csv.exists() and test_csv.exists():
        return train_csv, test_csv

    if not dataset_zip.exists():
        raise FileNotFoundError(
            f"Paper dataset archive not found at {dataset_zip}. "
            "Download the Southampton dataset ZIP before running paper reproduction."
        )

    destination = output_dir or dataset_zip.parent
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_zip) as archive:
        archive.extractall(destination)
    return train_csv, test_csv


def compute_jensen_shannon_distance(
    reference_values: np.ndarray,
    predicted_values: np.ndarray,
    *,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
) -> float:
    histogram_range = value_range
    if histogram_range is None:
        low = float(min(np.min(reference_values), np.min(predicted_values)))
        high = float(max(np.max(reference_values), np.max(predicted_values)))
        histogram_range = (low, high)

    ref_hist, _ = np.histogram(
        reference_values,
        bins=bins,
        range=histogram_range,
        density=False,
    )
    pred_hist, _ = np.histogram(
        predicted_values,
        bins=bins,
        range=histogram_range,
        density=False,
    )

    ref_prob = ref_hist.astype(np.float64)
    pred_prob = pred_hist.astype(np.float64)
    ref_prob /= max(float(ref_prob.sum()), 1.0)
    pred_prob /= max(float(pred_prob.sum()), 1.0)
    mean_prob = 0.5 * (ref_prob + pred_prob)

    def _kl_divergence(prob_a: np.ndarray, prob_b: np.ndarray) -> float:
        mask = prob_a > 0
        return float(np.sum(prob_a[mask] * np.log2(prob_a[mask] / np.maximum(prob_b[mask], 1e-12))))

    js_divergence = 0.5 * _kl_divergence(ref_prob, mean_prob) + 0.5 * _kl_divergence(pred_prob, mean_prob)
    return math.sqrt(max(js_divergence, 0.0))


def count_solution_groups_with_dbscan(
    predicted_designs_um: np.ndarray,
    *,
    eps: float = 0.03,
    min_samples: int = 10,
) -> int:
    if len(predicted_designs_um) == 0:
        return 0

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(predicted_designs_um)
    unique_clusters = {int(label) for label in labels if int(label) >= 0}
    return len(unique_clusters)


def build_ag_sio2_ag_dataset(
    *,
    bottom_points: int,
    sio2_points: int,
    top_points: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    designs: list[tuple[float, float, float]] = []
    labs: list[np.ndarray] = []
    hexes: list[str] = []

    for d_ag_bottom in np.linspace(10.0, 30.0, bottom_points):
        for d_sio2 in np.linspace(60.0, 180.0, sio2_points):
            for d_ag_top in np.linspace(10.0, 30.0, top_points):
                design = (float(d_ag_bottom), float(d_sio2), float(d_ag_top))
                spectrum = transmittance_spectrum_ag_sio2_ag(*design)
                xyz = spectrum_to_xyz(spectrum)
                designs.append(design)
                labs.append(xyz_to_lab(xyz))
                hexes.append(xyz_to_srgb_hex(xyz))

    return np.array(designs, dtype=np.float64), np.vstack(labs), hexes


def evaluate_design(design: np.ndarray, target_lab: np.ndarray) -> tuple[float, str]:
    spectrum = transmittance_spectrum_ag_sio2_ag(
        float(design[0]),
        float(design[1]),
        float(design[2]),
    )
    xyz = spectrum_to_xyz(spectrum)
    lab = xyz_to_lab(xyz)
    return delta_e_2000(lab, target_lab), xyz_to_srgb_hex(xyz)


def nearest_retrieval(
    lab_samples: np.ndarray,
    design_samples: np.ndarray,
    target_lab: np.ndarray,
) -> tuple[float, np.ndarray]:
    distances = np.linalg.norm(lab_samples - target_lab.reshape(1, -1), axis=1)
    index = int(np.argmin(distances))
    return float(distances[index]), design_samples[index]


def nearest_retrieval_delta_e_2000(
    lab_samples: np.ndarray,
    design_samples: np.ndarray,
    target_lab: np.ndarray,
) -> tuple[float, np.ndarray]:
    distances = np.array(
        [delta_e_2000(candidate_lab, target_lab) for candidate_lab in lab_samples],
        dtype=np.float64,
    )
    index = int(np.argmin(distances))
    return float(distances[index]), design_samples[index]


def nearest_retrieval_with_metric(
    lab_samples: np.ndarray,
    design_samples: np.ndarray,
    hex_samples: list[str] | None,
    target_lab: np.ndarray,
    *,
    retrieval_metric: RetrievalMetric,
) -> tuple[float, str, np.ndarray]:
    if retrieval_metric == "euclidean_lab":
        _, retrieved_design = nearest_retrieval(lab_samples, design_samples, target_lab)
    else:
        _, retrieved_design = nearest_retrieval_delta_e_2000(lab_samples, design_samples, target_lab)

    retrieval_delta_e, simulated_hex = evaluate_design(retrieved_design, target_lab)
    if hex_samples is not None:
        design_matches = np.all(np.isclose(design_samples, retrieved_design.reshape(1, -1)), axis=1)
        matching_indices = np.flatnonzero(design_matches)
        if matching_indices.size > 0:
            simulated_hex = hex_samples[int(matching_indices[0])]
    return float(retrieval_delta_e), simulated_hex, retrieved_design


def collect_candidate_records(
    *,
    bundle: CganBundle,
    lab_samples: np.ndarray,
    design_samples: np.ndarray,
    hex_samples: list[str] | None,
    targets: list[str],
    sample_count: int,
    top_generated: int,
    seed: int,
    retrieval_metric: RetrievalMetric = "euclidean_lab",
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []

    for target_index, target_hex in enumerate(targets):
        target_lab = hex_to_lab(target_hex)
        retrieval_delta_e, retrieval_hex, retrieval_design = nearest_retrieval_with_metric(
            lab_samples,
            design_samples,
            hex_samples,
            target_lab,
            retrieval_metric=retrieval_metric,
        )
        records.append(
            CandidateRecord(
                target_hex=target_hex.lower(),
                source="retrieval",
                rank=1,
                d_ag_bottom_nm=float(retrieval_design[0]),
                d_sio2_nm=float(retrieval_design[1]),
                d_ag_top_nm=float(retrieval_design[2]),
                delta_e=retrieval_delta_e,
                simulated_hex=retrieval_hex,
                retrieval_metric=retrieval_metric,
            )
        )

        generated_designs = sample_designs_from_bundle(
            bundle,
            target_lab,
            sample_count=sample_count,
            seed=seed + target_index,
        )
        generated_rows = []
        for design in generated_designs:
            delta_e, simulated_hex = evaluate_design(design, target_lab)
            generated_rows.append((delta_e, simulated_hex, design))
        generated_rows.sort(key=lambda row: row[0])

        for rank, (delta_e, simulated_hex, design) in enumerate(
            generated_rows[:top_generated],
            start=1,
        ):
            records.append(
                CandidateRecord(
                    target_hex=target_hex.lower(),
                    source="cgan",
                    rank=rank,
                    d_ag_bottom_nm=float(design[0]),
                    d_sio2_nm=float(design[1]),
                    d_ag_top_nm=float(design[2]),
                    delta_e=float(delta_e),
                    simulated_hex=simulated_hex,
                    retrieval_metric=retrieval_metric,
                )
            )

    return records


def evaluate_testing_set_distribution(
    *,
    bundle: CganBundle,
    test_designs: np.ndarray,
    test_labs: np.ndarray,
    samples_per_lab: int,
    seed: int,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval: int | None = None,
    progress_phase: str = "final",
) -> dict[str, object]:
    update_interval = max(progress_interval or 0, 0)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "paper_eval",
                "event": "start",
                "phase": progress_phase,
                "processed": 0,
                "total": int(len(test_labs)),
            }
        )

    generated_design_batches = sample_designs_for_labs_from_bundle(
        bundle,
        test_labs,
        sample_count=samples_per_lab,
        seed=seed,
    )
    best_delta_es: list[float] = []
    solution_group_counts: list[int] = []
    abs_d2_errors_nm: list[float] = []

    for index, generated_nm in enumerate(generated_design_batches):
        target_lab = test_labs[index]
        generated_delta_es = [float(evaluate_design(design, target_lab)[0]) for design in generated_nm]
        best_delta_es.append(float(min(generated_delta_es)))
        solution_group_counts.append(
            count_solution_groups_with_dbscan(
                generated_nm / 1000.0,
                eps=0.03,
                min_samples=min(10, max(2, len(generated_nm) // 20)),
            )
        )
        d2_errors = np.abs(generated_nm[:, 1] - test_designs[index, 1])
        abs_d2_errors_nm.append(float(np.min(d2_errors)))

        processed = index + 1
        if (
            progress_callback is not None
            and (
                processed == len(test_labs)
                or (update_interval > 0 and processed % update_interval == 0)
            )
        ):
            progress_callback(
                {
                    "stage": "paper_eval",
                    "event": "progress",
                    "phase": progress_phase,
                    "processed": processed,
                    "total": int(len(test_labs)),
                }
            )

    all_generated_designs = generated_design_batches.reshape(-1, generated_design_batches.shape[-1])
    jsd = {
        "d1": compute_jensen_shannon_distance(
            test_designs[:, 0],
            all_generated_designs[:, 0],
            bins=100,
            value_range=(0.0, 50.0),
        ),
        "d2": compute_jensen_shannon_distance(
            test_designs[:, 1],
            all_generated_designs[:, 1],
            bins=100,
            value_range=(0.0, 1000.0),
        ),
        "d3": compute_jensen_shannon_distance(
            test_designs[:, 2],
            all_generated_designs[:, 2],
            bins=100,
            value_range=(0.0, 50.0),
        ),
    }

    d2_within_5nm = float(np.mean(np.array(abs_d2_errors_nm, dtype=np.float64) <= 5.0))

    metrics = {
        "samples_per_lab": samples_per_lab,
        "mean_best_delta_e": float(np.mean(best_delta_es)),
        "median_best_delta_e": float(np.median(best_delta_es)),
        "mean_solution_groups": float(np.mean(solution_group_counts)),
        "max_solution_groups": int(max(solution_group_counts, default=0)),
        "d2_ground_truth_within_5nm_ratio": d2_within_5nm,
        "generated_designs_nm": generated_design_batches,
        "best_delta_e_values": best_delta_es,
        "solution_group_counts": solution_group_counts,
        "abs_d2_errors_nm": abs_d2_errors_nm,
        "jsd": jsd,
    }
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "paper_eval",
                "event": "complete",
                "phase": progress_phase,
                "processed": int(len(test_labs)),
                "total": int(len(test_labs)),
            }
        )
    return metrics


def evaluate_saved_checkpoint(
    *,
    checkpoint_path: Path,
    paper_test_csv: Path,
    samples_per_lab: int,
    seed: int,
    device: str = "auto",
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval: int | None = None,
    progress_phase: str = "final",
) -> tuple[CganBundle, dict[str, object]]:
    if load_model_bundle is None or np is None:
        load_runtime_dependencies()

    bundle = load_model_bundle(checkpoint_path, device=device)
    test_designs, test_labs = load_paper_dataset_csv(paper_test_csv)
    metrics = evaluate_testing_set_distribution(
        bundle=bundle,
        test_designs=test_designs,
        test_labs=test_labs,
        samples_per_lab=samples_per_lab,
        seed=seed,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
        progress_phase=progress_phase,
    )
    return bundle, metrics


def compute_checkpoint_score(
    metrics: dict[str, object],
    *,
    weight_mean_best_delta_e: float = DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEAN_BEST_DELTA_E,
    weight_median_best_delta_e: float = DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEDIAN_BEST_DELTA_E,
    weight_d2_within_5nm: float = DEFAULT_CHECKPOINT_SCORE_WEIGHT_D2_WITHIN_5NM,
    weight_mean_jsd: float = DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEAN_JSD,
    weight_d3_jsd: float = DEFAULT_CHECKPOINT_SCORE_WEIGHT_D3_JSD,
) -> float:
    mean_best_delta_e = float(metrics.get("mean_best_delta_e", float("inf")))
    median_best_delta_e = float(metrics.get("median_best_delta_e", float("inf")))
    d2_within_5nm_ratio = float(metrics.get("d2_ground_truth_within_5nm_ratio", 0.0))
    jsd_metrics = metrics.get("jsd", {}) or {}
    d3_jsd = float(jsd_metrics.get("d3", 1.0))
    mean_jsd = (
        float(jsd_metrics.get("d1", 1.0))
        + float(jsd_metrics.get("d2", 1.0))
        + d3_jsd
    ) / 3.0

    # Lower is better. Rank primarily by retrieval quality and d2 recovery, then use
    # whole-distribution alignment and d3 drift as lighter tie-breakers.
    return (
        (weight_mean_best_delta_e * mean_best_delta_e)
        + (weight_median_best_delta_e * median_best_delta_e)
        + (weight_d2_within_5nm * (1.0 - d2_within_5nm_ratio))
        + (weight_mean_jsd * mean_jsd)
        + (weight_d3_jsd * d3_jsd)
    )


def _configure_streams_for_realtime_logs() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


class _TeeStream:
    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        written = self._primary.write(data)
        self._secondary.write(data)
        return written

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def reconfigure(self, **kwargs: object) -> None:
        for stream in (self._primary, self._secondary):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                reconfigure(**kwargs)

    def isatty(self) -> bool:
        isatty = getattr(self._primary, "isatty", None)
        return bool(isatty()) if callable(isatty) else False

    def fileno(self) -> int:
        fileno = getattr(self._primary, "fileno", None)
        if callable(fileno):
            return int(fileno())
        raise OSError("stream does not use a file descriptor")

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8")

    def __getattr__(self, name: str) -> object:
        return getattr(self._primary, name)


@contextmanager
def _stream_logs_to_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = _TeeStream(original_stdout, log_file)
        sys.stderr = _TeeStream(original_stderr, log_file)
        _configure_streams_for_realtime_logs()
        try:
            yield log_path
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _default_progress_interval(total_steps: int, target_updates: int, max_interval: int) -> int:
    if total_steps <= 0:
        return 1
    return max(1, min(max_interval, math.ceil(total_steps / max(target_updates, 1))))


def _build_progress_logger() -> Callable[[dict[str, object]], None]:
    started_at: dict[tuple[str, str | None], float] = {}

    def _elapsed(stage: str, phase: str | None = None) -> float | None:
        key = (stage, phase)
        start_time = started_at.get(key)
        if start_time is None:
            return None
        return time.monotonic() - start_time

    def _progress_eta(completed: int, total: int, elapsed: float | None) -> float | None:
        if elapsed is None or elapsed <= 0 or completed <= 0 or total <= completed:
            return None
        return elapsed * (total - completed) / completed

    def callback(event: dict[str, object]) -> None:
        stage = str(event.get("stage", "unknown"))
        action = str(event.get("event", "progress"))
        phase = str(event["phase"]) if event.get("phase") is not None else None
        key = (stage, phase)

        if action == "start":
            started_at[key] = time.monotonic()
            if stage == "regressor":
                _log(
                    "Regressor pretraining started "
                    f"(epochs={int(event['total_epochs'])}, batch_size={int(event['batch_size'])}, "
                    f"samples={int(event['sample_count'])})"
                )
            elif stage == "training":
                _log(
                    "Adversarial training started "
                    f"(epochs={int(event['total_epochs'])}, batch_size={int(event['batch_size'])}, "
                    f"samples={int(event['sample_count'])}, device={event['device']})"
                )
            elif stage == "checkpoint":
                metric_name = event.get("metric_name") or "checkpoint metric"
                _log(f"Checkpoint evaluation started at epoch {int(event['epoch'])} ({metric_name})")
            elif stage == "paper_eval":
                label = "final paper evaluation" if phase == "final" else f"{phase} paper evaluation"
                _log(f"{label.capitalize()} started ({int(event['total'])} targets)")
            return

        if action == "progress":
            if stage == "regressor":
                epoch = int(event["epoch"])
                total_epochs = int(event["total_epochs"])
                elapsed = _elapsed("regressor")
                eta = _progress_eta(epoch, total_epochs, elapsed)
                _log(
                    "Regressor "
                    f"{epoch}/{total_epochs} | loss={float(event['loss']):.6f} | "
                    f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}"
                )
            elif stage == "training":
                epoch = int(event["epoch"])
                total_epochs = int(event["total_epochs"])
                elapsed = _elapsed("training")
                eta = _progress_eta(epoch, total_epochs, elapsed)
                _log(
                    "Train "
                    f"{epoch}/{total_epochs} | d_loss={float(event['evaluator_loss']):.6f} | "
                    f"g_loss={float(event['generator_loss']):.6f} | lab_mse={float(event['lab_mse']):.6f} | "
                    f"alpha={float(event['alpha']):.4f} | elapsed={_format_duration(elapsed)} | "
                    f"eta={_format_duration(eta)}"
                )
            elif stage == "paper_eval":
                processed = int(event["processed"])
                total = int(event["total"])
                elapsed = _elapsed("paper_eval", phase)
                eta = _progress_eta(processed, total, elapsed)
                label = "final paper evaluation" if phase == "final" else f"{phase} paper evaluation"
                _log(
                    f"{label.capitalize()} {processed}/{total} | "
                    f"elapsed={_format_duration(elapsed)} | eta={_format_duration(eta)}"
                )
            return

        if action == "early_stop" and stage == "training":
            metric_name = event.get("selected_checkpoint_metric_name")
            metric_value = event.get("selected_checkpoint_metric_value")
            best_epoch = event.get("selected_checkpoint_epoch")
            summary = (
                "Early stopping triggered "
                f"at epoch {int(event['epoch'])} after "
                f"{int(event['checkpoint_evals_without_improvement'])} checkpoint evaluations without improvement"
            )
            if best_epoch is not None:
                summary += f" | best_checkpoint_epoch={int(best_epoch)}"
            if metric_name is not None and metric_value is not None:
                summary += f" | {metric_name}={float(metric_value):.6f}"
            _log(summary)
            return

        if action == "complete":
            elapsed = _elapsed(stage, phase)
            if stage == "regressor":
                _log(f"Regressor pretraining complete in {_format_duration(elapsed)}")
            elif stage == "checkpoint":
                metric_name = event.get("metric_name") or "checkpoint metric"
                metric_value = event.get("metric_value")
                metric_suffix = (
                    f"{metric_name}={float(metric_value):.6f}"
                    if metric_value is not None
                    else f"{metric_name}=unavailable"
                )
                best_suffix = ""
                if event.get("is_best"):
                    best_suffix = " | new best checkpoint"
                _log(
                    f"Checkpoint evaluation complete at epoch {int(event['epoch'])} | "
                    f"{metric_suffix}{best_suffix}"
                )
            elif stage == "training":
                best_epoch = event.get("selected_checkpoint_epoch")
                metric_name = event.get("selected_checkpoint_metric_name")
                metric_value = event.get("selected_checkpoint_metric_value")
                summary = f"Training complete in {_format_duration(elapsed)}"
                if best_epoch is not None:
                    summary += f" | best_checkpoint_epoch={int(best_epoch)}"
                if metric_name is not None and metric_value is not None:
                    summary += f" | {metric_name}={float(metric_value):.6f}"
                _log(summary)
            elif stage == "paper_eval":
                label = "final paper evaluation" if phase == "final" else f"{phase} paper evaluation"
                _log(f"{label.capitalize()} complete in {_format_duration(elapsed)}")

    return callback


def _write_checkpoint_state(
    *,
    output_dir: Path,
    epoch: int,
    metric_name: str | None,
    metric_value: float | None,
) -> None:
    state = {
        "best_checkpoint_epoch": epoch,
        "best_checkpoint_metric_name": metric_name,
        "best_checkpoint_metric_value": metric_value,
    }
    (output_dir / "checkpoint_state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_checkpoint_evaluation_budgets(args: argparse.Namespace) -> dict[str, int]:
    final_samples_per_lab = int(max(1, args.paper_samples_per_lab))
    checkpoint_samples_per_lab = int(max(1, args.checkpoint_samples_per_lab))
    checkpoint_recheck_samples_per_lab = int(
        max(
            checkpoint_samples_per_lab,
            getattr(args, "checkpoint_recheck_samples_per_lab", checkpoint_samples_per_lab),
        )
    )
    mode = str(getattr(args, "checkpoint_budget_mode", DEFAULT_CHECKPOINT_BUDGET_MODE))

    if mode == "match_final":
        checkpoint_samples_per_lab = final_samples_per_lab
        checkpoint_recheck_samples_per_lab = final_samples_per_lab
    elif mode == "recheck_best":
        checkpoint_samples_per_lab = min(checkpoint_samples_per_lab, final_samples_per_lab)
        checkpoint_recheck_samples_per_lab = min(checkpoint_recheck_samples_per_lab, final_samples_per_lab)
    else:
        checkpoint_samples_per_lab = min(checkpoint_samples_per_lab, final_samples_per_lab)
        checkpoint_recheck_samples_per_lab = checkpoint_samples_per_lab

    return {
        "checkpoint_samples_per_lab": checkpoint_samples_per_lab,
        "checkpoint_recheck_samples_per_lab": checkpoint_recheck_samples_per_lab,
        "final_samples_per_lab": final_samples_per_lab,
    }


def compact_reproduction_metrics_for_json(reproduction_metrics: dict[str, object] | None) -> dict[str, object] | None:
    if reproduction_metrics is None:
        return None

    compact_metrics = dict(reproduction_metrics)
    generated_designs = compact_metrics.pop("generated_designs_nm", None)
    best_delta_e_values = compact_metrics.pop("best_delta_e_values", None)
    solution_group_counts = compact_metrics.pop("solution_group_counts", None)
    abs_d2_errors_nm = compact_metrics.pop("abs_d2_errors_nm", None)

    if generated_designs is not None:
        generated_designs_array = np.asarray(generated_designs)
        compact_metrics["generated_designs_shape"] = [
            int(generated_designs_array.shape[0]) if generated_designs_array.ndim >= 1 else 0,
            int(generated_designs_array.shape[1]) if generated_designs_array.ndim >= 2 else 0,
            int(generated_designs_array.shape[2]) if generated_designs_array.ndim >= 3 else 0,
        ]
        compact_metrics["details_file"] = "paper_reproduction_details.npz"

    if best_delta_e_values is not None:
        compact_metrics["best_delta_e_count"] = len(best_delta_e_values)
    if solution_group_counts is not None:
        compact_metrics["solution_group_count"] = len(solution_group_counts)
    if abs_d2_errors_nm is not None:
        compact_metrics["abs_d2_error_count"] = len(abs_d2_errors_nm)

    return compact_metrics


def write_paper_reproduction_details(
    reproduction_metrics: dict[str, object] | None,
    output_path: Path,
) -> None:
    if reproduction_metrics is None:
        return

    np.savez_compressed(
        output_path,
        generated_designs_nm=np.array(reproduction_metrics["generated_designs_nm"], dtype=np.float32),
        best_delta_e_values=np.array(reproduction_metrics["best_delta_e_values"], dtype=np.float32),
        solution_group_counts=np.array(reproduction_metrics["solution_group_counts"], dtype=np.int32),
        abs_d2_errors_nm=np.array(reproduction_metrics["abs_d2_errors_nm"], dtype=np.float32),
    )


def write_loss_csv(bundle: CganBundle, output_path: Path) -> None:
    fieldnames = [
        "epoch",
        "discriminator_loss",
        "generator_loss",
        "evaluator_loss",
        "real_score",
        "fake_score",
        "lab_mse",
        "lab_delta_e76",
        "color_loss",
        "mode_seeking_loss",
        "alpha",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in bundle.losses:
            writer.writerow({fieldname: row.get(fieldname) for fieldname in fieldnames})


def write_candidate_csv(records: list[CandidateRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        first_row = records[0] if records else None
        if first_row is None:
            fieldnames: list[str] = []
        elif hasattr(first_row, "__dataclass_fields__"):
            fieldnames = list(asdict(first_row).keys())
        else:
            fieldnames = list(vars(first_row).keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if hasattr(record, "__dataclass_fields__"):
                writer.writerow(asdict(record))
            else:
                writer.writerow(dict(vars(record)))


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=np.float64)))


def write_retrieval_comparison_json(
    *,
    comparison_records_by_metric: dict[str, list[CandidateRecord]],
    output_path: Path,
    elapsed_seconds_by_metric: dict[str, float] | None = None,
) -> None:
    metrics: dict[str, object] = {}
    for retrieval_metric, records in comparison_records_by_metric.items():
        target_hexes = sorted({record.target_hex for record in records})
        retrieval_best_values: list[float] = []
        cgan_best_values: list[float] = []
        retrieval_wins_vs_cgan = 0
        cgan_wins_vs_retrieval = 0
        ties = 0

        for target_hex in target_hexes:
            target_records = [record for record in records if record.target_hex == target_hex]
            retrieval = _best_record(target_records, target_hex, "retrieval")
            cgan = _best_record(target_records, target_hex, "cgan")
            retrieval_best_values.append(float(retrieval.delta_e))
            cgan_best_values.append(float(cgan.delta_e))
            if retrieval.delta_e < cgan.delta_e:
                retrieval_wins_vs_cgan += 1
            elif retrieval.delta_e > cgan.delta_e:
                cgan_wins_vs_retrieval += 1
            else:
                ties += 1

        metrics[retrieval_metric] = {
            "targets_compared": len(target_hexes),
            "mean_retrieval_best_delta_e": _mean_or_none(retrieval_best_values),
            "mean_cgan_best_delta_e": _mean_or_none(cgan_best_values),
            "retrieval_wins_vs_cgan": retrieval_wins_vs_cgan,
            "cgan_wins_vs_retrieval": cgan_wins_vs_retrieval,
            "ties": ties,
            "elapsed_seconds": (
                float(elapsed_seconds_by_metric[retrieval_metric])
                if elapsed_seconds_by_metric and retrieval_metric in elapsed_seconds_by_metric
                else None
            ),
        }

    output_path.write_text(
        json.dumps({"metrics": metrics}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_model_bundle(bundle: CganBundle, output_path: Path) -> None:
    generator_state_dict = bundle.generator.state_dict()
    torch.save(
        {
            "checkpoint_format_version": bundle.checkpoint_format_version,
            "generator_state_dict": generator_state_dict,
            "lab_regressor_state_dict": bundle.lab_regressor.state_dict(),
            "lab_scaling_type": bundle.lab_scaling_type,
            "lab_mean": bundle.lab_mean.tolist(),
            "lab_std": bundle.lab_std.tolist(),
            "design_scaling_type": bundle.design_scaling_type,
            "design_min": bundle.design_min.tolist(),
            "design_max": bundle.design_max.tolist(),
            "device": bundle.device,
            "noise_dim": bundle.noise_dim,
            "seed": bundle.seed,
            "generator_learning_rate": bundle.generator_learning_rate,
            "discriminator_learning_rate": bundle.discriminator_learning_rate,
            "steps_per_batch": bundle.steps_per_batch,
            "generator_hidden_dim": bundle.generator_hidden_dim,
            "generator_depth": bundle.generator_depth,
            "noise_hidden_dim": getattr(bundle.generator, "noise_hidden_dim", bundle.generator_hidden_dim),
            "lab_hidden_dim": getattr(bundle.generator, "lab_hidden_dim", bundle.generator_hidden_dim),
            "regressor_hidden_dims": list(
                getattr(bundle.generator, "regressor_hidden_dims", [bundle.generator_hidden_dim] * bundle.generator_depth)
            ),
            "discriminator_hidden_dim": bundle.discriminator_hidden_dim,
            "discriminator_depth": bundle.discriminator_depth,
            "discriminator_conditioning": bundle.discriminator_conditioning,
            "alpha_start": bundle.alpha_start,
            "alpha_ramp_epochs": bundle.alpha_ramp_epochs,
            "max_alpha": bundle.max_alpha,
            "lab_delta_e_weight": bundle.lab_delta_e_weight,
            "mode_seeking_weight": bundle.mode_seeking_weight,
            "selected_checkpoint_epoch": bundle.selected_checkpoint_epoch,
            "selected_checkpoint_metric_name": bundle.selected_checkpoint_metric_name,
            "selected_checkpoint_metric_value": bundle.selected_checkpoint_metric_value,
        },
        output_path,
    )


def save_best_model_bundle(bundle: CganBundle, output_path: Path) -> None:
    generator_state_dict = (
        bundle.best_generator_state_dict
        if bundle.best_generator_state_dict is not None
        else bundle.generator.state_dict()
    )
    torch.save(
        {
            "checkpoint_format_version": bundle.checkpoint_format_version,
            "generator_state_dict": generator_state_dict,
            "lab_regressor_state_dict": bundle.lab_regressor.state_dict(),
            "lab_scaling_type": bundle.lab_scaling_type,
            "lab_mean": bundle.lab_mean.tolist(),
            "lab_std": bundle.lab_std.tolist(),
            "design_scaling_type": bundle.design_scaling_type,
            "design_min": bundle.design_min.tolist(),
            "design_max": bundle.design_max.tolist(),
            "device": bundle.device,
            "noise_dim": bundle.noise_dim,
            "seed": bundle.seed,
            "generator_learning_rate": bundle.generator_learning_rate,
            "discriminator_learning_rate": bundle.discriminator_learning_rate,
            "steps_per_batch": bundle.steps_per_batch,
            "generator_hidden_dim": bundle.generator_hidden_dim,
            "generator_depth": bundle.generator_depth,
            "noise_hidden_dim": getattr(bundle.generator, "noise_hidden_dim", bundle.generator_hidden_dim),
            "lab_hidden_dim": getattr(bundle.generator, "lab_hidden_dim", bundle.generator_hidden_dim),
            "regressor_hidden_dims": list(
                getattr(bundle.generator, "regressor_hidden_dims", [bundle.generator_hidden_dim] * bundle.generator_depth)
            ),
            "discriminator_hidden_dim": bundle.discriminator_hidden_dim,
            "discriminator_depth": bundle.discriminator_depth,
            "discriminator_conditioning": bundle.discriminator_conditioning,
            "alpha_start": bundle.alpha_start,
            "alpha_ramp_epochs": bundle.alpha_ramp_epochs,
            "max_alpha": bundle.max_alpha,
            "lab_delta_e_weight": bundle.lab_delta_e_weight,
            "mode_seeking_weight": bundle.mode_seeking_weight,
            "selected_checkpoint_epoch": bundle.selected_checkpoint_epoch,
            "selected_checkpoint_metric_name": bundle.selected_checkpoint_metric_name,
            "selected_checkpoint_metric_value": bundle.selected_checkpoint_metric_value,
        },
        output_path,
    )


def write_artifact_manifest(
    *,
    output_dir: Path,
    selected_checkpoint_name: str,
    final_checkpoint_name: str,
    best_available_checkpoint_name: str | None = None,
) -> dict[str, object]:
    return {
        "selected_checkpoint": selected_checkpoint_name,
        "final_checkpoint": final_checkpoint_name,
        "best_available_checkpoint": best_available_checkpoint_name or selected_checkpoint_name,
        "available_checkpoints": [
            checkpoint_name
            for checkpoint_name in [selected_checkpoint_name, final_checkpoint_name]
            if (output_dir / checkpoint_name).exists()
        ],
    }


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_loss_curve(bundle: CganBundle, output_path: Path) -> None:
    plt = import_pyplot()
    epochs = [row["epoch"] for row in bundle.losses]
    d_losses = [row.get("evaluator_loss", row["discriminator_loss"]) for row in bundle.losses]
    g_losses = [row["generator_loss"] for row in bundle.losses]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(epochs, d_losses, label="Discriminator", color="#2f4858", linewidth=2.0)
    ax.plot(epochs, g_losses, label="Generator", color="#f26419", linewidth=2.0)
    ax.set_title("Lightweight cGAN Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hinge objective")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _best_record(records: list[CandidateRecord], target_hex: str, source: str) -> CandidateRecord:
    candidates = [
        record
        for record in records
        if record.target_hex == target_hex.lower() and record.source == source
    ]
    return min(candidates, key=lambda record: record.delta_e)


def plot_sampling_comparison(
    records: list[CandidateRecord],
    targets: list[str],
    output_path: Path,
) -> None:
    plt = import_pyplot()
    fig, axes = plt.subplots(len(targets), 3, figsize=(9.5, 2.7 * len(targets)))
    axes_array = np.atleast_2d(axes)
    headers = ["Target", "Nearest retrieval", "Best cGAN sample"]

    for row_index, target_hex in enumerate(targets):
        retrieval = _best_record(records, target_hex, "retrieval")
        cgan = _best_record(records, target_hex, "cgan")
        swatches = [
            (target_hex.lower(), "requested color"),
            (retrieval.simulated_hex, f"DeltaE {retrieval.delta_e:.2f}"),
            (cgan.simulated_hex, f"DeltaE {cgan.delta_e:.2f}"),
        ]
        for col_index, (hex_color, subtitle) in enumerate(swatches):
            ax = axes_array[row_index, col_index]
            ax.imshow(np.ones((32, 32, 3)) * _hex_to_rgb01(hex_color))
            title = headers[col_index] if row_index == 0 else ""
            ax.set_title(title, fontsize=11, pad=8)
            ax.set_xlabel(f"{hex_color}\n{subtitle}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Target vs. Retrieved vs. Generated Color Samples", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_candidate_diversity(
    records: list[CandidateRecord],
    target_hex: str,
    output_path: Path,
) -> None:
    plt = import_pyplot()
    cgan_records = [
        record
        for record in records
        if record.target_hex == target_hex.lower() and record.source == "cgan"
    ]
    retrieval = _best_record(records, target_hex, "retrieval")
    params = np.array(
        [
            [record.d_ag_bottom_nm, record.d_sio2_nm, record.d_ag_top_nm]
            for record in cgan_records
        ],
        dtype=np.float64,
    )
    errors = np.array([record.delta_e for record in cgan_records], dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    projections = [
        (0, 1, "Ag bottom (nm)", "SiO2 (nm)"),
        (2, 1, "Ag top (nm)", "SiO2 (nm)"),
        (0, 2, "Ag bottom (nm)", "Ag top (nm)"),
    ]
    for ax, (x_index, y_index, x_label, y_label) in zip(axes, projections):
        scatter = ax.scatter(
            params[:, x_index],
            params[:, y_index],
            c=errors,
            cmap="viridis_r",
            s=54,
            alpha=0.9,
            edgecolors="#172026",
            linewidths=0.4,
            label="cGAN candidates",
        )
        ax.scatter(
            [getattr(retrieval, _field_for_index(x_index))],
            [getattr(retrieval, _field_for_index(y_index))],
            marker="*",
            s=180,
            color="#f26419",
            edgecolors="#172026",
            linewidths=0.6,
            label="retrieval best",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.22)
    axes[0].legend(loc="best", fontsize=8)
    fig.colorbar(scatter, ax=axes, shrink=0.82, label="DeltaE")
    fig.suptitle(f"Candidate Diversity for {target_hex.lower()}")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_paper_distribution_comparison(
    *,
    test_designs: np.ndarray,
    generated_designs: np.ndarray,
    output_path: Path,
) -> None:
    plt = import_pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey="row")
    labels = [
        (0, "d1", "Ag bottom (nm)", (0.0, 50.0)),
        (1, "d2", "SiO2 (nm)", (0.0, 1000.0)),
        (2, "d3", "Ag top (nm)", (0.0, 50.0)),
    ]

    for col, (_, key, xlabel, value_range) in enumerate(labels):
        axes[0, col].hist(test_designs[:, col], bins=100, range=value_range, color="#2f4858")
        axes[0, col].set_title(f"Testing Set {key}")
        axes[0, col].set_xlabel(xlabel)
        axes[0, col].grid(True, alpha=0.2)

        axes[1, col].hist(generated_designs[:, col], bins=100, range=value_range, color="#f26419")
        axes[1, col].set_title(f"Generated {key}")
        axes[1, col].set_xlabel(xlabel)
        axes[1, col].grid(True, alpha=0.2)

    axes[0, 0].set_ylabel("Count")
    axes[1, 0].set_ylabel("Count")
    fig.suptitle("Paper Figure 4 Style Thickness Distribution Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_paper_solution_metrics(
    *,
    best_delta_e_values: np.ndarray,
    solution_group_counts: np.ndarray,
    output_path: Path,
) -> None:
    plt = import_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].hist(solution_group_counts, bins=np.arange(0.5, max(solution_group_counts.max(), 1) + 1.5, 1.0), color="#33658a")
    axes[0].set_title("Paper Figure 5 Style Solution Groups")
    axes[0].set_xlabel("DBSCAN Solution Groups")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.2)

    axes[1].hist(best_delta_e_values, bins=60, color="#86bbd8")
    axes[1].set_title("Paper Figure 5 Style Best DeltaE Distribution")
    axes[1].set_xlabel("Best DeltaE2000")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_paper_d2_accuracy(
    *,
    abs_d2_errors_nm: np.ndarray,
    output_path: Path,
) -> None:
    plt = import_pyplot()
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.hist(abs_d2_errors_nm, bins=80, color="#758e4f")
    for value, color, label in [(1.0, "#c1121f", "1 nm"), (5.0, "#ffb703", "5 nm"), (10.0, "#2a9d8f", "10 nm")]:
        ax.axvline(value, color=color, linestyle="--", linewidth=1.5, label=label)
    ax.set_title("Paper Figure S6 Style |Delta d2| Distribution")
    ax.set_xlabel("|Delta d2| (nm)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _field_for_index(index: int) -> str:
    return ["d_ag_bottom_nm", "d_sio2_nm", "d_ag_top_nm"][index]


def summarize_target(records: list[CandidateRecord], target_hex: str) -> dict[str, object]:
    target_records = [record for record in records if record.target_hex == target_hex.lower()]
    retrieval = _best_record(target_records, target_hex, "retrieval")
    cgan_records = [record for record in target_records if record.source == "cgan"]
    cgan_best = min(cgan_records, key=lambda record: record.delta_e)
    params = np.array(
        [
            [record.d_ag_bottom_nm, record.d_sio2_nm, record.d_ag_top_nm]
            for record in cgan_records
        ],
        dtype=np.float64,
    )
    rounded_unique = np.unique(np.round(params, decimals=1), axis=0)

    if len(params) > 1:
        distances = np.linalg.norm(params[:, None, :] - params[None, :, :], axis=2)
        upper = distances[np.triu_indices(len(params), k=1)]
        mean_pairwise_distance = float(np.mean(upper))
    else:
        mean_pairwise_distance = 0.0

    return {
        "target_hex": target_hex.lower(),
        "retrieval_best_delta_e": retrieval.delta_e,
        "cgan_best_delta_e": cgan_best.delta_e,
        "cgan_unique_rounded_0p1nm": int(len(rounded_unique)),
        "cgan_mean_pairwise_distance_nm": mean_pairwise_distance,
        "cgan_parameter_span_nm": {
            "ag_bottom": float(np.ptp(params[:, 0])) if len(params) else 0.0,
            "sio2": float(np.ptp(params[:, 1])) if len(params) else 0.0,
            "ag_top": float(np.ptp(params[:, 2])) if len(params) else 0.0,
        },
    }


def write_metrics(
    *,
    bundle: CganBundle,
    records: list[CandidateRecord],
    targets: list[str],
    dataset_size: int,
    args: argparse.Namespace,
    output_path: Path,
    reproduction_metrics: dict[str, object] | None = None,
    design_bounds_nm: dict[str, list[float]] | None = None,
    artifact_manifest: dict[str, object] | None = None,
) -> None:
    metrics = {
        "colorimetry": {
            "illuminant_source": "refer_data/D65.csv",
            "tristimulus_source": "refer_data/tristimulus.csv",
            "wavelength_grid_nm": [int(value) for value in range(380, 781, 5)],
            "conversion_backend": "colour-science",
        },
        "dataset": {
            "samples": dataset_size,
            "stack": "Ag-SiO2-Ag",
            "physics_mode": spectrum_mode_name,
            "design_bounds_nm": design_bounds_nm
            or {
                "ag_bottom": [10.0, 30.0],
                "sio2": [60.0, 180.0],
                "ag_top": [10.0, 30.0],
            },
        },
        "scaling": {
            "lab": bundle.lab_scaling_type,
            "design": bundle.design_scaling_type,
        },
        "runtime": get_torch_runtime_info(args.device),
        "training": {
            "experiment_preset": getattr(args, "experiment_preset", None),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "noise_dim": args.noise_dim,
            "generator_learning_rate": args.generator_learning_rate,
            "discriminator_learning_rate": args.discriminator_learning_rate,
            "steps_per_batch": args.steps_per_batch,
            "generator_hidden_dim": args.generator_hidden_dim,
            "generator_depth": args.generator_depth,
            "discriminator_hidden_dim": args.discriminator_hidden_dim,
            "discriminator_depth": args.discriminator_depth,
            "discriminator_conditioning": args.discriminator_conditioning,
            "alpha_start": args.alpha_start,
            "alpha_ramp_epochs": args.alpha_ramp_epochs,
            "max_alpha": args.max_alpha,
            "lab_delta_e_weight": args.lab_delta_e_weight,
            "mode_seeking_weight": args.mode_seeking_weight,
            "seed": args.seed,
            "device": bundle.device,
            "regressor_epochs": args.regressor_epochs,
            "best_checkpoint_epoch": bundle.selected_checkpoint_epoch,
            "best_checkpoint_metric_name": bundle.selected_checkpoint_metric_name,
            "best_checkpoint_metric_value": bundle.selected_checkpoint_metric_value,
            "checkpoint_eval_interval": args.checkpoint_eval_interval,
            "checkpoint_budget_mode": args.checkpoint_budget_mode,
            "checkpoint_samples_per_lab": args.checkpoint_samples_per_lab,
            "checkpoint_recheck_samples_per_lab": args.checkpoint_recheck_samples_per_lab,
            "final_samples_per_lab": args.paper_samples_per_lab,
            "checkpoint_score_weights": {
                "mean_best_delta_e": args.checkpoint_score_weight_mean_best_delta_e,
                "median_best_delta_e": args.checkpoint_score_weight_median_best_delta_e,
                "d2_within_5nm": args.checkpoint_score_weight_d2_within_5nm,
                "mean_jsd": args.checkpoint_score_weight_mean_jsd,
                "d3_jsd": args.checkpoint_score_weight_d3_jsd,
            },
        },
        "retrieval_metric": args.retrieval_metric,
        "targets": [summarize_target(records, target_hex) for target_hex in targets],
        "artifacts": artifact_manifest or {},
        "paper_reproduction": compact_reproduction_metrics_for_json(reproduction_metrics),
        "paper_targets": {
            "lab_regressor_mean_delta_e": 0.19,
            "generator_mean_best_delta_e_with_1000_z": 0.44,
            "mean_solution_groups_with_1000_z": 3.58,
            "d2_ground_truth_within_5nm_ratio": 0.939,
            "jsd": {"d1": 0.069, "d2": 0.067, "d3": 0.066},
        },
    }
    output_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def resolve_artifact_bundle(
    training_bundle: CganBundle,
    *,
    best_checkpoint_path: Path,
    device: str,
) -> CganBundle:
    if not best_checkpoint_path.exists():
        return training_bundle
    return load_model_bundle(best_checkpoint_path, device=device)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the lightweight cGAN and export paper-reproduction figures.",
    )
    parser.add_argument(
        "--experiment-preset",
        choices=sorted(EXPERIMENT_PRESETS.keys()),
        default=None,
        help=(
            "Apply a reproducible cGAN experiment preset before parsing the rest of the CLI. "
            "Later explicit flags still override preset values."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "backend" / "artifacts" / "cgan_reproduction",
    )
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--noise-dim", type=int, default=8)
    parser.add_argument("--generator-learning-rate", type=float, default=1e-3)
    parser.add_argument("--discriminator-learning-rate", type=float, default=2e-4)
    parser.add_argument("--steps-per-batch", type=int, default=1)
    parser.add_argument("--generator-hidden-dim", type=int, default=160)
    parser.add_argument("--generator-depth", type=int, default=5)
    parser.add_argument("--discriminator-hidden-dim", type=int, default=128)
    parser.add_argument("--discriminator-depth", type=int, default=4)
    parser.add_argument(
        "--discriminator-conditioning",
        choices=["none", "target_lab"],
        default="target_lab",
    )
    parser.add_argument("--alpha-start", type=float, default=0.0)
    parser.add_argument("--alpha-ramp-epochs", type=int, default=2000)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument("--lab-delta-e-weight", type=float, default=0.0)
    parser.add_argument("--mode-seeking-weight", type=float, default=0.1)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--top-generated", type=int, default=16)
    parser.add_argument("--bottom-points", type=int, default=9)
    parser.add_argument("--sio2-points", type=int, default=25)
    parser.add_argument("--top-points", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument(
        "--dataset-source",
        choices=["synthetic", "paper"],
        default="synthetic",
    )
    parser.add_argument("--paper-train-csv", type=Path, default=DEFAULT_PAPER_TRAINING_CSV)
    parser.add_argument("--paper-test-csv", type=Path, default=DEFAULT_PAPER_TESTING_CSV)
    parser.add_argument("--paper-samples-per-lab", type=int, default=1000)
    parser.add_argument("--regressor-epochs", type=int, default=10000)
    parser.add_argument("--checkpoint-eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-patience", type=int, default=None)
    parser.add_argument("--checkpoint-samples-per-lab", type=int, default=64)
    parser.add_argument(
        "--checkpoint-budget-mode",
        choices=["match_final", "small_budget", "recheck_best"],
        default=DEFAULT_CHECKPOINT_BUDGET_MODE,
    )
    parser.add_argument("--checkpoint-recheck-samples-per-lab", type=int, default=256)
    parser.add_argument(
        "--checkpoint-score-weight-mean-best-delta-e",
        type=float,
        default=DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEAN_BEST_DELTA_E,
    )
    parser.add_argument(
        "--checkpoint-score-weight-median-best-delta-e",
        type=float,
        default=DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEDIAN_BEST_DELTA_E,
    )
    parser.add_argument(
        "--checkpoint-score-weight-d2-within-5nm",
        type=float,
        default=DEFAULT_CHECKPOINT_SCORE_WEIGHT_D2_WITHIN_5NM,
    )
    parser.add_argument(
        "--checkpoint-score-weight-mean-jsd",
        type=float,
        default=DEFAULT_CHECKPOINT_SCORE_WEIGHT_MEAN_JSD,
    )
    parser.add_argument(
        "--checkpoint-score-weight-d3-jsd",
        type=float,
        default=DEFAULT_CHECKPOINT_SCORE_WEIGHT_D3_JSD,
    )
    parser.add_argument(
        "--retrieval-metric",
        choices=["euclidean_lab", "delta_e_2000"],
        default="euclidean_lab",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = _build_arg_parser()
    initial_args, _ = parser.parse_known_args()
    preset = get_experiment_preset(initial_args.experiment_preset)
    if preset is not None:
        parser.set_defaults(**preset.overrides)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.output_dir / "train.log")

    with _stream_logs_to_file(log_path):
        _log(f"Streaming training logs to {log_path}")
        load_runtime_dependencies()
        overall_start_time = time.monotonic()
        progress_logger = _build_progress_logger()

        reproduction_metrics: dict[str, object] | None = None
        if args.dataset_source == "paper":
            if not args.paper_train_csv.exists() or not args.paper_test_csv.exists():
                args.paper_train_csv, args.paper_test_csv = ensure_paper_dataset_files()
            design_samples, lab_samples = load_paper_dataset_csv(args.paper_train_csv)
            hex_samples = None
            test_designs, test_labs = load_paper_dataset_csv(args.paper_test_csv)
            design_bounds_nm = {
                "ag_bottom": [0.0, 50.0],
                "sio2": [0.0, 1000.0],
                "ag_top": [0.0, 50.0],
            }
        else:
            design_samples, lab_samples, hex_samples = build_ag_sio2_ag_dataset(
                bottom_points=args.bottom_points,
                sio2_points=args.sio2_points,
                top_points=args.top_points,
            )
            test_designs, test_labs = design_samples, lab_samples
            design_bounds_nm = {
                "ag_bottom": [10.0, 30.0],
                "sio2": [60.0, 180.0],
                "ag_top": [10.0, 30.0],
            }

        runtime_info = get_torch_runtime_info(args.device)
        _log(
            "Loaded dataset "
            f"(source={args.dataset_source}, train_samples={len(design_samples)}, "
            f"test_samples={len(test_designs)}, output_dir={args.output_dir})"
        )
        _log(
            "Runtime "
            f"(device={runtime_info['device']}, cuda_available={runtime_info['cuda_available']}, "
            f"torch={runtime_info['torch_version']})"
        )

        regressor_progress_interval = _default_progress_interval(
            total_steps=args.regressor_epochs,
            target_updates=20,
            max_interval=1000,
        )
        training_progress_interval = _default_progress_interval(
            total_steps=args.epochs,
            target_updates=50,
            max_interval=5000,
        )
        paper_eval_progress_interval = _default_progress_interval(
            total_steps=len(test_labs),
            target_updates=20,
            max_interval=500,
        )
        checkpoint_budgets = resolve_checkpoint_evaluation_budgets(args)
        args.checkpoint_samples_per_lab = checkpoint_budgets["checkpoint_samples_per_lab"]
        args.checkpoint_recheck_samples_per_lab = checkpoint_budgets["checkpoint_recheck_samples_per_lab"]
        args.paper_samples_per_lab = checkpoint_budgets["final_samples_per_lab"]
        _log(
            "Progress logging "
            f"(regressor_every={regressor_progress_interval} epochs, "
            f"train_every={training_progress_interval} epochs, "
            f"paper_eval_every={paper_eval_progress_interval} targets, "
            f"checkpoint_eval_interval={args.checkpoint_eval_interval}, "
            f"checkpoint_budget_mode={args.checkpoint_budget_mode}, "
            f"checkpoint_samples_per_lab={args.checkpoint_samples_per_lab}, "
            f"checkpoint_recheck_samples_per_lab={args.checkpoint_recheck_samples_per_lab}, "
            f"final_samples_per_lab={args.paper_samples_per_lab})"
        )
        _log(
            "Training config "
            f"(preset={args.experiment_preset or 'custom'}, "
            f"g_lr={args.generator_learning_rate}, d_lr={args.discriminator_learning_rate}, "
            f"steps_per_batch={args.steps_per_batch}, "
            f"g_hidden={args.generator_hidden_dim}, g_depth={args.generator_depth}, "
            f"d_hidden={args.discriminator_hidden_dim}, d_depth={args.discriminator_depth}, "
            f"d_conditioning={args.discriminator_conditioning}, "
            f"alpha_start={args.alpha_start}, alpha_ramp_epochs={args.alpha_ramp_epochs}, "
            f"max_alpha={args.max_alpha}, lab_delta_e_weight={args.lab_delta_e_weight}, "
            f"mode_seeking_weight={args.mode_seeking_weight}, "
            f"retrieval_metric={args.retrieval_metric})"
        )
        _log(
            "Checkpoint score weights "
            f"(mean_best_delta_e={args.checkpoint_score_weight_mean_best_delta_e}, "
            f"median_best_delta_e={args.checkpoint_score_weight_median_best_delta_e}, "
            f"d2_within_5nm={args.checkpoint_score_weight_d2_within_5nm}, "
            f"mean_jsd={args.checkpoint_score_weight_mean_jsd}, "
            f"d3_jsd={args.checkpoint_score_weight_d3_jsd})"
        )

        checkpoint_metric_fn = None
        checkpoint_metric_name = None
        checkpoint_metric_mode = "min"
        best_checkpoint_tracker = {
            "epoch": None,
            "metric_value": None,
        }
        if args.dataset_source == "paper":

            def checkpoint_metric_fn(bundle: CganBundle, epoch: int) -> float | None:
                metrics = evaluate_testing_set_distribution(
                    bundle=bundle,
                    test_designs=test_designs,
                    test_labs=test_labs,
                    samples_per_lab=args.checkpoint_samples_per_lab,
                    seed=args.seed,
                    progress_callback=progress_logger,
                    progress_interval=paper_eval_progress_interval,
                    progress_phase="checkpoint",
                )
                metric_value_float = compute_checkpoint_score(
                    metrics,
                    weight_mean_best_delta_e=args.checkpoint_score_weight_mean_best_delta_e,
                    weight_median_best_delta_e=args.checkpoint_score_weight_median_best_delta_e,
                    weight_d2_within_5nm=args.checkpoint_score_weight_d2_within_5nm,
                    weight_mean_jsd=args.checkpoint_score_weight_mean_jsd,
                    weight_d3_jsd=args.checkpoint_score_weight_d3_jsd,
                )
                metrics_for_selection = metrics
                if (
                    args.checkpoint_budget_mode == "recheck_best"
                    and args.checkpoint_recheck_samples_per_lab > args.checkpoint_samples_per_lab
                ):
                    best_so_far = best_checkpoint_tracker["metric_value"]
                    if best_so_far is None or metric_value_float < float(best_so_far):
                        metrics_for_selection = evaluate_testing_set_distribution(
                            bundle=bundle,
                            test_designs=test_designs,
                            test_labs=test_labs,
                            samples_per_lab=args.checkpoint_recheck_samples_per_lab,
                            seed=args.seed,
                            progress_callback=progress_logger,
                            progress_interval=paper_eval_progress_interval,
                            progress_phase="checkpoint_recheck",
                        )
                        metric_value_float = compute_checkpoint_score(
                            metrics_for_selection,
                            weight_mean_best_delta_e=args.checkpoint_score_weight_mean_best_delta_e,
                            weight_median_best_delta_e=args.checkpoint_score_weight_median_best_delta_e,
                            weight_d2_within_5nm=args.checkpoint_score_weight_d2_within_5nm,
                            weight_mean_jsd=args.checkpoint_score_weight_mean_jsd,
                            weight_d3_jsd=args.checkpoint_score_weight_d3_jsd,
                        )
                best_so_far = best_checkpoint_tracker["metric_value"]
                if best_so_far is None or metric_value_float < float(best_so_far):
                    best_checkpoint_tracker["epoch"] = epoch
                    best_checkpoint_tracker["metric_value"] = metric_value_float
                    save_best_model_bundle(bundle, args.output_dir / "generator_checkpoint_best.pt")
                    _write_checkpoint_state(
                        output_dir=args.output_dir,
                        epoch=epoch,
                        metric_name="paper_reproduction.checkpoint_score",
                        metric_value=metric_value_float,
                    )
                    _log(
                        "Persisted best checkpoint "
                        f"(epoch={epoch}, paper_reproduction.checkpoint_score={metric_value_float:.6f}, "
                        f"mean_best_delta_e={float(metrics_for_selection['mean_best_delta_e']):.6f}, "
                        f"median_best_delta_e={float(metrics_for_selection['median_best_delta_e']):.6f}, "
                        f"d2_within_5nm={float(metrics_for_selection['d2_ground_truth_within_5nm_ratio']):.6f})"
                    )
                return metric_value_float

            checkpoint_metric_name = "paper_reproduction.checkpoint_score"

        training_bundle = fit_lightweight_cgan(
            lab_samples,
            design_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            noise_dim=args.noise_dim,
            generator_learning_rate=args.generator_learning_rate,
            discriminator_learning_rate=args.discriminator_learning_rate,
            steps_per_batch=args.steps_per_batch,
            generator_hidden_dim=args.generator_hidden_dim,
            generator_depth=args.generator_depth,
            discriminator_hidden_dim=args.discriminator_hidden_dim,
            discriminator_depth=args.discriminator_depth,
            discriminator_conditioning=args.discriminator_conditioning,
            alpha_start=args.alpha_start,
            alpha_ramp_epochs=args.alpha_ramp_epochs,
            max_alpha=args.max_alpha,
            lab_delta_e_weight=args.lab_delta_e_weight,
            mode_seeking_weight=args.mode_seeking_weight,
            seed=args.seed,
            record_losses=True,
            device=args.device,
            regressor_epochs=args.regressor_epochs,
            checkpoint_metric_fn=checkpoint_metric_fn,
            checkpoint_metric_name=checkpoint_metric_name,
            checkpoint_metric_mode=checkpoint_metric_mode,
            checkpoint_metric_interval=args.checkpoint_eval_interval,
            checkpoint_patience=args.checkpoint_patience,
            progress_callback=progress_logger,
            progress_interval=training_progress_interval,
            regressor_progress_interval=regressor_progress_interval,
        )

        final_checkpoint_path = args.output_dir / "generator_checkpoint.pt"
        best_checkpoint_path = args.output_dir / "generator_checkpoint_best.pt"
        save_model_bundle(training_bundle, final_checkpoint_path)
        if training_bundle.best_generator_state_dict is not None:
            save_best_model_bundle(training_bundle, best_checkpoint_path)

        artifact_bundle = resolve_artifact_bundle(
            training_bundle,
            best_checkpoint_path=best_checkpoint_path,
            device=args.device,
        )
        if artifact_bundle.selected_checkpoint_epoch is not None:
            _log(
                "Using best checkpoint for final artifact export "
                f"(epoch={artifact_bundle.selected_checkpoint_epoch})"
            )

        _log("Collecting candidate records for target colors")
        record_collection_started_at = time.monotonic()
        records = collect_candidate_records(
            bundle=artifact_bundle,
            lab_samples=lab_samples,
            design_samples=design_samples,
            hex_samples=hex_samples,
            targets=[target.lower() for target in args.targets],
            sample_count=args.sample_count,
            top_generated=args.top_generated,
            seed=args.seed,
            retrieval_metric=args.retrieval_metric,
        )
        record_collection_elapsed_seconds = time.monotonic() - record_collection_started_at

        comparison_records_by_metric: dict[str, list[CandidateRecord]] = {}
        comparison_elapsed_seconds_by_metric: dict[str, float] = {}
        for comparison_metric in ("euclidean_lab", "delta_e_2000"):
            comparison_started_at = time.monotonic()
            comparison_records_by_metric[comparison_metric] = collect_candidate_records(
                bundle=artifact_bundle,
                lab_samples=lab_samples,
                design_samples=design_samples,
                hex_samples=hex_samples,
                targets=[target.lower() for target in args.targets],
                sample_count=args.sample_count,
                top_generated=args.top_generated,
                seed=args.seed,
                retrieval_metric=comparison_metric,
            )
            comparison_elapsed_seconds_by_metric[comparison_metric] = time.monotonic() - comparison_started_at

        comparison_elapsed_seconds_by_metric[args.retrieval_metric] = record_collection_elapsed_seconds

        if args.dataset_source == "paper":
            _log(
                "Running final paper evaluation "
                f"(samples_per_lab={args.paper_samples_per_lab}, test_targets={len(test_labs)})"
            )
            reproduction_metrics = evaluate_testing_set_distribution(
                bundle=artifact_bundle,
                test_designs=test_designs,
                test_labs=test_labs,
                samples_per_lab=args.paper_samples_per_lab,
                seed=args.seed,
                progress_callback=progress_logger,
                progress_interval=paper_eval_progress_interval,
                progress_phase="final",
            )

        _log("Writing artifacts to disk")
        write_loss_csv(training_bundle, args.output_dir / "loss_history.csv")
        write_candidate_csv(records, args.output_dir / "candidate_samples.csv")
        write_retrieval_comparison_json(
            comparison_records_by_metric=comparison_records_by_metric,
            output_path=args.output_dir / "retrieval_metric_comparison.json",
            elapsed_seconds_by_metric=comparison_elapsed_seconds_by_metric,
        )
        artifact_manifest = write_artifact_manifest(
            output_dir=args.output_dir,
            selected_checkpoint_name="generator_checkpoint_best.pt",
            final_checkpoint_name="generator_checkpoint.pt",
            best_available_checkpoint_name="generator_checkpoint_best.pt",
        )
        write_metrics(
            bundle=artifact_bundle,
            records=records,
            targets=[target.lower() for target in args.targets],
            dataset_size=len(design_samples),
            args=args,
            output_path=args.output_dir / "metrics.json",
            reproduction_metrics=reproduction_metrics,
            design_bounds_nm=design_bounds_nm,
            artifact_manifest=artifact_manifest,
        )
        write_paper_reproduction_details(
            reproduction_metrics,
            args.output_dir / "paper_reproduction_details.npz",
        )
        plot_loss_curve(training_bundle, args.output_dir / "loss_curve.png")
        plot_sampling_comparison(
            records,
            [target.lower() for target in args.targets],
            args.output_dir / "sampling_comparison.png",
        )
        plot_candidate_diversity(
            records,
            args.targets[0].lower(),
            args.output_dir / "candidate_diversity.png",
        )
        if reproduction_metrics is not None:
            generated_designs = np.array(reproduction_metrics["generated_designs_nm"], dtype=np.float64)
            plot_paper_distribution_comparison(
                test_designs=test_designs,
                generated_designs=generated_designs.reshape(-1, generated_designs.shape[-1]),
                output_path=args.output_dir / "paper_figure4_distribution_comparison.png",
            )
            plot_paper_solution_metrics(
                best_delta_e_values=np.array(reproduction_metrics["best_delta_e_values"], dtype=np.float64),
                solution_group_counts=np.array(reproduction_metrics["solution_group_counts"], dtype=np.float64),
                output_path=args.output_dir / "paper_figure5_solution_metrics.png",
            )
            plot_paper_d2_accuracy(
                abs_d2_errors_nm=np.array(reproduction_metrics["abs_d2_errors_nm"], dtype=np.float64),
                output_path=args.output_dir / "paper_figure_s6_d2_accuracy.png",
            )

        total_runtime = time.monotonic() - overall_start_time
        _log(f"Wrote cGAN reproduction artifacts to {args.output_dir}")
        _log(
            "Key outputs: train.log, loss_curve.png, sampling_comparison.png, candidate_diversity.png, "
            "paper_figure4_distribution_comparison.png, paper_figure5_solution_metrics.png, "
            "retrieval_metric_comparison.json, metrics.json"
        )
        _log(f"Total runtime: {_format_duration(total_runtime)}")


if __name__ == "__main__":
    main()

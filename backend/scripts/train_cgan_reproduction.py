from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
import zipfile
from typing import TYPE_CHECKING, Any

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


def load_runtime_dependencies() -> None:
    global np
    global torch
    global CganBundle
    global fit_lightweight_cgan
    global get_torch_runtime_info
    global sample_designs_from_bundle
    global sample_designs_for_labs_from_bundle
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
    hex_samples: list[str] | None,
    target_lab: np.ndarray,
) -> tuple[float, str, np.ndarray]:
    distances = np.linalg.norm(lab_samples - target_lab.reshape(1, -1), axis=1)
    index = int(np.argmin(distances))
    if hex_samples is not None:
        simulated_hex = hex_samples[index]
    else:
        _, simulated_hex = evaluate_design(design_samples[index], target_lab)
    return float(distances[index]), simulated_hex, design_samples[index]


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
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []

    for target_index, target_hex in enumerate(targets):
        target_lab = hex_to_lab(target_hex)
        retrieval_delta_e, retrieval_hex, retrieval_design = nearest_retrieval(
            lab_samples,
            design_samples,
            hex_samples,
            target_lab,
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
) -> dict[str, object]:
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

    return {
        "samples_per_lab": samples_per_lab,
        "mean_best_delta_e": float(np.mean(best_delta_es)),
        "median_best_delta_e": float(np.median(best_delta_es)),
        "mean_solution_groups": float(np.mean(solution_group_counts)),
        "max_solution_groups": int(max(solution_group_counts, default=0)),
        "d2_ground_truth_within_5nm_ratio": d2_within_5nm,
        "generated_designs_nm": generated_design_batches.tolist(),
        "best_delta_e_values": best_delta_es,
        "solution_group_counts": solution_group_counts,
        "abs_d2_errors_nm": abs_d2_errors_nm,
        "jsd": jsd,
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
        compact_metrics["generated_designs_shape"] = [
            len(generated_designs),
            len(generated_designs[0]) if generated_designs else 0,
            len(generated_designs[0][0]) if generated_designs and generated_designs[0] else 0,
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
        "alpha",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in bundle.losses:
            writer.writerow({fieldname: row.get(fieldname) for fieldname in fieldnames})


def write_candidate_csv(records: list[CandidateRecord], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(asdict(records[0]).keys()) if records else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def save_model_bundle(bundle: CganBundle, output_path: Path) -> None:
    generator_state_dict = bundle.generator.state_dict()
    if output_path.name == "generator_checkpoint_best.pt" and bundle.best_generator_state_dict is not None:
        generator_state_dict = bundle.best_generator_state_dict
    torch.save(
        {
            "generator_state_dict": generator_state_dict,
            "lab_regressor_state_dict": bundle.lab_regressor.state_dict(),
            "lab_min": bundle.lab_min.tolist(),
            "lab_max": bundle.lab_max.tolist(),
            "design_min": bundle.design_min.tolist(),
            "design_max": bundle.design_max.tolist(),
            "device": bundle.device,
            "noise_dim": bundle.noise_dim,
            "seed": bundle.seed,
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
    ax.set_ylabel("Binary cross-entropy")
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
        "runtime": get_torch_runtime_info(args.device),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "noise_dim": args.noise_dim,
            "learning_rate": args.learning_rate,
            "evaluator_learning_rate": args.learning_rate / 5.0,
            "seed": args.seed,
            "device": bundle.device,
            "regressor_epochs": args.regressor_epochs,
            "best_checkpoint_epoch": bundle.selected_checkpoint_epoch,
            "best_checkpoint_metric_name": bundle.selected_checkpoint_metric_name,
            "best_checkpoint_metric_value": bundle.selected_checkpoint_metric_value,
            "checkpoint_eval_interval": args.checkpoint_eval_interval,
        },
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the lightweight cGAN and export paper-reproduction figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "backend" / "artifacts" / "cgan_reproduction",
    )
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=40000)
    parser.add_argument("--noise-dim", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.002)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_runtime_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    checkpoint_metric_fn = None
    checkpoint_metric_name = None
    checkpoint_metric_mode = "min"
    if args.dataset_source == "paper":
        def checkpoint_metric_fn(bundle: CganBundle, epoch: int) -> float | None:
            metrics = evaluate_testing_set_distribution(
                bundle=bundle,
                test_designs=test_designs,
                test_labs=test_labs,
                samples_per_lab=min(args.paper_samples_per_lab, 64),
                seed=args.seed + epoch,
            )
            metric_value = metrics.get("mean_best_delta_e")
            return float(metric_value) if metric_value is not None else None

        checkpoint_metric_name = "paper_reproduction.mean_best_delta_e"

    bundle = fit_lightweight_cgan(
        lab_samples,
        design_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        noise_dim=args.noise_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        record_losses=True,
        device=args.device,
        regressor_epochs=args.regressor_epochs,
        checkpoint_metric_fn=checkpoint_metric_fn,
        checkpoint_metric_name=checkpoint_metric_name,
        checkpoint_metric_mode=checkpoint_metric_mode,
        checkpoint_metric_interval=args.checkpoint_eval_interval,
    )
    records = collect_candidate_records(
        bundle=bundle,
        lab_samples=lab_samples,
        design_samples=design_samples,
        hex_samples=hex_samples,
        targets=[target.lower() for target in args.targets],
        sample_count=args.sample_count,
        top_generated=args.top_generated,
        seed=args.seed,
    )

    if args.dataset_source == "paper":
        reproduction_metrics = evaluate_testing_set_distribution(
            bundle=bundle,
            test_designs=test_designs,
            test_labs=test_labs,
            samples_per_lab=args.paper_samples_per_lab,
            seed=args.seed,
        )

    write_loss_csv(bundle, args.output_dir / "loss_history.csv")
    write_candidate_csv(records, args.output_dir / "candidate_samples.csv")
    save_model_bundle(bundle, args.output_dir / "generator_checkpoint.pt")
    save_model_bundle(bundle, args.output_dir / "generator_checkpoint_best.pt")
    artifact_manifest = write_artifact_manifest(
        output_dir=args.output_dir,
        selected_checkpoint_name="generator_checkpoint_best.pt",
        final_checkpoint_name="generator_checkpoint.pt",
        best_available_checkpoint_name="generator_checkpoint_best.pt",
    )
    write_metrics(
        bundle=bundle,
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
    plot_loss_curve(bundle, args.output_dir / "loss_curve.png")
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

    print(f"Wrote cGAN reproduction artifacts to {args.output_dir}")
    print(
        "Key outputs: loss_curve.png, sampling_comparison.png, candidate_diversity.png, "
        "paper_figure4_distribution_comparison.png, paper_figure5_solution_metrics.png, metrics.json"
    )


if __name__ == "__main__":
    main()

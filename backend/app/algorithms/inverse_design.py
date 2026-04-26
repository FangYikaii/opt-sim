from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..algorithm_overview import get_algorithm_overview, resolve_selected_checkpoint_path
from ..models import CandidateMetric, CandidateParameter, CandidateSolution, ConstraintCheck, ExportEstimate
from .cgan import load_model_bundle, sample_designs_from_bundle, sample_designs_from_cgan
from .optics import (
    Polarization,
    delta_e_2000,
    hex_to_lab,
    reflectance_spectrum_ag_sio2_ag,
    spectrum_to_xyz,
    transmittance_spectrum_ag_sio2_ag,
    xyz_to_lab,
    xyz_to_srgb_hex,
)

RetrievalMetric = Literal["euclidean_lab", "delta_e_2000"]


@dataclass
class InverseDesignResult:
    candidates: list[CandidateSolution]
    constraints: list[ConstraintCheck]
    export_estimate: ExportEstimate


@dataclass(frozen=True)
class EvaluatedDesign:
    design: tuple[float, float, float]
    delta_e: float
    simulated_hex: str
    plus_delta_e: float
    plus_hex: str
    minus_delta_e: float
    minus_hex: str
    drift: float
    source: str
    composite_score: float


@lru_cache(maxsize=65536)
def _cached_design_response(
    d_ag_bottom: float,
    d_sio2: float,
    d_ag_top: float,
    theta_deg: float,
    polarization: Polarization,
) -> tuple[np.ndarray, np.ndarray, str]:
    spectrum = transmittance_spectrum_ag_sio2_ag(
        d_ag_bottom,
        d_sio2,
        d_ag_top,
        theta_deg=theta_deg,
        polarization=polarization,
    )
    xyz = spectrum_to_xyz(spectrum)
    lab = xyz_to_lab(xyz)
    return spectrum, lab, xyz_to_srgb_hex(xyz)


def _evaluate_design(
    d_ag_bottom: float,
    d_sio2: float,
    d_ag_top: float,
    target_lab: np.ndarray,
    *,
    theta_deg: float,
    polarization: Polarization,
) -> tuple[float, str]:
    _, lab, simulated_hex = _cached_design_response(
        d_ag_bottom,
        d_sio2,
        d_ag_top,
        float(theta_deg),
        polarization,
    )
    delta_e = delta_e_2000(lab, target_lab)
    return delta_e, simulated_hex


def _evaluate_candidate(
    d_ag_bottom: float,
    d_sio2: float,
    d_ag_top: float,
    target_lab: np.ndarray,
    *,
    source: str,
    theta_deg: float,
    polarization: Polarization,
) -> EvaluatedDesign:
    base_delta_e, base_hex = _evaluate_design(
        d_ag_bottom,
        d_sio2,
        d_ag_top,
        target_lab,
        theta_deg=theta_deg,
        polarization=polarization,
    )
    plus_delta_e, plus_hex = _evaluate_design(
        min(30.0, d_ag_bottom + 0.5),
        min(180.0, d_sio2 + 0.5),
        min(30.0, d_ag_top + 0.5),
        target_lab,
        theta_deg=theta_deg,
        polarization=polarization,
    )
    minus_delta_e, minus_hex = _evaluate_design(
        max(10.0, d_ag_bottom - 0.5),
        max(60.0, d_sio2 - 0.5),
        max(10.0, d_ag_top - 0.5),
        target_lab,
        theta_deg=theta_deg,
        polarization=polarization,
    )
    plus_drift = plus_delta_e - base_delta_e
    minus_drift = minus_delta_e - base_delta_e
    drift = max(abs(plus_drift), abs(minus_drift))
    composite_score = base_delta_e + 0.35 * drift
    return EvaluatedDesign(
        design=(float(d_ag_bottom), float(d_sio2), float(d_ag_top)),
        delta_e=float(base_delta_e),
        simulated_hex=base_hex,
        plus_delta_e=float(plus_delta_e),
        plus_hex=plus_hex,
        minus_delta_e=float(minus_delta_e),
        minus_hex=minus_hex,
        drift=float(drift),
        source=source,
        composite_score=float(composite_score),
    )


def _iter_local_variants(
    design: tuple[float, float, float],
    *,
    step_ag_nm: float,
    step_sio2_nm: float,
) -> list[tuple[float, float, float]]:
    d_ag_bottom, d_sio2, d_ag_top = design
    variants: set[tuple[float, float, float]] = set()
    for delta_bottom in (-step_ag_nm, 0.0, step_ag_nm):
        for delta_sio2 in (-step_sio2_nm, 0.0, step_sio2_nm):
            for delta_top in (-step_ag_nm, 0.0, step_ag_nm):
                variants.add(
                    (
                        float(np.clip(d_ag_bottom + delta_bottom, 10.0, 30.0)),
                        float(np.clip(d_sio2 + delta_sio2, 60.0, 180.0)),
                        float(np.clip(d_ag_top + delta_top, 10.0, 30.0)),
                    )
                )
    return sorted(variants)


def _refine_design(
    base_design: tuple[float, float, float],
    target_lab: np.ndarray,
    *,
    source: str,
    theta_deg: float,
    polarization: Polarization,
) -> EvaluatedDesign:
    current_best = _evaluate_candidate(
        *base_design,
        target_lab,
        source=source,
        theta_deg=theta_deg,
        polarization=polarization,
    )
    search_schedule = [(1.0, 2.0)]

    for step_ag_nm, step_sio2_nm in search_schedule:
        improved = True
        while improved:
            improved = False
            for candidate_design in _iter_local_variants(
                current_best.design,
                step_ag_nm=step_ag_nm,
                step_sio2_nm=step_sio2_nm,
            ):
                candidate = _evaluate_candidate(
                    *candidate_design,
                    target_lab,
                    source=source,
                    theta_deg=theta_deg,
                    polarization=polarization,
                )
                if candidate.composite_score + 1e-9 < current_best.composite_score:
                    current_best = candidate
                    improved = True

    refined_source = "cGAN+refined" if "cGAN" in source else "refined"
    return EvaluatedDesign(
        design=current_best.design,
        delta_e=current_best.delta_e,
        simulated_hex=current_best.simulated_hex,
        plus_delta_e=current_best.plus_delta_e,
        plus_hex=current_best.plus_hex,
        minus_delta_e=current_best.minus_delta_e,
        minus_hex=current_best.minus_hex,
        drift=current_best.drift,
        source=refined_source,
        composite_score=current_best.composite_score,
    )


def _status_and_rationale(rank: int, drift: float, source: str) -> tuple[str, str]:
    if rank == 1:
        return "Recommended", (
            "Best overall match after combining nominal color error with a local manufacturability refinement pass."
        )
    if drift < 0.8:
        return "Robust", (
            f"{source} candidate remains relatively stable under +/-0.5 nm process perturbation."
        )
    return "Watch", (
        "Nominal color match is usable, but small fabrication perturbations still shift color noticeably."
    )


def _load_saved_cgan_bundle():
    overview = get_algorithm_overview()
    if overview.bestExperimentId is None:
        return None

    checkpoint_path = resolve_selected_checkpoint_path(overview.bestExperimentId)
    if checkpoint_path is None:
        return None

    try:
        return load_model_bundle(checkpoint_path)
    except Exception:
        return None


def _sample_gan_designs(
    *,
    target_lab: np.ndarray,
    lab_array: np.ndarray,
    design_array: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    lower_bounds = design_array.min(axis=0)
    upper_bounds = design_array.max(axis=0)

    saved_bundle = _load_saved_cgan_bundle()
    if saved_bundle is not None:
        generated_designs = sample_designs_from_bundle(saved_bundle, target_lab, sample_count=sample_count)
    else:
        generated_designs = sample_designs_from_cgan(
            target_lab=target_lab,
            lab_samples=lab_array,
            design_samples=design_array,
            sample_count=sample_count,
        )

    return np.clip(generated_designs, lower_bounds, upper_bounds)


def _retrieval_grid(
    theta_deg: float,
    polarization: Polarization,
) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    grid: list[tuple[float, float, float]] = []
    labs: list[np.ndarray] = []
    for d_ag_bottom in np.arange(10.0, 31.0, 2.0):
        for d_sio2 in np.arange(60.0, 181.0, 5.0):
            for d_ag_top in np.arange(10.0, 31.0, 2.0):
                _, lab, _ = _cached_design_response(
                    float(d_ag_bottom),
                    float(d_sio2),
                    float(d_ag_top),
                    float(theta_deg),
                    polarization,
                )
                grid.append((float(d_ag_bottom), float(d_sio2), float(d_ag_top)))
                labs.append(lab)
    return grid, np.vstack(labs)


def _retrieve_candidate_indices(
    *,
    lab_array: np.ndarray,
    target_lab: np.ndarray,
    top_k: int,
    retrieval_metric: RetrievalMetric,
) -> list[int]:
    if retrieval_metric == "euclidean_lab":
        neighbors = NearestNeighbors(n_neighbors=top_k)
        neighbors.fit(lab_array)
        _, indices = neighbors.kneighbors(target_lab.reshape(1, -1))
        return [int(index) for index in indices[0]]

    delta_es = np.array(
        [delta_e_2000(candidate_lab, target_lab) for candidate_lab in lab_array],
        dtype=np.float64,
    )
    ranked = np.argsort(delta_es, kind="stable")
    return [int(index) for index in ranked[:top_k]]


def _format_retrieval_metric_label(retrieval_metric: RetrievalMetric) -> str:
    if retrieval_metric == "delta_e_2000":
        return "DeltaE 2000"
    return "Lab Euclidean"


def _format_polarization_label(polarization: Polarization) -> str:
    if polarization == "te":
        return "TE"
    if polarization == "tm":
        return "TM"
    return "Unpolarized"


def run_inverse_design(
    target_hex: str,
    top_k: int = 3,
    theta_deg: float = 0.0,
    polarization: Polarization = "unpolarized",
    retrieval_metric: RetrievalMetric = "euclidean_lab",
) -> InverseDesignResult:
    target_lab = hex_to_lab(target_hex)

    grid, lab_array = _retrieval_grid(theta_deg, polarization)
    design_array = np.array(grid, dtype=np.float64)
    indices = _retrieve_candidate_indices(
        lab_array=lab_array,
        target_lab=target_lab,
        top_k=top_k,
        retrieval_metric=retrieval_metric,
    )

    gan_designs = _sample_gan_designs(
        target_lab=target_lab,
        lab_array=lab_array[::12],
        design_array=design_array[::12],
        sample_count=max(12, top_k * 6),
    )

    evaluated_candidates: dict[tuple[float, float, float], EvaluatedDesign] = {}

    for idx in indices:
        retrieval_design = tuple(float(value) for value in grid[int(idx)])
        retrieval_candidate = _evaluate_candidate(
            *retrieval_design,
            target_lab,
            source="retrieval",
            theta_deg=theta_deg,
            polarization=polarization,
        )
        retrieval_refined = _refine_design(
            retrieval_design,
            target_lab,
            source="retrieval",
            theta_deg=theta_deg,
            polarization=polarization,
        )
        evaluated_candidates[retrieval_candidate.design] = retrieval_candidate
        evaluated_candidates[retrieval_refined.design] = retrieval_refined

    for design in gan_designs[: max(top_k * 2, 4)]:
        design_tuple = tuple(float(value) for value in design)
        gan_candidate = _evaluate_candidate(
            *design_tuple,
            target_lab,
            source="cGAN+TMM",
            theta_deg=theta_deg,
            polarization=polarization,
        )
        gan_refined = _refine_design(
            design_tuple,
            target_lab,
            source="cGAN+TMM",
            theta_deg=theta_deg,
            polarization=polarization,
        )
        evaluated_candidates[gan_candidate.design] = gan_candidate
        evaluated_candidates[gan_refined.design] = gan_refined

    ranked_candidates = sorted(
        evaluated_candidates.values(),
        key=lambda candidate: (candidate.composite_score, candidate.delta_e, candidate.drift),
    )

    candidates: list[CandidateSolution] = []
    for rank, candidate in enumerate(ranked_candidates[:top_k], start=1):
        d_ag_bottom, d_sio2, d_ag_top = candidate.design
        plus_drift = candidate.plus_delta_e - candidate.delta_e
        minus_drift = candidate.minus_delta_e - candidate.delta_e
        status, rationale = _status_and_rationale(rank, candidate.drift, candidate.source)

        candidates.append(
            CandidateSolution(
                id=f"C-{rank:03d}",
                rank=rank,
                group=f"Group {chr(64 + rank)}",
                selected=rank == 1,
                status=status,
                parameters=[
                    CandidateParameter(label="Ag bottom", value=f"{d_ag_bottom:.1f} nm"),
                    CandidateParameter(label="SiO2", value=f"{d_sio2:.1f} nm"),
                    CandidateParameter(label="Ag top", value=f"{d_ag_top:.1f} nm"),
                ],
                metrics=[
                    CandidateMetric(label="DeltaE", value=f"{candidate.delta_e:.2f}"),
                    CandidateMetric(label="Composite score", value=f"{candidate.composite_score:.2f}"),
                    CandidateMetric(label="Incidence angle", value=f"{theta_deg:.1f} deg"),
                    CandidateMetric(label="Polarization", value=_format_polarization_label(polarization)),
                    CandidateMetric(
                        label="Retrieval metric",
                        value=_format_retrieval_metric_label(retrieval_metric),
                    ),
                    CandidateMetric(label="Process drift", value=f"{plus_drift:+.2f} / {minus_drift:+.2f}"),
                    CandidateMetric(
                        label="Manufacturability",
                        value="High" if candidate.drift < 1.0 else ("Medium" if candidate.drift < 2.0 else "Low"),
                    ),
                    CandidateMetric(label="Source", value=candidate.source),
                ],
                targetColorHex=target_hex.lower(),
                simulatedColorHex=candidate.simulated_hex,
                processPlusColorHex=candidate.plus_hex,
                processMinusColorHex=candidate.minus_hex,
                rationale=rationale,
            )
        )

    constraints = [
        ConstraintCheck(
            id="constraint-height",
            label="Ag thickness bounds",
            detail="All returned Ag layers remain within the sampled 10-30 nm fabrication window.",
            state="pass",
        ),
        ConstraintCheck(
            id="constraint-sio2",
            label="SiO2 thickness bounds",
            detail="Dielectric thickness remains inside the 60-180 nm search window used for retrieval.",
            state="pass",
        ),
        ConstraintCheck(
            id="constraint-robustness",
            label="Process sensitivity",
            detail="Ranking includes plus/minus 0.5 nm perturbation to approximate fabrication error.",
            state="warning",
        ),
        ConstraintCheck(
            id="constraint-angle",
            label="Incidence angle",
            detail=f"Candidates were simulated at an incident angle of {theta_deg:.1f} degrees.",
            state="pass",
        ),
        ConstraintCheck(
            id="constraint-polarization",
            label="Polarization",
            detail=f"Candidates were ranked using {_format_polarization_label(polarization)} light response.",
            state="pass",
        ),
        ConstraintCheck(
            id="constraint-retrieval-metric",
            label="Retrieval metric",
            detail=f"Retrieval seeds were selected using {_format_retrieval_metric_label(retrieval_metric)}.",
            state="pass",
        ),
    ]

    export_estimate = ExportEstimate(
        dimensions="160000 x 320000 px",
        fileSize="2.4 GB est.",
        tilePlan="512 tiles, 10k x 10k each",
        format="16-bit TIFF",
        progress=0,
        tileProgress="0 / 512 tiles",
    )

    return InverseDesignResult(
        candidates=candidates,
        constraints=constraints,
        export_estimate=export_estimate,
    )

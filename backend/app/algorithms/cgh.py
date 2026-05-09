from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..models import CandidateMetric, CandidateParameter, CandidateSolution, ConstraintCheck, ExportEstimate


# ---------------------------------------------------------------------------
# Wave propagation
# ---------------------------------------------------------------------------

def _transfer_function_asm(
    shape: tuple[int, int],
    wavelength_m: float,
    pixel_pitch_m: float,
    distance_m: float,
) -> np.ndarray:
    """Angular spectrum transfer function with evanescent-wave filtering."""
    H, W = shape
    ky = 2 * np.pi * np.fft.fftfreq(H, d=pixel_pitch_m)
    kx = 2 * np.pi * np.fft.fftfreq(W, d=pixel_pitch_m)
    KX, KY = np.meshgrid(kx, ky)
    k = 2 * np.pi / wavelength_m
    kz_sq = k**2 - KX**2 - KY**2
    kz = np.sqrt(np.maximum(kz_sq, 0.0).astype(np.complex128))
    H_tf = np.exp(1j * kz * distance_m)
    H_tf[kz_sq <= 0] = 0.0
    return np.fft.ifftshift(H_tf)


def _transfer_function_fresnel(
    shape: tuple[int, int],
    wavelength_m: float,
    pixel_pitch_m: float,
    distance_m: float,
) -> np.ndarray:
    """Fresnel (paraxial) transfer function."""
    H, W = shape
    ky = 2 * np.pi * np.fft.fftfreq(H, d=pixel_pitch_m)
    kx = 2 * np.pi * np.fft.fftfreq(W, d=pixel_pitch_m)
    KX, KY = np.meshgrid(kx, ky)
    k = 2 * np.pi / wavelength_m
    H_tf = np.exp(1j * k * distance_m) * np.exp(-1j * distance_m * (KX**2 + KY**2) / (2 * k))
    return np.fft.ifftshift(H_tf)


def _propagate(
    field: np.ndarray,
    transfer_function: np.ndarray,
    forward: bool = True,
) -> np.ndarray:
    """Core FFT-based propagation.  *forward* uses *H*; backward uses conj(H)."""
    H = transfer_function if forward else np.conj(transfer_function)
    F = np.fft.fft2(np.fft.ifftshift(field))
    return np.fft.fftshift(np.fft.ifft2(F * H))


def asm_propagate(
    field: np.ndarray,
    wavelength_m: float,
    pixel_pitch_m: float,
    distance_m: float,
    *,
    forward: bool = True,
) -> np.ndarray:
    H = _transfer_function_asm(field.shape, wavelength_m, pixel_pitch_m, distance_m)
    return _propagate(field, H, forward=forward)


def fresnel_propagate(
    field: np.ndarray,
    wavelength_m: float,
    pixel_pitch_m: float,
    distance_m: float,
    *,
    forward: bool = True,
) -> np.ndarray:
    H = _transfer_function_fresnel(field.shape, wavelength_m, pixel_pitch_m, distance_m)
    return _propagate(field, H, forward=forward)


# ---------------------------------------------------------------------------
# Target-image generation
# ---------------------------------------------------------------------------

def _hex_to_amplitude(target_hex: str) -> float:
    r = int(target_hex[1:3], 16) / 255.0
    g = int(target_hex[3:5], 16) / 255.0
    b = int(target_hex[5:7], 16) / 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return float(np.clip(luminance, 0.05, 1.0))


def generate_target_image(
    shape: tuple[int, int],
    target_hex: str,
    pattern: str = "siemens",
) -> np.ndarray:
    amplitude = _hex_to_amplitude(target_hex)
    H, W = shape
    y, x = np.mgrid[-H // 2 : H // 2, -W // 2 : W // 2]
    r = np.sqrt(x**2 + y**2)
    radius = min(H, W) // 2 - 2

    if pattern == "siemens":
        theta = np.arctan2(y, x) + np.pi
        n_spokes = 32
        sector = np.floor(theta * n_spokes / (2 * np.pi))
        target = np.where(sector % 2 == 0, amplitude, 0.0)
        target[r > radius] = 0.0
    elif pattern == "bars":
        bar_height = max(2, H // 16)
        target = np.zeros(shape, dtype=np.float64)
        for i in range(0, H, bar_height * 2):
            target[i : i + bar_height, :] = amplitude
    elif pattern == "rings":
        period = 8.0
        target = np.where(np.floor(r / period) % 2 == 0, amplitude, 0.0)
        target[r > radius] = 0.0
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return target.astype(np.float64)


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def psnr(target: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((target - reconstructed) ** 2)
    if mse == 0:
        return 100.0
    max_val = max(target.max(), reconstructed.max(), 1.0)
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def ssim(
    target: np.ndarray,
    reconstructed: np.ndarray,
    L: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    kernel = _gaussian_kernel(7, 1.5)

    mu_x = _convolve2d(target, kernel)
    mu_y = _convolve2d(reconstructed, kernel)
    mu_xx = _convolve2d(target**2, kernel)
    mu_yy = _convolve2d(reconstructed**2, kernel)
    mu_xy = _convolve2d(target * reconstructed, kernel)

    sigma_x = np.sqrt(np.maximum(mu_xx - mu_x**2, 0))
    sigma_y = np.sqrt(np.maximum(mu_yy - mu_y**2, 0))
    sigma_xy = mu_xy - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2)
    )
    return float(ssim_map.mean())


def _convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Valid-mode 2-D convolution via FFT (handles small kernels efficiently)."""
    H, W = arr.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2
    padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    f_shape = (padded.shape[0], padded.shape[1])
    F_arr = np.fft.rfft2(padded)
    F_kernel = np.fft.rfft2(np.flip(np.flip(kernel, 0), 1), s=f_shape)
    result = np.fft.irfft2(F_arr * F_kernel)
    return result[pad_h : pad_h + H, pad_w : pad_w + W]


# ---------------------------------------------------------------------------
# CGH algorithms
# ---------------------------------------------------------------------------

def run_gerchberg_saxton(
    target_amplitude: np.ndarray,
    wavelength_m: float,
    pixel_pitch_m: float,
    distance_m: float,
    iterations: int = 50,
    propagator: str = "asm",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    H = _transfer_function_asm(target_amplitude.shape, wavelength_m, pixel_pitch_m, distance_m) if propagator == "asm" else _transfer_function_fresnel(target_amplitude.shape, wavelength_m, pixel_pitch_m, distance_m)

    phase = rng.uniform(0, 2 * np.pi, size=target_amplitude.shape)
    source_field = np.exp(1j * phase)

    for _ in range(iterations):
        image_field = _propagate(source_field, H, forward=True)
        image_field = target_amplitude * np.exp(1j * np.angle(image_field))
        source_field = _propagate(image_field, H, forward=False)
        source_field = np.exp(1j * np.angle(source_field))

    phase_map = np.angle(source_field) % (2 * np.pi)
    replay_field = _propagate(np.exp(1j * phase_map), H, forward=True)
    replay_amplitude = np.abs(replay_field)
    return phase_map, replay_amplitude


def run_wirtinger_holography(
    target_amplitude: np.ndarray,
    wavelength_m: float,
    pixel_pitch_m: float,
    distance_m: float,
    iterations: int = 200,
    lr: float = 0.05,
    propagator: str = "asm",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    H = _transfer_function_asm(target_amplitude.shape, wavelength_m, pixel_pitch_m, distance_m) if propagator == "asm" else _transfer_function_fresnel(target_amplitude.shape, wavelength_m, pixel_pitch_m, distance_m)

    phase = rng.uniform(0, 2 * np.pi, size=target_amplitude.shape)
    momentum = np.zeros_like(phase)
    beta = 0.9

    for _ in range(iterations):
        u = np.exp(1j * phase)
        v = _propagate(u, H, forward=True)
        a = np.abs(v)
        scale = np.where(a > 1e-10, (target_amplitude - a) / a, 0.0)
        delta_v = scale * v
        back = _propagate(delta_v, H, forward=False)
        gradient = np.imag(np.conj(u) * back)
        momentum = beta * momentum + (1 - beta) * gradient
        phase = (phase + lr * momentum) % (2 * np.pi)

    u = np.exp(1j * phase)
    replay_field = _propagate(u, H, forward=True)
    replay_amplitude = np.abs(replay_field)
    return phase, replay_amplitude


def run_sgd_cgh(
    target_amplitude: np.ndarray,
    wavelength_m: float,
    pixel_pitch_m: float,
    distance_m: float,
    iterations: int = 300,
    lr: float = 0.01,
    batch_fraction: float = 0.1,
    propagator: str = "asm",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    H = _transfer_function_asm(target_amplitude.shape, wavelength_m, pixel_pitch_m, distance_m) if propagator == "asm" else _transfer_function_fresnel(target_amplitude.shape, wavelength_m, pixel_pitch_m, distance_m)

    phase = rng.uniform(0, 2 * np.pi, size=target_amplitude.shape)
    H_px, W_px = target_amplitude.shape
    total_pixels = H_px * W_px
    batch_size = max(1, int(batch_fraction * total_pixels))

    m = np.zeros_like(phase)
    v = np.zeros_like(phase)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    for t in range(iterations):
        u = np.exp(1j * phase)
        v_field = _propagate(u, H, forward=True)
        a = np.abs(v_field)

        scale = np.where(a > 1e-10, (target_amplitude - a) / a, 0.0)
        delta_v = scale * v_field
        back = _propagate(delta_v, H, forward=False)
        full_grad = np.imag(np.conj(u) * back)

        batch_mask = np.zeros_like(phase)
        indices = rng.choice(total_pixels, batch_size, replace=False)
        batch_mask.flat[indices] = 1.0
        gradient = full_grad * batch_mask

        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        m_hat = m / (1 - beta1 ** (t + 1))
        v_hat = v / (1 - beta2 ** (t + 1))
        phase = (phase + lr * m_hat / (np.sqrt(v_hat) + eps_adam)) % (2 * np.pi)

    u = np.exp(1j * phase)
    replay_field = _propagate(u, H, forward=True)
    replay_amplitude = np.abs(replay_field)
    return phase, replay_amplitude


# ---------------------------------------------------------------------------
# Phase-map storage
# ---------------------------------------------------------------------------

def _save_phase_map(phase_map: np.ndarray, output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_phase.npy"
    np.save(path, phase_map)
    return path


def _save_replay_preview(replay_amplitude: np.ndarray, output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{label}_replay.npy"
    np.save(path, replay_amplitude)
    return path


def _save_metadata(output_dir: Path, results: list[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

@dataclass
class CghDesignResult:
    candidates: list[CandidateSolution]
    constraints: list[ConstraintCheck]
    export_estimate: ExportEstimate


def _amplitude_to_hex(replay_amplitude: np.ndarray) -> str:
    mean_amp = float(np.mean(replay_amplitude))
    gray = int(np.clip(mean_amp * 255, 0, 255))
    return f"#{gray:02x}{gray:02x}{gray:02x}"


def run_cgh_design(
    target_hex: str,
    top_k: int = 3,
    theta_deg: float = 0.0,
    polarization: str = "unpolarized",
    image_size: int = 128,
    wavelength_m: float = 532e-9,
    pixel_pitch_m: float = 8e-6,
    distance_m: float = 0.1,
    output_dir: Path | None = None,
) -> CghDesignResult:
    shape = (image_size, image_size)
    target = generate_target_image(shape, target_hex)
    rng = np.random.default_rng(42)

    algo_specs = [
        ("cgh-gs", "Gerchberg-Saxton", run_gerchberg_saxton, {"iterations": 50}),
        ("cgh-wh", "Wirtinger Holography", run_wirtinger_holography, {"iterations": 200, "lr": 0.05}),
        ("cgh-sgd", "SGD CGH (Adam)", run_sgd_cgh, {"iterations": 300, "lr": 0.01, "batch_fraction": 0.1}),
    ]

    wavelength_nm = wavelength_m * 1e9
    algo_results: list[dict] = []

    for algo_id, algo_label, algo_fn, algo_kwargs in algo_specs:
        phase_map, replay = algo_fn(
            target_amplitude=target,
            wavelength_m=wavelength_m,
            pixel_pitch_m=pixel_pitch_m,
            distance_m=distance_m,
            propagator="asm",
            rng=rng,
            **algo_kwargs,
        )
        algo_results.append({
            "id": algo_id,
            "label": algo_label,
            "phase_map": phase_map,
            "replay": replay,
            "psnr": psnr(target, replay),
            "ssim": ssim(target, replay),
        })

    algo_results.sort(key=lambda r: r["ssim"], reverse=True)
    top_results = algo_results[:top_k]

    if output_dir is not None:
        metadata = []
        for r in top_results:
            _save_phase_map(r["phase_map"], output_dir, r["id"])
            _save_replay_preview(r["replay"], output_dir, r["id"])
            metadata.append({
                "id": r["id"],
                "label": r["label"],
                "psnr_db": round(r["psnr"], 2),
                "ssim": round(r["ssim"], 4),
            })
        _save_metadata(output_dir, metadata)

    candidates: list[CandidateSolution] = []
    for rank, r in enumerate(top_results, 1):
        replay_hex = _amplitude_to_hex(r["replay"])
        candidates.append(CandidateSolution(
            id=r["id"],
            rank=rank,
            group="CGH Simulation Baseline",
            selected=(rank == 1),
            status="Recommended" if rank == 1 else ("Robust" if rank == 2 else "Watch"),
            parameters=[
                CandidateParameter(label="Algorithm", value=r["label"]),
                CandidateParameter(label="Propagator", value="Angular Spectrum Method"),
                CandidateParameter(label="Image size", value=f"{image_size}x{image_size}"),
                CandidateParameter(label="Wavelength", value=f"{wavelength_nm:.0f} nm"),
                CandidateParameter(label="Pixel pitch", value=f"{pixel_pitch_m*1e6:.1f} μm"),
                CandidateParameter(label="Distance", value=f"{distance_m*1e3:.1f} mm"),
            ],
            metrics=[
                CandidateMetric(label="Source", value="CGH simulation (ASM)"),
                CandidateMetric(label="PSNR", value=f"{r['psnr']:.2f} dB"),
                CandidateMetric(label="SSIM", value=f"{r['ssim']:.4f}"),
                CandidateMetric(label="Quality gate", value="ideal-model PSNR"),
                CandidateMetric(label="Runtime", value="fast smoke"),
                CandidateMetric(label="Bench dependency", value="None"),
            ],
            targetColorHex=target_hex,
            simulatedColorHex=replay_hex,
            processPlusColorHex=replay_hex,
            processMinusColorHex=replay_hex,
            rationale=(
                f"{r['label']} replay achieves PSNR {r['psnr']:.1f} dB and SSIM {r['ssim']:.3f} "
                f"under ideal ASM propagation at {wavelength_nm:.0f} nm."
            ),
        ))

    constraints: list[ConstraintCheck] = [
        ConstraintCheck(
            id="cgh-target-image",
            label="Target image resolution",
            detail=f"CGH simulation uses a {image_size}x{image_size} Siemens star derived from {target_hex}.",
            state="pass",
        ),
        ConstraintCheck(
            id="cgh-propagation-model",
            label="Propagation model",
            detail="Angular Spectrum Method with ideal plane-wave illumination; real SLM may differ due to pixel crosstalk and fill factor.",
            state="warning",
        ),
        ConstraintCheck(
            id="cgh-phase-output",
            label="Phase map wrapping",
            detail="Phase maps wrapped to [0, 2π); SLM calibration needed before bench deployment.",
            state="pass",
        ),
        ConstraintCheck(
            id="cgh-bench-required",
            label="Bench verification",
            detail="Simulation-only metrics (PSNR/SSIM) are upper bounds; camera-in-the-loop capture is required for production quality gates.",
            state="warning",
        ),
    ]

    export_estimate = ExportEstimate(
        dimensions=f"{image_size} x {image_size} phase map",
        fileSize=f"{image_size * image_size * 8 // 1024} KB per frame",
        tilePlan="single frame + metadata manifest",
        format="NPY phase map + JSON metadata",
        progress=0,
        tileProgress="0 / 1 frames",
    )

    return CghDesignResult(
        candidates=candidates,
        constraints=constraints,
        export_estimate=export_estimate,
    )

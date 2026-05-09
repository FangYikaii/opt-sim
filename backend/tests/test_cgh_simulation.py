from __future__ import annotations

import numpy as np
import pytest

from backend.app.algorithms.cgh import (
    _transfer_function_asm,
    _transfer_function_fresnel,
    _propagate,
    asm_propagate,
    fresnel_propagate,
    generate_target_image,
    psnr,
    ssim,
    run_gerchberg_saxton,
    run_wirtinger_holography,
    run_sgd_cgh,
    _save_phase_map,
    _save_metadata,
    run_cgh_design,
)


# ---------------------------------------------------------------------------
# Propagation tests
# ---------------------------------------------------------------------------

def test_asm_propagation_preserves_energy():
    rng = np.random.default_rng(42)
    shape = (64, 64)
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    energy_in = float(np.sum(np.abs(field) ** 2))

    propagated = asm_propagate(field, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1)
    energy_out = float(np.sum(np.abs(propagated) ** 2))
    assert energy_out == pytest.approx(energy_in, rel=0.01)


def test_fresnel_propagation_preserves_energy():
    rng = np.random.default_rng(42)
    shape = (64, 64)
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    energy_in = float(np.sum(np.abs(field) ** 2))

    propagated = fresnel_propagate(field, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=1.0)
    energy_out = float(np.sum(np.abs(propagated) ** 2))
    assert energy_out == pytest.approx(energy_in, rel=0.01)


def test_asm_propagation_is_deterministic():
    rng = np.random.default_rng(0)
    shape = (32, 32)
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    result_a = asm_propagate(field, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.05)
    result_b = asm_propagate(field, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.05)
    np.testing.assert_allclose(result_a, result_b, atol=1e-14)


def test_propagation_roundtrip_reconstruction():
    """Forward + backward propagation should recover the original field."""
    rng = np.random.default_rng(0)
    shape = (64, 64)
    field = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    forwarded = asm_propagate(field, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1, forward=True)
    recovered = asm_propagate(forwarded, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1, forward=False)

    # Zero out evanescent-wave components that were lost
    H = _transfer_function_asm(shape, 532e-9, 8e-6, 0.1)
    F = np.fft.fft2(np.fft.ifftshift(field))
    F_filtered = F * (np.abs(H) > 0)
    field_filtered = np.fft.fftshift(np.fft.ifft2(F_filtered))

    np.testing.assert_allclose(np.abs(recovered), np.abs(field_filtered), atol=1e-6)


def test_evanescent_waves_are_filtered():
    """ASM transfer function should zero out spatial frequencies beyond the diffraction limit.

    Use sub-wavelength pixel pitch so that the highest sampled spatial frequencies
    lie beyond the k=2*pi/lambda wavenumber, triggering evanescent cutoff."""
    shape = (64, 64)
    wavelength_m = 532e-9
    pixel_pitch_m = wavelength_m * 0.6  # sub-wavelength sampling
    H = _transfer_function_asm(shape, wavelength_m, pixel_pitch_m, 0.1)
    # _transfer_function_asm returns ifftshift(H_tf).  fftshift puts us back
    # in natural FFT order where DC is at (0,0) and the Nyquist region sits
    # near (N/2, N/2).  Evanescent waves appear in that high-frequency band.
    H_nat = np.fft.fftshift(H)
    N = shape[0]
    nyq_region = H_nat[N // 2 - 2 : N // 2 + 2, N // 2 - 2 : N // 2 + 2]
    assert np.any(np.abs(nyq_region) == 0.0)


# ---------------------------------------------------------------------------
# Target image tests
# ---------------------------------------------------------------------------

def test_target_image_shape_and_range():
    target = generate_target_image((128, 128), "#FF8040", pattern="siemens")
    assert target.shape == (128, 128)
    assert target.min() >= 0.0
    assert target.max() <= 1.0


def test_target_image_has_structure():
    target = generate_target_image((128, 128), "#FF8040", pattern="siemens")
    assert float(np.std(target)) > 0.05  # not uniform


def test_target_image_different_hex_produce_different_amplitudes():
    t_bright = generate_target_image((64, 64), "#FFFFFF", pattern="bars")
    t_dim = generate_target_image((64, 64), "#111111", pattern="bars")
    assert t_bright.max() > t_dim.max()


def test_target_image_bars_pattern():
    target = generate_target_image((64, 64), "#FFFFFF", pattern="bars")
    assert float(np.std(target)) > 0.05


def test_target_image_rings_pattern():
    target = generate_target_image((64, 64), "#FFFFFF", pattern="rings")
    assert float(np.std(target)) > 0.05


# ---------------------------------------------------------------------------
# PSNR / SSIM tests
# ---------------------------------------------------------------------------

def test_psnr_identical():
    img = np.random.default_rng(0).uniform(0, 1, size=(32, 32))
    assert psnr(img, img) == 100.0


def test_psnr_noise_floor():
    rng = np.random.default_rng(0)
    target = np.ones((32, 32)) * 0.5
    noise = rng.uniform(0, 1, size=(32, 32))
    assert psnr(target, noise) < 15.0


def test_ssim_identical_is_one():
    img = np.random.default_rng(0).uniform(0.3, 0.7, size=(64, 64))
    result = ssim(img, img)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_ssim_negated_is_low():
    """SSIM between an image and its photometric negative should be far from 1."""
    rng = np.random.default_rng(0)
    img = rng.uniform(0.3, 0.7, size=(64, 64))
    result = ssim(img, 1.0 - img)
    assert result < 0.5


# ---------------------------------------------------------------------------
# Gerchberg-Saxton tests
# ---------------------------------------------------------------------------

def test_gerchberg_saxton_converges():
    target = generate_target_image((64, 64), "#FF8040", pattern="siemens")
    phase_map, replay = run_gerchberg_saxton(
        target, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1,
        iterations=50, propagator="asm", rng=np.random.default_rng(42),
    )
    p = psnr(target, replay)
    assert p > 5.0, f"GS PSNR {p:.1f} dB <= 5.0 dB threshold"


def test_gs_deterministic_replay():
    target = generate_target_image((32, 32), "#FF8040", pattern="siemens")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    p1, r1 = run_gerchberg_saxton(target, 532e-9, 8e-6, 0.1, iterations=10, rng=rng1)
    p2, r2 = run_gerchberg_saxton(target, 532e-9, 8e-6, 0.1, iterations=10, rng=rng2)
    np.testing.assert_allclose(p1, p2, atol=1e-14)
    np.testing.assert_allclose(r1, r2, atol=1e-14)


def test_gs_phase_in_range():
    target = generate_target_image((32, 32), "#FF8040", pattern="siemens")
    phase_map, _ = run_gerchberg_saxton(
        target, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1,
        iterations=10, rng=np.random.default_rng(42),
    )
    assert phase_map.min() >= 0.0
    assert phase_map.max() < 2 * np.pi + 1e-10


# ---------------------------------------------------------------------------
# Wirtinger Holography tests
# ---------------------------------------------------------------------------

def test_wirtinger_improves_over_gs():
    target = generate_target_image((32, 32), "#FF8040", pattern="siemens")
    _, replay_gs = run_gerchberg_saxton(
        target, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1,
        iterations=50, rng=np.random.default_rng(42),
    )
    _, replay_wh = run_wirtinger_holography(
        target, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1,
        iterations=200, lr=0.05, rng=np.random.default_rng(42),
    )
    assert psnr(target, replay_wh) > psnr(target, replay_gs)


def test_wirtinger_loss_decreases():
    """Loss should decrease over iterations (tested on a small target)."""
    target = generate_target_image((32, 32), "#FF8040", pattern="siemens")
    rng = np.random.default_rng(42)
    H = _transfer_function_asm(target.shape, 532e-9, 8e-6, 0.1)
    phase = rng.uniform(0, 2 * np.pi, size=target.shape)
    momentum = np.zeros_like(phase)
    beta = 0.9
    lr = 0.05

    losses = []
    for _ in range(100):
        u = np.exp(1j * phase)
        v = _propagate(u, H, forward=True)
        a = np.abs(v)
        losses.append(float(np.mean((a - target) ** 2)))
        scale = np.where(a > 1e-10, (target - a) / a, 0.0)
        delta_v = scale * v
        back = _propagate(delta_v, H, forward=False)
        gradient = np.imag(np.conj(u) * back)
        momentum = beta * momentum + (1 - beta) * gradient
        phase = (phase + lr * momentum) % (2 * np.pi)

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"


# ---------------------------------------------------------------------------
# SGD CGH tests
# ---------------------------------------------------------------------------

def test_sgd_produces_valid_phase():
    target = generate_target_image((32, 32), "#FF8040", pattern="siemens")
    phase_map, replay = run_sgd_cgh(
        target, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1,
        iterations=150, lr=0.01, batch_fraction=0.2, rng=np.random.default_rng(42),
    )
    assert phase_map.min() >= 0.0
    assert phase_map.max() < 2 * np.pi + 1e-10
    assert replay.min() >= 0.0
    # Replay amplitude can exceed 1.0 due to constructive interference from
    # a phase-only unit-amplitude SLM across many pixels.
    assert psnr(target, replay) > 3.0


def test_sgd_beats_or_matches_gs():
    target = generate_target_image((32, 32), "#FF8040", pattern="siemens")
    _, replay_gs = run_gerchberg_saxton(
        target, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1,
        iterations=50, rng=np.random.default_rng(42),
    )
    _, replay_sgd = run_sgd_cgh(
        target, wavelength_m=532e-9, pixel_pitch_m=8e-6, distance_m=0.1,
        iterations=300, lr=0.01, batch_fraction=0.1, rng=np.random.default_rng(42),
    )
    assert psnr(target, replay_sgd) > psnr(target, replay_gs)


# ---------------------------------------------------------------------------
# Phase-map storage tests
# ---------------------------------------------------------------------------

def test_phase_map_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    phase_map = rng.uniform(0, 2 * np.pi, size=(32, 32))
    path = _save_phase_map(phase_map, tmp_path, "test-algo")
    loaded = np.load(path)
    np.testing.assert_allclose(loaded, phase_map, atol=1e-14)


def test_metadata_written(tmp_path):
    results = [{"id": "algo-1", "psnr_db": 15.2, "ssim": 0.85}]
    path = _save_metadata(tmp_path, results)
    assert path.exists()


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

def test_run_cgh_design_returns_top_k_candidates():
    result = run_cgh_design("#FF8040", top_k=3, image_size=32, distance_m=0.05)
    assert len(result.candidates) == 3
    assert result.candidates[0].rank == 1
    assert result.candidates[0].selected is True
    assert result.candidates[1].selected is False


def test_run_cgh_design_candidates_have_metrics():
    result = run_cgh_design("#6f8fd8", top_k=2, image_size=32, distance_m=0.05)
    for c in result.candidates:
        metric_labels = {m.label for m in c.metrics}
        assert "PSNR" in metric_labels
        assert "SSIM" in metric_labels
        assert "Source" in metric_labels
        assert c.targetColorHex == "#6f8fd8"


def test_run_cgh_design_constraints_present():
    result = run_cgh_design("#6f8fd8", top_k=1, image_size=32, distance_m=0.05)
    assert len(result.constraints) >= 2
    states = {c.state for c in result.constraints}
    assert "pass" in states


def test_run_cgh_design_export_estimate():
    result = run_cgh_design("#6f8fd8", top_k=1, image_size=32, distance_m=0.05)
    assert result.export_estimate.progress == 0
    assert "phase map" in result.export_estimate.format.lower()


def test_run_cgh_design_top_k_1():
    result = run_cgh_design("#FF8040", top_k=1, image_size=32, distance_m=0.05)
    assert len(result.candidates) == 1


def test_run_cgh_design_with_output_dir(tmp_path):
    result = run_cgh_design("#FF8040", top_k=2, image_size=32, distance_m=0.05, output_dir=tmp_path)
    npy_files = list(tmp_path.glob("*.npy"))
    assert len(npy_files) >= 4
    assert (tmp_path / "metadata.json").exists()


def test_run_cgh_design_ssim_ordering():
    """Candidates should be ordered by descending SSIM."""
    result = run_cgh_design("#FF8040", top_k=3, image_size=32, distance_m=0.05)
    ssims = []
    for c in result.candidates:
        for m in c.metrics:
            if m.label == "SSIM":
                ssims.append(float(m.value))
    assert ssims == sorted(ssims, reverse=True), f"SSIM not sorted descending: {ssims}"

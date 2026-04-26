from __future__ import annotations

import colour
import numpy as np
import pytest

from backend.app.algorithms.colorimetry_data import get_colorimetry_reference
from backend.app.algorithms.optics import (
    WAVELENGTHS_NM,
    delta_e_2000,
    spectrum_to_xyz,
    xyz_to_lab,
    xyz_to_srgb_hex,
)


def test_spectrum_to_xyz_matches_manual_reference_integration() -> None:
    reference = get_colorimetry_reference(WAVELENGTHS_NM)
    spectrum = np.ones_like(WAVELENGTHS_NM, dtype=np.float64)

    xyz = spectrum_to_xyz(spectrum)
    normalizer = float(np.sum(reference.d65 * reference.y_bar))
    expected_xyz = np.array(
        [
            float(np.sum(reference.d65 * reference.x_bar)) / normalizer,
            float(np.sum(reference.d65 * reference.y_bar)) / normalizer,
            float(np.sum(reference.d65 * reference.z_bar)) / normalizer,
        ],
        dtype=np.float64,
    )

    assert np.allclose(xyz, expected_xyz)
    assert xyz[1] == pytest.approx(1.0)


def test_reference_white_maps_to_neutral_lab() -> None:
    spectrum = np.ones_like(WAVELENGTHS_NM, dtype=np.float64)

    lab = xyz_to_lab(spectrum_to_xyz(spectrum))

    assert lab[0] == pytest.approx(100.0, abs=1e-7)
    assert lab[1] == pytest.approx(0.0, abs=1e-7)
    assert lab[2] == pytest.approx(0.0, abs=1e-7)


def test_reference_white_maps_to_white_srgb_hex() -> None:
    spectrum = np.ones_like(WAVELENGTHS_NM, dtype=np.float64)

    srgb_hex = xyz_to_srgb_hex(spectrum_to_xyz(spectrum))

    assert srgb_hex == "#ffffff"


def test_delta_e_2000_is_zero_for_identical_labs() -> None:
    lab = np.array([53.23288179, 80.11117774, 67.22370367], dtype=np.float64)

    assert delta_e_2000(lab, lab) == pytest.approx(0.0)


def test_xyz_to_lab_matches_known_colour_science_reference_example() -> None:
    xyz = np.array([0.20654008, 0.12197225, 0.05136952], dtype=np.float64)
    reference = get_colorimetry_reference(WAVELENGTHS_NM)
    normalizer = float(np.sum(reference.d65 * reference.y_bar))
    ref_white = np.array(
        [
            float(np.sum(reference.d65 * reference.x_bar)) / normalizer,
            1.0,
            float(np.sum(reference.d65 * reference.z_bar)) / normalizer,
        ],
        dtype=np.float64,
    )
    illuminant_xy = ref_white[:2] / float(np.sum(ref_white))

    lab = xyz_to_lab(xyz)
    expected_lab = colour.XYZ_to_Lab(xyz, illuminant=illuminant_xy)

    assert np.allclose(lab, expected_lab)

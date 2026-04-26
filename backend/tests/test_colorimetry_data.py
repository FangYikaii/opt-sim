from __future__ import annotations

import numpy as np
import pytest

from backend.app.algorithms.colorimetry_data import (
    DEFAULT_WAVELENGTHS_NM,
    get_colorimetry_reference,
    load_d65_reference,
    load_tristimulus_reference,
)


def test_load_d65_reference_reads_expected_native_grid() -> None:
    reference = load_d65_reference()

    assert len(reference.wavelengths_nm) == 401
    assert reference.wavelengths_nm[0] == pytest.approx(380.0)
    assert reference.wavelengths_nm[-1] == pytest.approx(780.0)
    assert np.allclose(np.diff(reference.wavelengths_nm), 1.0)
    assert reference.values[0] == pytest.approx(49.9755)
    assert reference.values[-1] == pytest.approx(63.3828)


def test_load_tristimulus_reference_reads_expected_native_grid_and_d65_column() -> None:
    reference = load_tristimulus_reference()

    assert len(reference.wavelengths_nm) == 81
    assert reference.wavelengths_nm[0] == pytest.approx(380.0)
    assert reference.wavelengths_nm[-1] == pytest.approx(780.0)
    assert np.allclose(np.diff(reference.wavelengths_nm), 5.0)
    assert reference.x_bar[0] == pytest.approx(0.001368)
    assert reference.y_bar[0] == pytest.approx(0.000039)
    assert reference.z_bar[0] == pytest.approx(0.00645)
    assert reference.d65[0] == pytest.approx(49.9755)
    assert reference.d65[-1] == pytest.approx(63.3828)


def test_get_colorimetry_reference_default_grid_matches_tristimulus_samples() -> None:
    reference = get_colorimetry_reference()
    tristimulus_reference = load_tristimulus_reference()

    assert np.allclose(reference.wavelengths_nm, DEFAULT_WAVELENGTHS_NM)
    assert np.allclose(reference.wavelengths_nm, tristimulus_reference.wavelengths_nm)
    assert np.allclose(reference.x_bar, tristimulus_reference.x_bar)
    assert np.allclose(reference.y_bar, tristimulus_reference.y_bar)
    assert np.allclose(reference.z_bar, tristimulus_reference.z_bar)
    assert np.allclose(reference.d65, tristimulus_reference.d65)


def test_get_colorimetry_reference_rejects_out_of_range_grid() -> None:
    with pytest.raises(ValueError, match="outside"):
        get_colorimetry_reference(np.array([379.0, 380.0, 385.0], dtype=np.float64))


def test_load_d65_reference_rejects_non_contiguous_wavelengths(tmp_path) -> None:
    dataset_path = tmp_path / "bad_d65.csv"
    dataset_path.write_text(
        "380,1.0\n382,2.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        load_d65_reference(dataset_path)


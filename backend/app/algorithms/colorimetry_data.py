from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import csv

import numpy as np


DEFAULT_WAVELENGTHS_NM = np.arange(380.0, 781.0, 5.0, dtype=np.float64)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
D65_SOURCE_PATH = PROJECT_ROOT / "refer_data" / "D65.csv"
TRISTIMULUS_SOURCE_PATH = PROJECT_ROOT / "refer_data" / "tristimulus.csv"
_D65_STEP_NM = 1.0
_TRISTIMULUS_STEP_NM = 5.0
_D65_MATCH_TOLERANCE = 1e-9


@dataclass(frozen=True)
class D65Reference:
    wavelengths_nm: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class TristimulusReference:
    wavelengths_nm: np.ndarray
    x_bar: np.ndarray
    y_bar: np.ndarray
    z_bar: np.ndarray
    d65: np.ndarray


@dataclass(frozen=True)
class ColorimetryReference:
    wavelengths_nm: np.ndarray
    x_bar: np.ndarray
    y_bar: np.ndarray
    z_bar: np.ndarray
    d65: np.ndarray


def _read_numeric_csv(path: Path, expected_columns: int) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row_index, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != expected_columns:
                raise ValueError(
                    f"{path} row {row_index} has {len(row)} columns; expected {expected_columns}."
                )
            try:
                rows.append([float(cell) for cell in row])
            except ValueError as exc:
                raise ValueError(f"{path} row {row_index} contains non-numeric data.") from exc

    if not rows:
        raise ValueError(f"{path} does not contain any numeric rows.")

    return np.array(rows, dtype=np.float64)


def _validate_wavelength_grid(
    wavelengths_nm: np.ndarray,
    *,
    expected_step_nm: float,
    label: str,
) -> None:
    if wavelengths_nm.ndim != 1 or len(wavelengths_nm) == 0:
        raise ValueError(f"{label} wavelengths must be a non-empty 1D array.")
    if not np.all(np.diff(wavelengths_nm) > 0.0):
        raise ValueError(f"{label} wavelengths must be strictly increasing.")
    if not np.allclose(np.diff(wavelengths_nm), expected_step_nm):
        raise ValueError(
            f"{label} wavelengths must be contiguous with a {expected_step_nm:.1f} nm step."
        )


@lru_cache(maxsize=4)
def load_d65_reference(path: Path = D65_SOURCE_PATH) -> D65Reference:
    dataset = _read_numeric_csv(Path(path), expected_columns=2)
    wavelengths_nm = dataset[:, 0]
    values = dataset[:, 1]
    _validate_wavelength_grid(
        wavelengths_nm,
        expected_step_nm=_D65_STEP_NM,
        label="D65",
    )
    return D65Reference(
        wavelengths_nm=wavelengths_nm,
        values=values,
    )


@lru_cache(maxsize=4)
def load_tristimulus_reference(path: Path = TRISTIMULUS_SOURCE_PATH) -> TristimulusReference:
    dataset = _read_numeric_csv(Path(path), expected_columns=5)
    wavelengths_nm = dataset[:, 0]
    _validate_wavelength_grid(
        wavelengths_nm,
        expected_step_nm=_TRISTIMULUS_STEP_NM,
        label="tristimulus",
    )
    return TristimulusReference(
        wavelengths_nm=wavelengths_nm,
        x_bar=dataset[:, 1],
        y_bar=dataset[:, 2],
        z_bar=dataset[:, 3],
        d65=dataset[:, 4],
    )


@lru_cache(maxsize=1)
def _validate_reference_files() -> None:
    d65_reference = load_d65_reference()
    tristimulus_reference = load_tristimulus_reference()
    resampled_d65 = np.interp(
        tristimulus_reference.wavelengths_nm,
        d65_reference.wavelengths_nm,
        d65_reference.values,
    )
    if not np.allclose(resampled_d65, tristimulus_reference.d65, atol=_D65_MATCH_TOLERANCE):
        raise ValueError(
            "D65.csv does not match the D65 column embedded in tristimulus.csv at the shared 5 nm grid."
        )


def _coerce_wavelength_key(wavelengths_nm: np.ndarray | None) -> tuple[float, ...]:
    if wavelengths_nm is None:
        return tuple(float(value) for value in DEFAULT_WAVELENGTHS_NM.tolist())

    requested = np.asarray(wavelengths_nm, dtype=np.float64)
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError("Requested wavelengths must be a non-empty 1D array.")
    if not np.all(np.diff(requested) > 0.0):
        raise ValueError("Requested wavelengths must be strictly increasing.")
    return tuple(float(value) for value in requested.tolist())


@lru_cache(maxsize=8)
def _get_colorimetry_reference_cached(wavelengths_key: tuple[float, ...]) -> ColorimetryReference:
    _validate_reference_files()
    d65_reference = load_d65_reference()
    tristimulus_reference = load_tristimulus_reference()
    target_wavelengths_nm = np.array(wavelengths_key, dtype=np.float64)

    lower_bound = max(
        float(d65_reference.wavelengths_nm[0]),
        float(tristimulus_reference.wavelengths_nm[0]),
    )
    upper_bound = min(
        float(d65_reference.wavelengths_nm[-1]),
        float(tristimulus_reference.wavelengths_nm[-1]),
    )
    if np.any(target_wavelengths_nm < lower_bound) or np.any(target_wavelengths_nm > upper_bound):
        raise ValueError(
            f"Requested wavelengths fall outside the supported [{lower_bound:.1f}, {upper_bound:.1f}] nm range."
        )

    return ColorimetryReference(
        wavelengths_nm=target_wavelengths_nm,
        x_bar=np.interp(
            target_wavelengths_nm,
            tristimulus_reference.wavelengths_nm,
            tristimulus_reference.x_bar,
        ),
        y_bar=np.interp(
            target_wavelengths_nm,
            tristimulus_reference.wavelengths_nm,
            tristimulus_reference.y_bar,
        ),
        z_bar=np.interp(
            target_wavelengths_nm,
            tristimulus_reference.wavelengths_nm,
            tristimulus_reference.z_bar,
        ),
        d65=np.interp(
            target_wavelengths_nm,
            d65_reference.wavelengths_nm,
            d65_reference.values,
        ),
    )


def get_colorimetry_reference(
    wavelengths_nm: np.ndarray | None = None,
) -> ColorimetryReference:
    return _get_colorimetry_reference_cached(_coerce_wavelength_key(wavelengths_nm))


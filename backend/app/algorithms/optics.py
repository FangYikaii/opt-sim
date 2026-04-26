from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import colour
import numpy as np

from .colorimetry_data import get_colorimetry_reference


WAVELENGTHS_NM = np.arange(380.0, 781.0, 5.0)
Polarization = Literal["te", "tm", "unpolarized"]


@dataclass(frozen=True)
class Material:
    name: str
    refractive_index: complex


AIR = Material("air", 1.0 + 0.0j)
SIO2 = Material("SiO2", 1.46 + 0.0j)
AG = Material("Ag", 0.13 - 3.98j)
QUARTZ = Material("quartz", 1.46 + 0.0j)


def _snell_cosine(refractive_index: complex, incident_index: complex, sin_incident: complex) -> complex:
    sin_theta = incident_index * sin_incident / refractive_index
    cos_theta = np.sqrt(1.0 - sin_theta * sin_theta + 0.0j)
    if np.real(cos_theta) < 0.0:
        cos_theta = -cos_theta
    if abs(np.real(cos_theta)) < 1e-12 and np.imag(cos_theta) < 0.0:
        cos_theta = -cos_theta
    return cos_theta


def _admittance(refractive_index: complex, cos_theta: complex, polarization: Literal["te", "tm"]) -> complex:
    if polarization == "te":
        return refractive_index * cos_theta
    return refractive_index / cos_theta


def _characteristic_matrix(
    refractive_index: complex,
    thickness_nm: float,
    wavelength_nm: float,
    *,
    incident_index: complex,
    sin_incident: complex,
    polarization: Literal["te", "tm"],
) -> np.ndarray:
    cos_theta = _snell_cosine(refractive_index, incident_index, sin_incident)
    delta = 2.0 * np.pi * refractive_index * cos_theta * thickness_nm / wavelength_nm
    eta = _admittance(refractive_index, cos_theta, polarization)
    return np.array(
        [
            [np.cos(delta), 1j * np.sin(delta) / eta],
            [1j * eta * np.sin(delta), np.cos(delta)],
        ],
        dtype=np.complex128,
    )


def _stack_amplitudes(
    layers: list[tuple[complex, float]],
    wavelength_nm: float,
    *,
    incident_index: complex,
    substrate_index: complex,
    theta_deg: float,
    polarization: Literal["te", "tm"],
) -> tuple[complex, complex]:
    incident_theta = np.deg2rad(theta_deg)
    sin_incident = np.sin(incident_theta) + 0.0j
    matrix = np.identity(2, dtype=np.complex128)
    for refractive_index, thickness_nm in layers:
        matrix = matrix @ _characteristic_matrix(
            refractive_index,
            thickness_nm,
            wavelength_nm,
            incident_index=incident_index,
            sin_incident=sin_incident,
            polarization=polarization,
        )

    cos_theta_0 = _snell_cosine(incident_index, incident_index, sin_incident)
    cos_theta_s = _snell_cosine(substrate_index, incident_index, sin_incident)
    eta0 = _admittance(incident_index, cos_theta_0, polarization)
    etas = _admittance(substrate_index, cos_theta_s, polarization)
    b = matrix[0, 0] + matrix[0, 1] * etas
    c = matrix[1, 0] + matrix[1, 1] * etas
    denominator = eta0 * b + c
    reflection = (eta0 * b - c) / denominator
    transmission = (2.0 * eta0) / denominator
    return reflection, transmission


def _stack_response(
    layers: list[tuple[complex, float]],
    *,
    incident_index: complex,
    substrate_index: complex,
    theta_deg: float,
    polarization: Literal["te", "tm"],
) -> tuple[np.ndarray, np.ndarray]:
    reflectance: list[float] = []
    transmittance: list[float] = []
    incident_theta = np.deg2rad(theta_deg)
    sin_incident = np.sin(incident_theta) + 0.0j
    cos_theta_0 = _snell_cosine(incident_index, incident_index, sin_incident)
    cos_theta_s = _snell_cosine(substrate_index, incident_index, sin_incident)
    eta0 = _admittance(incident_index, cos_theta_0, polarization)
    etas = _admittance(substrate_index, cos_theta_s, polarization)

    for wavelength_nm in WAVELENGTHS_NM:
        reflection, transmission = _stack_amplitudes(
            layers,
            wavelength_nm,
            incident_index=incident_index,
            substrate_index=substrate_index,
            theta_deg=theta_deg,
            polarization=polarization,
        )
        reflectance.append(float(np.clip(np.abs(reflection) ** 2, 0.0, 1.0)))
        power_ratio = float(np.real(etas / eta0))
        transmittance.append(float(np.clip(power_ratio * (np.abs(transmission) ** 2), 0.0, 1.0)))

    return (
        np.array(reflectance, dtype=np.float64),
        np.array(transmittance, dtype=np.float64),
    )


def fabry_perot_response_ag_sio2_ag(
    d_ag_bottom_nm: float,
    d_sio2_nm: float,
    d_ag_top_nm: float,
    *,
    theta_deg: float = 0.0,
    polarization: Polarization = "unpolarized",
) -> tuple[np.ndarray, np.ndarray]:
    layers = [
        (AG.refractive_index, d_ag_bottom_nm),
        (SIO2.refractive_index, d_sio2_nm),
        (AG.refractive_index, d_ag_top_nm),
    ]

    if polarization == "unpolarized":
        reflectance_te, transmittance_te = _stack_response(
            layers,
            incident_index=AIR.refractive_index,
            substrate_index=QUARTZ.refractive_index,
            theta_deg=theta_deg,
            polarization="te",
        )
        reflectance_tm, transmittance_tm = _stack_response(
            layers,
            incident_index=AIR.refractive_index,
            substrate_index=QUARTZ.refractive_index,
            theta_deg=theta_deg,
            polarization="tm",
        )
        return (
            0.5 * (reflectance_te + reflectance_tm),
            0.5 * (transmittance_te + transmittance_tm),
        )

    return _stack_response(
        layers,
        incident_index=AIR.refractive_index,
        substrate_index=QUARTZ.refractive_index,
        theta_deg=theta_deg,
        polarization=polarization,
    )


def reflectance_spectrum_ag_sio2_ag(
    d_ag_bottom_nm: float,
    d_sio2_nm: float,
    d_ag_top_nm: float,
    *,
    theta_deg: float = 0.0,
    polarization: Polarization = "unpolarized",
) -> np.ndarray:
    reflectance, _ = fabry_perot_response_ag_sio2_ag(
        d_ag_bottom_nm,
        d_sio2_nm,
        d_ag_top_nm,
        theta_deg=theta_deg,
        polarization=polarization,
    )
    return reflectance


def transmittance_spectrum_ag_sio2_ag(
    d_ag_bottom_nm: float,
    d_sio2_nm: float,
    d_ag_top_nm: float,
    *,
    theta_deg: float = 0.0,
    polarization: Polarization = "unpolarized",
) -> np.ndarray:
    _, transmittance = fabry_perot_response_ag_sio2_ag(
        d_ag_bottom_nm,
        d_sio2_nm,
        d_ag_top_nm,
        theta_deg=theta_deg,
        polarization=polarization,
    )
    return transmittance


_COLORIMETRY_REFERENCE = get_colorimetry_reference(WAVELENGTHS_NM)
_XYZ_NORMALIZER = float(np.sum(_COLORIMETRY_REFERENCE.d65 * _COLORIMETRY_REFERENCE.y_bar))
_REF_WHITE = np.array(
    [
        float(np.sum(_COLORIMETRY_REFERENCE.d65 * _COLORIMETRY_REFERENCE.x_bar)) / _XYZ_NORMALIZER,
        1.0,
        float(np.sum(_COLORIMETRY_REFERENCE.d65 * _COLORIMETRY_REFERENCE.z_bar)) / _XYZ_NORMALIZER,
    ],
    dtype=np.float64,
)
_REF_WHITE_XY = _REF_WHITE[:2] / float(np.sum(_REF_WHITE))


def spectrum_to_xyz(spectrum: np.ndarray) -> np.ndarray:
    weighted = spectrum * _COLORIMETRY_REFERENCE.d65
    x = float(np.sum(weighted * _COLORIMETRY_REFERENCE.x_bar)) / _XYZ_NORMALIZER
    y = float(np.sum(weighted * _COLORIMETRY_REFERENCE.y_bar)) / _XYZ_NORMALIZER
    z = float(np.sum(weighted * _COLORIMETRY_REFERENCE.z_bar)) / _XYZ_NORMALIZER
    return np.array([x, y, z], dtype=np.float64)


def _lab_f(value: float) -> float:
    delta = 6.0 / 29.0
    if value > delta**3:
        return value ** (1.0 / 3.0)
    return value / (3 * delta * delta) + 4.0 / 29.0


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    # colour.XYZ_to_Lab expects XYZ scaled to [0, 1] under the provided illuminant xy.
    return np.array(
        colour.XYZ_to_Lab(
            np.asarray(xyz, dtype=np.float64),
            illuminant=_REF_WHITE_XY,
        ),
        dtype=np.float64,
    )


def xyz_to_srgb_hex(xyz: np.ndarray) -> str:
    rgb = np.clip(
        colour.XYZ_to_sRGB(
            np.asarray(xyz, dtype=np.float64),
            illuminant=_REF_WHITE_XY,
            apply_cctf_encoding=True,
        ),
        0.0,
        1.0,
    )
    rgb_int = [int(round(float(v) * 255.0)) for v in rgb]
    return "#{:02x}{:02x}{:02x}".format(*rgb_int)


def hex_to_lab(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    rgb = np.array(
        [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)],
        dtype=np.float64,
    )

    def linearize(channel: float) -> float:
        if channel <= 0.04045:
            return channel / 12.92
        return ((channel + 0.055) / 1.055) ** 2.4

    rgb_linear = np.array([linearize(float(v)) for v in rgb], dtype=np.float64)
    inv_matrix = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ],
        dtype=np.float64,
    )
    xyz = inv_matrix @ rgb_linear
    return xyz_to_lab(xyz)


def delta_e_76(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    return float(np.linalg.norm(lab_a - lab_b))


def delta_e_2000(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    return float(
        colour.delta_E(
            np.asarray(lab_a, dtype=np.float64),
            np.asarray(lab_b, dtype=np.float64),
            method="CIE 2000",
        )
    )

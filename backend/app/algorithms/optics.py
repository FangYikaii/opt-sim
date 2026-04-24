from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WAVELENGTHS_NM = np.arange(380.0, 781.0, 5.0)


@dataclass(frozen=True)
class Material:
    name: str
    refractive_index: complex


AIR = Material("air", 1.0 + 0.0j)
SIO2 = Material("SiO2", 1.46 + 0.0j)
AG = Material("Ag", 0.13 - 3.98j)
QUARTZ = Material("quartz", 1.46 + 0.0j)


def _characteristic_matrix(refractive_index: complex, thickness_nm: float, wavelength_nm: float) -> np.ndarray:
    delta = 2.0 * np.pi * refractive_index * thickness_nm / wavelength_nm
    eta = refractive_index
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
) -> tuple[complex, complex]:
    matrix = np.identity(2, dtype=np.complex128)
    for refractive_index, thickness_nm in layers:
        matrix = matrix @ _characteristic_matrix(refractive_index, thickness_nm, wavelength_nm)

    eta0 = incident_index
    etas = substrate_index
    b = matrix[0, 0] + matrix[0, 1] * etas
    c = matrix[1, 0] + matrix[1, 1] * etas
    denominator = eta0 * b + c
    reflection = (eta0 * b - c) / denominator
    transmission = (2.0 * eta0) / denominator
    return reflection, transmission


def fabry_perot_response_ag_sio2_ag(
    d_ag_bottom_nm: float,
    d_sio2_nm: float,
    d_ag_top_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    layers = [
        (AG.refractive_index, d_ag_bottom_nm),
        (SIO2.refractive_index, d_sio2_nm),
        (AG.refractive_index, d_ag_top_nm),
    ]

    reflectance: list[float] = []
    transmittance: list[float] = []
    for wavelength_nm in WAVELENGTHS_NM:
        reflection, transmission = _stack_amplitudes(
            layers,
            wavelength_nm,
            incident_index=AIR.refractive_index,
            substrate_index=QUARTZ.refractive_index,
        )
        reflectance.append(float(np.clip(np.abs(reflection) ** 2, 0.0, 1.0)))
        power_ratio = float(np.real(QUARTZ.refractive_index / AIR.refractive_index))
        transmittance.append(float(np.clip(power_ratio * (np.abs(transmission) ** 2), 0.0, 1.0)))

    return (
        np.array(reflectance, dtype=np.float64),
        np.array(transmittance, dtype=np.float64),
    )


def reflectance_spectrum_ag_sio2_ag(
    d_ag_bottom_nm: float,
    d_sio2_nm: float,
    d_ag_top_nm: float,
) -> np.ndarray:
    reflectance, _ = fabry_perot_response_ag_sio2_ag(
        d_ag_bottom_nm,
        d_sio2_nm,
        d_ag_top_nm,
    )
    return reflectance


def transmittance_spectrum_ag_sio2_ag(
    d_ag_bottom_nm: float,
    d_sio2_nm: float,
    d_ag_top_nm: float,
) -> np.ndarray:
    _, transmittance = fabry_perot_response_ag_sio2_ag(
        d_ag_bottom_nm,
        d_sio2_nm,
        d_ag_top_nm,
    )
    return transmittance


def _gaussian(center_nm: float, width_nm: float) -> np.ndarray:
    return np.exp(-0.5 * ((WAVELENGTHS_NM - center_nm) / width_nm) ** 2)


_X_BAR = _gaussian(600.0, 55.0) + 0.35 * _gaussian(450.0, 30.0)
_Y_BAR = _gaussian(550.0, 40.0)
_Z_BAR = 1.8 * _gaussian(445.0, 25.0)
_ILLUMINANT_D65 = (
    0.9 * _gaussian(460.0, 60.0)
    + 1.0 * _gaussian(560.0, 90.0)
    + 0.75 * _gaussian(610.0, 120.0)
)

_XYZ_NORMALIZER = float(np.sum(_ILLUMINANT_D65 * _Y_BAR))
_REF_WHITE = np.array(
    [
        float(np.sum(_ILLUMINANT_D65 * _X_BAR)) / _XYZ_NORMALIZER,
        1.0,
        float(np.sum(_ILLUMINANT_D65 * _Z_BAR)) / _XYZ_NORMALIZER,
    ],
    dtype=np.float64,
)


def spectrum_to_xyz(spectrum: np.ndarray) -> np.ndarray:
    weighted = spectrum * _ILLUMINANT_D65
    x = float(np.sum(weighted * _X_BAR)) / _XYZ_NORMALIZER
    y = float(np.sum(weighted * _Y_BAR)) / _XYZ_NORMALIZER
    z = float(np.sum(weighted * _Z_BAR)) / _XYZ_NORMALIZER
    return np.array([x, y, z], dtype=np.float64)


def _lab_f(value: float) -> float:
    delta = 6.0 / 29.0
    if value > delta**3:
        return value ** (1.0 / 3.0)
    return value / (3 * delta * delta) + 4.0 / 29.0


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    ratios = xyz / _REF_WHITE
    fx, fy, fz = (_lab_f(float(r)) for r in ratios)
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.array([l, a, b], dtype=np.float64)


def xyz_to_srgb_hex(xyz: np.ndarray) -> str:
    matrix = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ],
        dtype=np.float64,
    )
    rgb_linear = matrix @ xyz
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)

    def gamma_correct(channel: float) -> float:
        if channel <= 0.0031308:
            return 12.92 * channel
        return 1.055 * (channel ** (1.0 / 2.4)) - 0.055

    rgb = np.clip([gamma_correct(float(v)) for v in rgb_linear], 0.0, 1.0)
    rgb_int = [int(round(v * 255)) for v in rgb]
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
    l1, a1, b1 = (float(value) for value in lab_a)
    l2, a2, b2 = (float(value) for value in lab_b)

    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7))) if c_bar > 0 else 0.0

    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    c1_prime = np.hypot(a1_prime, b1)
    c2_prime = np.hypot(a2_prime, b2)

    def _hue_angle_degrees(a_prime: float, b_value: float) -> float:
        if a_prime == 0.0 and b_value == 0.0:
            return 0.0
        angle = np.degrees(np.arctan2(b_value, a_prime))
        return angle + 360.0 if angle < 0 else angle

    h1_prime = _hue_angle_degrees(a1_prime, b1)
    h2_prime = _hue_angle_degrees(a2_prime, b2)

    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime

    if c1_prime == 0.0 or c2_prime == 0.0:
        delta_h_prime = 0.0
    else:
        hue_difference = h2_prime - h1_prime
        if abs(hue_difference) <= 180.0:
            delta_h_prime = hue_difference
        elif hue_difference > 180.0:
            delta_h_prime = hue_difference - 360.0
        else:
            delta_h_prime = hue_difference + 360.0

    delta_h_term = 2.0 * np.sqrt(c1_prime * c2_prime) * np.sin(np.radians(delta_h_prime / 2.0))
    l_bar_prime = 0.5 * (l1 + l2)
    c_bar_prime = 0.5 * (c1_prime + c2_prime)

    if c1_prime == 0.0 or c2_prime == 0.0:
        h_bar_prime = h1_prime + h2_prime
    else:
        hue_sum = h1_prime + h2_prime
        if abs(h1_prime - h2_prime) <= 180.0:
            h_bar_prime = 0.5 * hue_sum
        elif hue_sum < 360.0:
            h_bar_prime = 0.5 * (hue_sum + 360.0)
        else:
            h_bar_prime = 0.5 * (hue_sum - 360.0)

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )
    delta_theta = 30.0 * np.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))
    c_bar_prime7 = c_bar_prime**7
    r_c = 2.0 * np.sqrt(c_bar_prime7 / (c_bar_prime7 + 25.0**7)) if c_bar_prime > 0 else 0.0
    s_l = 1.0 + (0.015 * ((l_bar_prime - 50.0) ** 2)) / np.sqrt(20.0 + ((l_bar_prime - 50.0) ** 2))
    s_c = 1.0 + 0.045 * c_bar_prime
    s_h = 1.0 + 0.015 * c_bar_prime * t
    r_t = -np.sin(np.radians(2.0 * delta_theta)) * r_c

    delta_e = np.sqrt(
        (delta_l_prime / s_l) ** 2
        + (delta_c_prime / s_c) ** 2
        + (delta_h_term / s_h) ** 2
        + r_t * (delta_c_prime / s_c) * (delta_h_term / s_h)
    )
    return float(delta_e)

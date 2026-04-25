from __future__ import annotations

import numpy as np

from backend.app.algorithms.optics import (
    fabry_perot_response_ag_sio2_ag,
    reflectance_spectrum_ag_sio2_ag,
    transmittance_spectrum_ag_sio2_ag,
)


STACK_NM = (30.0, 150.0, 30.0)


def test_zero_angle_matches_existing_normal_incidence_response() -> None:
    reflectance_default = reflectance_spectrum_ag_sio2_ag(*STACK_NM)
    transmittance_default = transmittance_spectrum_ag_sio2_ag(*STACK_NM)

    reflectance_te = reflectance_spectrum_ag_sio2_ag(*STACK_NM, theta_deg=0.0, polarization="te")
    reflectance_tm = reflectance_spectrum_ag_sio2_ag(*STACK_NM, theta_deg=0.0, polarization="tm")
    reflectance_unpolarized = reflectance_spectrum_ag_sio2_ag(
        *STACK_NM,
        theta_deg=0.0,
        polarization="unpolarized",
    )
    transmittance_te = transmittance_spectrum_ag_sio2_ag(*STACK_NM, theta_deg=0.0, polarization="te")
    transmittance_tm = transmittance_spectrum_ag_sio2_ag(*STACK_NM, theta_deg=0.0, polarization="tm")
    transmittance_unpolarized = transmittance_spectrum_ag_sio2_ag(
        *STACK_NM,
        theta_deg=0.0,
        polarization="unpolarized",
    )

    assert np.allclose(reflectance_default, reflectance_te)
    assert np.allclose(reflectance_default, reflectance_tm)
    assert np.allclose(reflectance_default, reflectance_unpolarized)
    assert np.allclose(transmittance_default, transmittance_te)
    assert np.allclose(transmittance_default, transmittance_tm)
    assert np.allclose(transmittance_default, transmittance_unpolarized)


def test_oblique_incidence_changes_response_and_unpolarized_averages_te_tm() -> None:
    normal_reflectance, normal_transmittance = fabry_perot_response_ag_sio2_ag(
        *STACK_NM,
        theta_deg=0.0,
        polarization="unpolarized",
    )
    oblique_reflectance, oblique_transmittance = fabry_perot_response_ag_sio2_ag(
        *STACK_NM,
        theta_deg=45.0,
        polarization="unpolarized",
    )
    te_reflectance, te_transmittance = fabry_perot_response_ag_sio2_ag(
        *STACK_NM,
        theta_deg=45.0,
        polarization="te",
    )
    tm_reflectance, tm_transmittance = fabry_perot_response_ag_sio2_ag(
        *STACK_NM,
        theta_deg=45.0,
        polarization="tm",
    )

    assert not np.allclose(oblique_reflectance, normal_reflectance)
    assert not np.allclose(oblique_transmittance, normal_transmittance)
    assert np.allclose(oblique_reflectance, 0.5 * (te_reflectance + tm_reflectance))
    assert np.allclose(oblique_transmittance, 0.5 * (te_transmittance + tm_transmittance))
    assert np.all((oblique_reflectance >= 0.0) & (oblique_reflectance <= 1.0))
    assert np.all((oblique_transmittance >= 0.0) & (oblique_transmittance <= 1.0))
    assert np.all(oblique_reflectance + oblique_transmittance <= 1.000001)

# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Focused tests for fast observer-frame synchrotron radiation products."""

import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, m_e

from fbpic.openpmd_diag import SynchrotronRadiationDiagnostic
from fbpic.openpmd_diag.observer_radiation_diag import (
    ObserverRadiationWriter, angular_cell_measure, source_moment_components,
)
from fbpic.particles.elementary_process.synchrotron.radiator import (
    SynchrotronRadiator,
)
from fbpic.utils.cuda import cuda_installed

if cuda_installed:
    import cupy


POWER_FACTOR = e**2 / (6.0 * np.pi * epsilon_0 * c)
ANGULAR_POWER_FACTOR = e**2 / (16.0 * np.pi**2 * epsilon_0 * c)
E_MC = e / (m_e * c)


def _species(u, electric, c_magnetic, dt=2.e-18, count=1, weight=1.0,
             positions=None):
    u = np.asarray(u, dtype=np.float64)
    electric = np.asarray(electric, dtype=np.float64)
    c_magnetic = np.asarray(c_magnetic, dtype=np.float64)
    gamma = math.sqrt(1.0 + np.dot(u, u))
    if positions is None:
        positions = np.zeros((count, 3))
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape == (3,):
        positions = np.repeat(positions[None, :], count, axis=0)

    def full(value):
        return np.full(count, value, dtype=np.float64)

    return SimpleNamespace(
        use_cuda=False, dt=dt, Ntot=count,
        x=positions[:, 0].copy(), y=positions[:, 1].copy(),
        z=positions[:, 2].copy(),
        ux=full(u[0]), uy=full(u[1]), uz=full(u[2]),
        Ex=full(electric[0]), Ey=full(electric[1]), Ez=full(electric[2]),
        Bx=full(c_magnetic[0] / c), By=full(c_magnetic[1] / c),
        Bz=full(c_magnetic[2] / c), w=full(weight),
        inv_gamma=full(1.0 / gamma),
    )


def _activate(species, boost=None, n_samples=256):
    species.synchrotron_radiator = SynchrotronRadiator(
        species, None, None, None, 1.0, False, 20.0, n_samples, boost)
    return species.synchrotron_radiator


def _power_oracle(u, electric, c_magnetic):
    u = np.asarray(u)
    gamma = math.sqrt(1.0 + np.dot(u, u))
    beta = u / gamma
    du = -E_MC * (electric + np.cross(beta, c_magnetic))
    dgamma = -E_MC * np.dot(beta, electric)
    beta2 = np.dot(beta, beta)
    cross2 = np.dot(np.cross(beta, du), np.cross(beta, du))
    perpendicular = POWER_FACTOR * gamma**2 * cross2 / beta2
    parallel = POWER_FACTOR * dgamma**2 / beta2
    omega_c = 1.5 * gamma**2 * math.sqrt(cross2) / beta2**1.5
    dot_beta = (du - beta * dgamma) / gamma
    return perpendicular, parallel, omega_c, beta, dot_beta


def _angular_power_oracle(direction, beta, dot_beta):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    numerator = np.linalg.norm(np.cross(
        direction, np.cross(direction - beta, dot_beta)))**2
    return (ANGULAR_POWER_FACTOR * numerator
            / (1.0 - np.dot(direction, beta))**5)


def _boost_lab_event_to_simulation(gamma_boost, u_lab, electric_lab,
                                   c_magnetic_lab):
    beta_boost = math.sqrt(1.0 - gamma_boost**-2)
    gamma_lab = math.sqrt(1.0 + np.dot(u_lab, u_lab))
    ux, uy, uz_lab = u_lab
    gamma_sim = gamma_boost * (gamma_lab - beta_boost * uz_lab)
    uz_sim = gamma_boost * (uz_lab - beta_boost * gamma_lab)
    ex, ey, ez = electric_lab
    cbx, cby, cbz = c_magnetic_lab
    electric_sim = np.array([
        gamma_boost * (ex - beta_boost * cby),
        gamma_boost * (ey + beta_boost * cbx),
        ez,
    ])
    c_magnetic_sim = np.array([
        gamma_boost * (cbx + beta_boost * ey),
        gamma_boost * (cby - beta_boost * ex),
        cbz,
    ])
    return np.array([ux, uy, uz_sim]), gamma_sim, electric_sim, c_magnetic_sim


def test_transverse_longitudinal_split_and_bin_independence(tmp_path):
    gamma = 80.0
    u = np.array([2.0, -1.0, math.sqrt(gamma**2 - 6.0)])
    electric = np.array([1.4e11, -0.7e11, 2.2e11])
    c_magnetic = np.array([-0.2e11, 0.8e11, 0.3e11])
    count, weight, dt = 19, 3.5, 4.e-18
    expected_perp, expected_parallel, _, _, _ = _power_oracle(
        u, electric, c_magnetic)

    accounting = []
    represented = []
    for label, energy_edges in (
            ("coarse", np.geomspace(1.e-20, 1.e-12, 7)),
            ("fine", np.geomspace(1.e-20, 1.e-12, 101))):
        species = _species(
            u, electric, c_magnetic, dt=dt, count=count, weight=weight)
        radiator = _activate(species)
        diagnostic = SynchrotronRadiationDiagnostic(
            period=1, species={"electrons": species},
            comm=SimpleNamespace(rank=0, size=1),
            write_dir=str(tmp_path / label),
            photon_energy_edges=energy_edges,
            angular_bin_edges=(np.linspace(-0.1, 0.1, 9),
                               np.linspace(-0.1, 0.1, 11)),
            samples_per_particle=2,
        )
        np.random.seed(17)
        radiator.handle_radiation(0.0)
        values = diagnostic.accumulators["electrons"].accounting
        accounting.append((
            float(values["transverse_energy"][0]),
            float(values["longitudinal_energy"][0]),
        ))
        represented.append(float(values[
            "represented/angular_spectral_energy"][0]))

    expected_scale = count * weight * dt
    assert accounting[0][0] == pytest.approx(
        expected_perp * expected_scale, rel=2.e-14)
    assert accounting[0][1] == pytest.approx(
        expected_parallel * expected_scale, rel=2.e-14)
    assert accounting[1] == pytest.approx(accounting[0], rel=2.e-15)
    assert represented[1] == pytest.approx(represented[0], rel=2.e-15)


def test_longitudinal_power_is_not_put_in_curvature_spectrum(tmp_path):
    gamma = 60.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    electric = np.array([0.0, 0.0, 8.e11])
    c_magnetic = np.zeros(3)
    dt, weight = 3.e-18, 4.0
    species = _species(u, electric, c_magnetic, dt=dt, weight=weight)
    radiator = _activate(species)
    direction = np.array([0.06, 0.0, 1.0])
    direction /= np.linalg.norm(direction)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        photon_energy_edges=np.geomspace(1.e-20, 1.e-12, 12),
        angular_bin_edges=(np.linspace(-0.2, 0.2, 9),
                           np.linspace(-0.2, 0.2, 9)),
        observer_time_edges=np.linspace(-1.e-14, 1.e-14, 9),
        detectors=[{"name": "off_axis", "direction": direction}],
    )
    radiator.handle_radiation(0.0)
    accumulator = diagnostic.accumulators["electrons"]
    perpendicular, parallel, _, beta, dot_beta = _power_oracle(
        u, electric, c_magnetic)
    assert perpendicular == pytest.approx(0.0, abs=0.0)
    assert accumulator.data["angular_spectral"].sum() == 0.0
    assert accumulator.accounting["transverse_energy"][0] == 0.0
    assert accumulator.accounting["longitudinal_energy"][0] == pytest.approx(
        weight * parallel * dt, rel=2.e-14)
    expected_directional = (
        weight * dt * _angular_power_oracle(direction, beta, dot_beta))
    assert accumulator.data[
        "detector/off_axis/direction"].sum() == pytest.approx(
            expected_directional, rel=2.e-14)


def test_energy_band_uses_exact_transverse_angular_power(tmp_path):
    gamma = 55.0
    u = np.array([1.2, -0.7, math.sqrt(gamma**2 - 1.0 - 1.2**2 - 0.7**2)])
    electric = np.array([0.7e11, -0.3e11, 1.1e11])
    c_magnetic = np.array([0.2e11, 0.9e11, -0.1e11])
    direction = np.array([0.025, -0.012, 1.0])
    direction /= np.linalg.norm(direction)
    dt, weight = 2.5e-18, 3.2
    species = _species(
        u, electric, c_magnetic, dt=dt, weight=weight,
        positions=np.array([0.0, 0.0, 0.0]))
    radiator = _activate(species)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        observer_time_edges=np.array([-1.e-14, 1.e-14]),
        detectors=[{
            "name": "selected", "direction": direction,
            "energy_bands": [{"name": "all", "energy_range": (0.0, 1.0)}],
        }],
    )
    accumulator = diagnostic.accumulators["electrons"]
    # A direction plus deterministic energy band needs no Monte-Carlo packet.
    assert not accumulator.needs_spectral_samples
    assert accumulator.angular_kernel_x is None

    radiator.handle_radiation(0.0)
    _, _, _, beta, dot_beta = _power_oracle(u, electric, c_magnetic)
    du = -E_MC * (electric + np.cross(beta, c_magnetic))
    dgamma = -E_MC * np.dot(beta, electric)
    du_perp = du - beta * dgamma / np.dot(beta, beta)
    dot_beta_perp = du_perp / gamma
    scale = weight * dt
    broadband = accumulator.data[
        "detector/selected/direction"].sum()
    band = accumulator.data[
        "detector/selected/band/all/direction"].sum()
    assert broadband == pytest.approx(
        scale * _angular_power_oracle(direction, beta, dot_beta), rel=2.e-14)
    assert band == pytest.approx(
        scale * _angular_power_oracle(direction, beta, dot_beta_perp),
        rel=2.e-14)


def test_complete_boosted_event_reconstruction_and_worldline_interval(tmp_path):
    gamma_boost = 5.0
    beta_boost = math.sqrt(1.0 - gamma_boost**-2)
    boost = SimpleNamespace(gamma0=gamma_boost, beta0=beta_boost)
    u_lab = np.array([4.0, -2.0, 70.0])
    electric_lab = np.array([1.1e11, -0.6e11, 0.4e11])
    c_magnetic_lab = np.array([-0.3e11, 0.8e11, 0.2e11])
    u_sim, gamma_sim, electric_sim, c_magnetic_sim = \
        _boost_lab_event_to_simulation(
            gamma_boost, u_lab, electric_lab, c_magnetic_lab)

    event_lab = np.array([2.3e-15, 0.7e-6, -0.4e-6, 12.e-6])
    time_sim = gamma_boost * (
        event_lab[0] - beta_boost * event_lab[3] / c)
    z_sim = gamma_boost * (
        event_lab[3] - beta_boost * c * event_lab[0])
    translation = np.array([c * 0.8e-15, 0.2e-6, -0.1e-6, 1.7e-6])
    positions = np.array([event_lab[1], event_lab[2], z_sim])
    species = _species(
        u_sim, electric_sim, c_magnetic_sim, positions=positions)
    species.inv_gamma[:] = 1.0 / gamma_sim
    radiator = _activate(species, boost)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species}, boost=boost,
        observer_translation=translation,
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        output_channels=["source_moments", "accounting"],
        source_moments=[{"name": "all"}],
    )
    accumulator = diagnostic.accumulators["electrons"]
    reconstructed = accumulator._observer_event(species, time_sim, np)
    gamma_lab = math.sqrt(1.0 + np.dot(u_lab, u_lab))
    assert reconstructed["gamma"][0] == pytest.approx(gamma_lab, rel=3.e-14)
    assert np.array([
        reconstructed["ux"][0], reconstructed["uy"][0],
        reconstructed["uz"][0]]) == pytest.approx(u_lab, rel=3.e-14)
    assert reconstructed["dt_ratio"][0] == pytest.approx(
        gamma_lab / gamma_sim, rel=3.e-14)
    expected_perp, expected_parallel, expected_omega, _, expected_dot_beta = \
        _power_oracle(u_lab, electric_lab, c_magnetic_lab)
    assert reconstructed["p_perp"][0] == pytest.approx(
        expected_perp, rel=4.e-14)
    assert reconstructed["p_parallel"][0] == pytest.approx(
        expected_parallel, rel=4.e-14)
    assert reconstructed["omega_c"][0] == pytest.approx(
        expected_omega, rel=4.e-14)
    assert np.array([
        reconstructed["dot_beta_x"][0], reconstructed["dot_beta_y"][0],
        reconstructed["dot_beta_z"][0]]) == pytest.approx(
            expected_dot_beta, rel=4.e-14)
    assert reconstructed["x"][0] == pytest.approx(
        event_lab[1] + translation[1], rel=0.0, abs=2.e-20)
    assert reconstructed["y"][0] == pytest.approx(
        event_lab[2] + translation[2], rel=0.0, abs=2.e-20)
    assert reconstructed["z"][0] == pytest.approx(
        event_lab[3] + translation[3], rel=3.e-14)
    assert reconstructed["time"][0] == pytest.approx(
        event_lab[0] + translation[0] / c, rel=3.e-14)

    np.random.seed(31)
    radiator.handle_radiation(time_sim)
    components = source_moment_components(accumulator.moment_stats["all"])
    assert components["centroid_x"][0] == pytest.approx(
        event_lab[1] + translation[1], rel=2.e-14)
    assert components["centroid_y"][0] == pytest.approx(
        event_lab[2] + translation[2], rel=2.e-14)
    assert components["centroid_z"][0] == pytest.approx(
        event_lab[3] + translation[3], rel=2.e-14)


def test_default_angles_use_acceleration_plane_and_photon_energy(tmp_path):
    count = 40000
    gamma = 120.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    species = _species(
        u, np.zeros(3), np.array([0.0, 3.e11, 0.0]), count=count)
    _activate(species, n_samples=64)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        observer_frame="laboratory",
        photon_energy_edges=np.array([0.0, 1.0]),
        angular_bin_edges=(np.linspace(-0.5, 0.5, 3),
                           np.linspace(-0.5, 0.5, 3)),
        output_channels=["angular_spectral"],
    )
    accumulator = diagnostic.accumulators["electrons"]
    event = accumulator._observer_event(species, 0.0, np)

    np.random.seed(101)
    low = accumulator._sample_direction(
        event, np.full(count, 0.02), np)
    np.random.seed(101)
    high = accumulator._sample_direction(
        event, np.full(count, 5.0), np)
    low_angles = np.array([
        np.arctan2(low[0], low[2]), np.arctan2(low[1], low[2])])
    high_angles = np.array([
        np.arctan2(high[0], high[2]), np.arctan2(high[1], high[2])])
    low_rms = low_angles.std(axis=1)
    high_rms = high_angles.std(axis=1)
    assert high_rms[1] < 0.45 * low_rms[1]
    # B_y curves the orbit in x-z.  The local velocity supplies the tangent
    # in that plane, while the Schwinger conditional samples normal to it.
    assert high_rms[0] < 1.e-12 * high_rms[1]
    assert high_rms[1] > 0.0

    accumulator.local_angular_model = "legacy_gaussian"
    np.random.seed(207)
    legacy_low = accumulator._sample_direction(
        event, np.full(count, 0.02), np)
    np.random.seed(207)
    legacy_high = accumulator._sample_direction(
        event, np.full(count, 5.0), np)
    for low_component, high_component in zip(legacy_low, legacy_high):
        assert np.array_equal(low_component, high_component)


def test_source_moment_derived_geometry_and_correlations():
    angle = 0.37
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    base = np.array([
        [-3.0, -1.0], [-3.0, 1.0], [3.0, -1.0], [3.0, 1.0],
    ]) * 1.e-6
    transverse = base @ rotation.T
    positions = np.column_stack((
        transverse, np.array([-2.0, -1.0, 1.0, 2.0]) * 1.e-6))
    theta = np.column_stack((
        2.e3 * positions[:, 0], -1.e3 * positions[:, 1]))
    observer_time = 4.e-9 * positions[:, 2] / 1.e-6
    weights = np.ones(4)

    stats = np.zeros(30)
    stats[0] = weights.sum()
    stats[1:4] = np.sum(weights[:, None] * positions, axis=0)
    stats[4:13] = np.einsum(
        "n,ni,nj->ij", weights, positions, positions).ravel()
    stats[13:15] = np.sum(weights[:, None] * theta, axis=0)
    stats[15:19] = np.einsum(
        "n,ni,nj->ij", weights, theta, theta).ravel()
    stats[19:25] = np.einsum(
        "n,ni,nj->ij", weights, positions, theta).ravel()
    stats[25] = np.dot(weights, observer_time)
    stats[26] = np.dot(weights, observer_time**2)
    stats[27:30] = np.sum(
        weights[:, None] * positions * observer_time[:, None], axis=0)

    components = source_moment_components(stats)
    assert components["transverse_major_rms"][0] \
        > components["transverse_minor_rms"][0]
    orientation = components["transverse_orientation"][0]
    # Principal-axis sign is arbitrary, hence compare modulo pi.
    difference = (orientation - angle + 0.5 * math.pi) % math.pi \
        - 0.5 * math.pi
    assert abs(difference) < 1.e-12
    assert components["correlation_x_theta_x"][0] == pytest.approx(
        1.0, abs=2.e-15)
    assert components["correlation_z_observer_time"][0] == pytest.approx(
        1.0, abs=2.e-15)
    assert components["longitudinal_emission_extent"][0] \
        == pytest.approx(components["rms_z"][0], rel=0.0)


def test_openpmd_products_metadata_normalization_and_intervals(tmp_path):
    gamma = 90.0
    u = np.array([1.0, 0.0, math.sqrt(gamma**2 - 2.0)])
    count = 64
    positions = np.column_stack((
        np.linspace(-1.e-6, 1.e-6, count),
        np.linspace(0.5e-6, -0.5e-6, count),
        np.linspace(-2.e-6, 2.e-6, count),
    ))
    species = _species(
        u, np.array([0.2e11, 0.0, 0.1e11]),
        np.array([0.0, 2.e11, 0.0]), count=count, positions=positions)
    radiator = _activate(species)
    energy_edges = np.geomspace(1.e-20, 2.e-12, 13)
    theta_x_edges = np.linspace(-0.08, 0.08, 11)
    theta_y_edges = np.linspace(-0.07, 0.07, 9)
    time_edges = np.linspace(-2.e-14, 2.e-14, 17)
    x_edges = np.linspace(-2.e-6, 2.e-6, 9)
    z_edges = np.linspace(-3.e-6, 3.e-6, 11)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        photon_energy_bin_edges=energy_edges,
        angular_bin_edges={
            "theta_x": theta_x_edges, "theta_y": theta_y_edges},
        angular_measure="solid_angle", observer_time_bin_edges=time_edges,
        detectors=[{
            "name": "axis", "direction": (0.0, 0.0, 1.0),
            "half_angle": 0.025,
            "energy_bands": [{
                "name": "band", "energy_range": (1.e-18, 1.e-13)}],
        }],
        source_coordinate_bin_edges={"x": x_edges, "z": z_edges},
        source_distribution_projections=[
            {"name": "xz", "axes": ("x", "z")},
            {"name": "x_energy", "axes": ("x", "energy")},
        ],
        source_moment_selections=[{
            "name": "selected", "energy_range": (1.e-18, 1.e-13),
            "angular_range": ((-0.08, 0.08), (-0.07, 0.07)),
        }],
        output_mode="both", samples_per_particle=2, particle_batch_size=13,
    )
    accumulator = diagnostic.accumulators["electrons"]
    np.random.seed(411)
    radiator.handle_radiation(0.0)
    first_raw = accumulator.snapshot()
    diagnostic.write_hdf5(1)
    np.random.seed(733)
    radiator.handle_radiation(species.dt)
    diagnostic.write_hdf5(2)

    first_path = Path(tmp_path) / "hdf5" / "data00000001.h5"
    second_path = Path(tmp_path) / "hdf5" / "data00000002.h5"
    with h5py.File(first_path, "r") as output:
        iteration = output["/data/1"]
        fields = iteration["fields"]
        spectrum = fields["radiationAngularSpectrum_electrons_cumulative"]
        assert spectrum.shape == (10, 8, 12)
        assert spectrum.attrs["observerFrame"] == np.bytes_("laboratory")
        assert spectrum.attrs["angularMeasure"] == np.bytes_("solid_angle")
        assert spectrum.attrs["localAngularModel"] == np.bytes_("synchrotron")
        assert spectrum.attrs["spectralAngularKernel"] == np.bytes_(
            "polarization_summed_Schwinger_vertical_conditional")
        assert spectrum.attrs["longitudinalAccelerationIncluded"] == 0
        assert np.array_equal(
            spectrum.attrs["unitDimension"], np.zeros(7))
        assert np.array_equal(spectrum.attrs["binEdges_energy"], energy_edges)
        assert list(spectrum.attrs["axisLabels"]) == [
            b"theta_x", b"theta_y", b"energy"]
        measure = angular_cell_measure(
            theta_x_edges, theta_y_edges, "solid_angle")
        integrated = np.sum(
            spectrum[:] * measure[:, :, None] * np.diff(energy_edges)[None, None, :])
        assert integrated == pytest.approx(
            first_raw["data"]["angular_spectral"].sum(), rel=2.e-15)

        source = fields["radiationSource_xz_electrons_cumulative"]
        source_integral = np.sum(
            source[:] * np.diff(x_edges)[:, None] * np.diff(z_edges)[None, :])
        assert source_integral == pytest.approx(
            first_raw["data"]["source/xz"].sum(), rel=2.e-15)
        assert source.attrs["observerFrame"] == np.bytes_("laboratory")
        assert np.array_equal(
            source.attrs["unitDimension"], [0.0, 1.0, -2.0, 0, 0, 0, 0])
        edge_path = (
            "/data/1/radiationAxes/"
            "radiationSource_xz_electrons_cumulative/xEdges")
        assert np.array_equal(output[edge_path][:], x_edges)
        accounting = fields[
            "radiationAccounting_electrons_cumulative_total_radiated_energy"]
        total = accounting[0]
        expected_total = (
            first_raw["accounting"]["transverse_energy"][0]
            + first_raw["accounting"]["longitudinal_energy"][0])
        assert total == pytest.approx(expected_total, rel=2.e-15)
        assert np.array_equal(
            accounting.attrs["unitDimension"], [2.0, 1.0, -2.0, 0, 0, 0, 0])
        moments = fields[
            "radiationSourceMoments_selected_electrons_cumulative_energy"]
        assert b"energy_range" in moments.attrs["sourceMomentSelection"]
        assert np.array_equal(
            moments.attrs["unitDimension"], [2.0, 1.0, -2.0, 0, 0, 0, 0])
        pulse = fields[
            "radiationPulseMetrics_axis_aperture_electrons_cumulative_energy"]
        assert pulse[0] > 0.0
        assert pulse.attrs["cumulativeEnergyInterval"] == pytest.approx(
            [0.05, 0.95])
        assert iteration.attrs["timeReferenceFrame"] == np.bytes_("laboratory")
        assert iteration.attrs["observerLorentzTransform"].shape == (16,)
        assert np.array_equal(
            iteration.attrs["observerLorentzTransformShape"], [4, 4])
        band_profile = fields[
            "radiationObserverTime_axis_band_band_direction_"
            "electrons_cumulative"]
        assert np.array_equal(
            band_profile.attrs["unitDimension"],
            [2.0, 1.0, -3.0, 0, 0, 0, 0])
        assert band_profile.attrs["bandAngularModel"] == np.bytes_(
            "exact_transverse_Lienard_pattern")

    with h5py.File(second_path, "r") as output:
        fields = output["/data/2/fields"]
        cumulative = fields[
            "radiationAccounting_electrons_cumulative_transverse_energy"]
        interval = fields[
            "radiationAccounting_electrons_interval_transverse_energy"]
        assert cumulative[0] == pytest.approx(
            2.0 * interval[0], rel=2.e-15)
        assert cumulative.attrs["cumulative"] == 1
        assert interval.attrs["cumulative"] == 0

    openpmd_viewer = pytest.importorskip("openpmd_viewer")
    time_series = openpmd_viewer.OpenPMDTimeSeries(
        str(Path(tmp_path) / "hdf5"), check_all_files=True)
    assert np.array_equal(time_series.iterations, [1, 2])
    for field_name, shape in (
            ("radiationAngularSpectrum_electrons_cumulative", (10, 8, 12)),
            ("radiationAccounting_electrons_cumulative_"
             "total_radiated_energy", (1,)),
            ("radiationSourceMoments_selected_electrons_cumulative_"
             "centroid_x", (1,)),
            ("radiationObserverTime_axis_band_band_direction_"
             "electrons_cumulative", (16,))):
        values, _ = time_series.get_field(field=field_name, iteration=1)
        assert values.shape == shape


def test_disabled_channels_allocate_only_accounting(tmp_path):
    gamma = 30.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    species = _species(
        u, np.zeros(3), np.array([0.0, 1.e11, 0.0]), count=10)
    # Prove that an accounting-only channel does not even require positions.
    del species.x, species.y, species.z
    radiator = _activate(species, n_samples=32)
    assert radiator.radiation_data is None
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        observer_frame="laboratory", output_channels=["accounting"],
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert accumulator.data == {}
    assert accumulator.moment_stats == {}
    assert not accumulator.needs_positions
    assert not accumulator.needs_spectral_samples
    radiator.handle_radiation(0.0)
    assert accumulator.accounting["transverse_energy"][0] > 0.0


def test_reduced_source_projection_does_not_allocate_angular_spectrum(tmp_path):
    gamma = 35.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    positions = np.array([[-0.5e-6, 0.0, 0.0], [0.5e-6, 0.0, 0.0]])
    species = _species(
        u, np.zeros(3), np.array([0.0, 1.e11, 0.0]), count=2,
        positions=positions)
    radiator = _activate(species, n_samples=64)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        photon_energy_edges=np.array([0.0, 1.0]),
        source_coordinate_edges={"x": np.linspace(-1.e-6, 1.e-6, 5)},
        source_projections=[{"name": "x_energy", "axes": ("x", "energy")}],
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert set(accumulator.data) == {"source/x_energy"}
    radiator.handle_radiation(0.0)
    assert accumulator.data["source/x_energy"].sum() == pytest.approx(
        accumulator.accounting["transverse_energy"][0], rel=2.e-14)


def test_coordinate_only_source_projection_needs_no_photon_packets(tmp_path):
    gamma = 35.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    positions = np.array([
        [-0.5e-6, 0.0, -0.4e-6], [0.5e-6, 0.0, 0.4e-6]])
    species = _species(
        u, np.zeros(3), np.array([0.0, 1.e11, 0.0]), count=2,
        positions=positions)
    radiator = _activate(species, n_samples=64)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        source_coordinate_edges={
            "x": np.linspace(-1.e-6, 1.e-6, 5),
            "z": np.linspace(-1.e-6, 1.e-6, 5),
        },
        source_projections=[{"name": "xz", "axes": ("x", "z")}],
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert not accumulator.needs_spectral_samples
    assert accumulator.angular_kernel_x is None
    radiator.handle_radiation(0.0)
    assert accumulator.data["source/xz"].sum() == pytest.approx(
        accumulator.accounting["transverse_energy"][0], rel=2.e-14)


def test_full_source_product_and_expanded_moment_selections(tmp_path):
    count, gamma = 32, 40.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    positions = np.column_stack((
        np.linspace(-0.5e-6, 0.5e-6, count),
        np.linspace(0.4e-6, -0.4e-6, count),
        np.linspace(-0.8e-6, 0.8e-6, count),
    ))
    species = _species(
        u, np.zeros(3), np.array([0.0, 1.e11, 0.0]), count=count,
        positions=positions)
    radiator = _activate(species, n_samples=64)
    three_edges = np.array([-math.pi, 0.0, math.pi])
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1), write_dir=str(tmp_path),
        photon_energy_edges=np.array([0.0, 1.e-17, 1.0]),
        angular_bin_edges=(three_edges, three_edges),
        observer_time_edges=np.array([-1.e-8, 0.0, 1.e-8]),
        source_coordinate_edges={
            "x": np.array([-1.e-6, 0.0, 1.e-6]),
            "y": np.array([-1.e-6, 0.0, 1.e-6]),
            "z": np.array([-1.e-6, 0.0, 1.e-6]),
        },
        source_projections="full",
        source_moments=[
            {"name": "energy", "energy_bins": [0.0, 1.e-17, 1.0]},
            {"name": "angle", "theta_x_bins": three_edges,
             "theta_y_bins": three_edges},
        ],
        output_channels=["source", "source_moments"],
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert accumulator.data["source/full"].shape == (2,) * 7
    np.random.seed(1203)
    radiator.handle_radiation(0.0)
    transverse = accumulator.accounting["transverse_energy"][0]
    assert accumulator.data["source/full"].sum() == pytest.approx(
        transverse, rel=2.e-14)
    energy_partition = sum(
        stats[0] for name, stats in accumulator.moment_stats.items()
        if name.startswith("energy_energy_"))
    angle_partition = sum(
        stats[0] for name, stats in accumulator.moment_stats.items()
        if name.startswith("angle_angle_"))
    assert energy_partition == pytest.approx(transverse, rel=2.e-14)
    assert angle_partition == pytest.approx(transverse, rel=2.e-14)


@pytest.mark.skipif(not cuda_installed, reason="CUDA is unavailable")
def test_cuda_observer_products_and_accounting_parity(tmp_path):
    count, gamma = 257, 75.0
    u = np.array([1.5, -0.7, math.sqrt(gamma**2 - 1.0 - 1.5**2 - 0.7**2)])
    positions = np.column_stack((
        np.linspace(-1.e-6, 1.e-6, count),
        np.linspace(0.2e-6, -0.2e-6, count),
        np.linspace(-1.5e-6, 1.5e-6, count),
    ))

    def make(use_cuda):
        species = _species(
            u, np.array([0.4e11, -0.2e11, 0.1e11]),
            np.array([0.1e11, 1.1e11, -0.3e11]),
            count=count, weight=1.7, positions=positions)
        species.use_cuda = use_cuda
        if use_cuda:
            for name in (
                    "x", "y", "z", "ux", "uy", "uz", "Ex", "Ey", "Ez",
                    "Bx", "By", "Bz", "w", "inv_gamma"):
                setattr(species, name, cupy.asarray(getattr(species, name)))
        radiator = _activate(species, n_samples=128)
        diagnostic = SynchrotronRadiationDiagnostic(
            period=1, species={"electrons": species},
            comm=SimpleNamespace(rank=0, size=1),
            write_dir=str(tmp_path / ("gpu" if use_cuda else "cpu")),
            photon_energy_edges=np.array([0.0, 1.0]),
            angular_bin_edges=(np.linspace(-math.pi, math.pi, 7),
                               np.linspace(-math.pi, math.pi, 7)),
            observer_time_edges=np.array([-1.e-10, 1.e-10]),
            detectors=[{
                "name": "axis", "direction": (0.0, 0.0, 1.0),
                "half_angle": 0.03, "aperture_quadrature": 5,
            }],
            source_coordinate_edges={
                "x": np.linspace(-2.e-6, 2.e-6, 9)},
            source_projections=[("x", "energy")],
            source_moments=[{"name": "all"}],
            particle_batch_size=31,
        )
        return radiator, diagnostic.accumulators["electrons"]

    cpu_radiator, cpu = make(False)
    gpu_radiator, gpu = make(True)
    np.random.seed(619)
    cpu_radiator.handle_radiation(0.0)
    cupy.random.seed(619)
    gpu_radiator.handle_radiation(0.0)
    cupy.cuda.runtime.deviceSynchronize()
    gpu_radiator.receive_from_gpu()

    for key in ("transverse_energy", "longitudinal_energy"):
        assert gpu.accounting[key][0] == pytest.approx(
            cpu.accounting[key][0], rel=2.e-13)
    for key in (
            "detector/axis/direction", "detector/axis/aperture"):
        assert gpu.data[key] == pytest.approx(cpu.data[key], rel=2.e-12)
    assert gpu.data["angular_spectral"].sum() == pytest.approx(
        cpu.accounting["transverse_energy"][0], rel=2.e-13)
    assert cpu.data["angular_spectral"].sum() == pytest.approx(
        cpu.accounting["transverse_energy"][0], rel=2.e-13)
    assert gpu.data["source/x_energy"].sum() == pytest.approx(
        cpu.data["source/x_energy"].sum(), rel=2.e-13)


def test_advanced_snapshot_mpi_reduces_every_additive_record(monkeypatch):
    from fbpic.openpmd_diag import observer_radiation_diag

    diagnostic = SimpleNamespace(
        comm=SimpleNamespace(size=2), rank=0)
    writer = ObserverRadiationWriter(diagnostic, "cumulative")
    snapshot = {
        "data": {"angular_spectral": np.arange(6.0).reshape(2, 3)},
        "accounting": {"transverse_energy": np.array([7.0])},
        "moments": {"all": np.arange(30.0)},
    }

    def reduce(send, receive, root):
        assert root == 0
        receive[:] = 2.0 * send

    monkeypatch.setattr(
        observer_radiation_diag.comm_simple, "Reduce", reduce, raising=False)
    reduced = writer._reduce_snapshot(snapshot)
    for category in snapshot:
        for key in snapshot[category]:
            assert np.array_equal(
                reduced[category][key], 2.0 * snapshot[category][key])


def _run_observer_mpi_worker(write_dir):
    """Reduce advanced deterministic products from a partitioned population."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    global_count = 64
    lower = global_count * world.rank // world.size
    upper = global_count * (world.rank + 1) // world.size
    particle_index = np.arange(lower, upper, dtype=np.float64)
    local_count = upper - lower
    gamma = 45.0
    uz = math.sqrt(gamma**2 - 1.0)
    positions = np.column_stack((
        -0.9e-6 + 1.8e-6 * particle_index / (global_count - 1),
        np.zeros(local_count), np.zeros(local_count),
    ))
    species = _species(
        (0.0, 0.0, uz), np.zeros(3), (0.0, 1.0e11, 0.0),
        dt=2.e-18, count=local_count, positions=positions)
    species.w[:] = 0.5 + particle_index / global_count
    species.By[:] *= 1.0 + 0.2 * np.sin(0.17 * particle_index)
    radiator = _activate(species, n_samples=64)
    comm = SimpleNamespace(rank=world.rank, size=world.size)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species}, comm=comm,
        write_dir=write_dir,
        observer_time_edges=np.array([-1.e-12, 1.e-12]),
        detectors=[{"name": "axis", "direction": (0.0, 0.0, 1.0)}],
        source_coordinate_edges={"x": np.linspace(-1.e-6, 1.e-6, 9)},
        source_projections=[{"name": "x", "axes": ("x",)}],
        source_moments="all",
    )
    np.random.seed(900 + world.rank)
    radiator.handle_radiation(0.0)
    diagnostic.write_hdf5(1)
    world.Barrier()


def test_observer_mpi_reduction_and_openpmd_output(tmp_path):
    """Advanced products agree between one-rank and two-rank partitions."""
    pytest.importorskip("mpi4py")
    mpi_exec = shutil.which("mpiexec")
    if mpi_exec is None:
        adjacent_exec = Path(sys.executable).with_name("mpiexec")
        if adjacent_exec.exists():
            mpi_exec = str(adjacent_exec)
        else:
            pytest.skip("mpiexec is unavailable")

    test_file = Path(__file__).resolve()
    repo_root = str(test_file.parents[1])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        repo_root + os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else repo_root)
    output_dirs = {}
    for rank_count in (1, 2):
        output_dir = tmp_path / ("ranks-%d" % rank_count)
        output_dirs[rank_count] = output_dir
        result = subprocess.run(
            [mpi_exec, "-n", str(rank_count), sys.executable,
             str(test_file), "--mpi-observer-worker", str(output_dir)],
            env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, (
            "%d-rank observer worker failed:\nstdout:\n%s\nstderr:\n%s"
            % (rank_count, result.stdout, result.stderr))

    names = (
        "radiationAccounting_electrons_cumulative_total_radiated_energy",
        "radiationObserverTime_axis_direction_electrons_cumulative",
        "radiationSource_x_electrons_cumulative",
        "radiationSourceMoments_all_electrons_cumulative_energy",
        "radiationPulseMetrics_axis_direction_electrons_cumulative_"
        "energy_per_solid_angle",
    )
    outputs = {}
    for rank_count, output_dir in output_dirs.items():
        with h5py.File(
                output_dir / "hdf5" / "data00000001.h5", "r") as output:
            fields = output["/data/1/fields"]
            outputs[rank_count] = {
                name: fields[name][:] for name in names}
    for name in names:
        assert outputs[2][name] == pytest.approx(
            outputs[1][name], rel=2.e-13, abs=0.0)


if __name__ == "__main__" and len(sys.argv) == 3:
    if sys.argv[1] == "--mpi-observer-worker":
        _run_observer_mpi_worker(sys.argv[2])

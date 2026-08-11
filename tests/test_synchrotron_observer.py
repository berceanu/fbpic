# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Focused contract tests for observer-frame synchrotron radiation."""

import math
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, m_e

from fbpic.openpmd_diag import SynchrotronRadiationDiagnostic
from fbpic.openpmd_diag.observer_radiation_diag import (
    angular_cell_measure,
    source_moment_components,
)
from fbpic.particles.elementary_process.synchrotron.observer import (
    ObserverFrameRadiationAccumulator,
)
from fbpic.particles.elementary_process.synchrotron.radiator import (
    SynchrotronRadiator,
    _cached_spectral_cdf,
)


POWER_FACTOR = e**2 / (6.0 * np.pi * epsilon_0 * c)
ANGULAR_POWER_FACTOR = e**2 / (16.0 * np.pi**2 * epsilon_0 * c)
E_MC = e / (m_e * c)


def _species(u, electric, c_magnetic, dt=2.0e-18, count=1, weight=1.0,
             positions=None, charge=-e, mass=m_e):
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
        q=charge, m=mass, use_cuda=False, dt=dt, Ntot=count,
        x=positions[:, 0].copy(), y=positions[:, 1].copy(),
        z=positions[:, 2].copy(),
        ux=full(u[0]), uy=full(u[1]), uz=full(u[2]),
        Ex=full(electric[0]), Ey=full(electric[1]), Ez=full(electric[2]),
        Bx=full(c_magnetic[0] / c), By=full(c_magnetic[1] / c),
        Bz=full(c_magnetic[2] / c), w=full(weight),
        inv_gamma=full(1.0 / gamma), synchrotron_radiator=None,
    )


def _activate(species, boost=None, gamma_cutoff=2.0, x_max=8.0):
    radiator = SynchrotronRadiator(
        species, gamma_cutoff=gamma_cutoff, x_max=x_max,
        n_samples=64, boost=boost,
    )
    species.synchrotron_radiator = radiator
    return radiator


def _diagnostic(tmp_path, species, **kwargs):
    return SynchrotronRadiationDiagnostic(
        period=1,
        species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1),
        write_dir=str(tmp_path),
        **kwargs,
    )


def _power_oracle(u, electric, c_magnetic):
    u = np.asarray(u, dtype=np.float64)
    gamma = math.sqrt(1.0 + np.dot(u, u))
    beta = u / gamma
    du = -E_MC * (electric + np.cross(beta, c_magnetic))
    dgamma = -E_MC * np.dot(beta, electric)
    beta2 = np.dot(beta, beta)
    cross2 = np.dot(np.cross(beta, du), np.cross(beta, du))
    perpendicular = POWER_FACTOR * gamma**2 * cross2 / beta2
    parallel = POWER_FACTOR * dgamma**2 / beta2
    dot_beta = (du - beta * dgamma) / gamma
    parallel_coefficient = dgamma / beta2
    dot_beta_perp = (du - beta * parallel_coefficient) / gamma
    omega_c = 1.5 * gamma**2 * math.sqrt(cross2) / beta2**1.5
    return perpendicular, parallel, omega_c, beta, dot_beta, dot_beta_perp


def _angular_power_oracle(direction, beta, dot_beta):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    numerator = np.linalg.norm(np.cross(
        direction, np.cross(direction - beta, dot_beta)))**2
    return (
        ANGULAR_POWER_FACTOR * numerator
        / (1.0 - np.dot(direction, beta))**5
    )


def _boost_lab_event(gamma_boost, u_lab, electric_lab, c_magnetic_lab):
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
    return (
        np.array([ux, uy, uz_sim]), gamma_sim,
        electric_sim, c_magnetic_sim, beta_boost,
    )


def _attribute_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def test_activation_validates_species_cutoff_and_spectral_tail():
    u = np.array([0.0, 0.0, 20.0])
    fields = np.zeros(3)
    with pytest.raises(ValueError, match="electrons and positrons"):
        _activate(_species(
            u, fields, fields, charge=2.0 * e, mass=m_e))
    with pytest.raises(ValueError, match="electrons and positrons"):
        _activate(_species(
            u, fields, fields, charge=-e, mass=2.0 * m_e))
    with pytest.raises(ValueError, match="greater than one"):
        _activate(_species(u, fields, fields), gamma_cutoff=1.0)

    positrons = _species(u, fields, fields, charge=e)
    radiator = _activate(positrons, x_max=2.0)
    assert 0.0 < radiator.spectral_truncated_fraction < 1.0
    assert radiator.spectral_cdf[-1] == pytest.approx(
        1.0 - radiator.spectral_truncated_fraction, rel=2.0e-15)

    cached = _cached_spectral_cdf(2.0, 64)
    repeated = _cached_spectral_cdf(2.0, 64)
    assert cached[0] is repeated[0]
    assert cached[0][0] == 0.0
    assert cached[0][1] < 1.0e-5
    assert np.allclose(
        np.diff(np.log(cached[0][1:])),
        np.diff(np.log(cached[0][1:]))[0],
    )


def test_projected_angle_chart_is_validated_at_every_entry_point(tmp_path):
    gamma = 20.0
    species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [1.0e10, 0.0, 0.0], [0.0, 0.0, 0.0],
    )
    radiator = _activate(species)
    with pytest.raises(ValueError, match="forward hemisphere"):
        _diagnostic(
            tmp_path / "edges", species,
            photon_energy_edges=np.array([0.0, 1.0e-12]),
            theta_x_edges=np.array([-math.pi, math.pi]),
            theta_y_edges=np.array([-0.1, 0.1]),
        )
    assert radiator.observer_accumulator is None

    with pytest.raises(ValueError, match="inside"):
        _diagnostic(
            tmp_path / "detector", species,
            observer_time_edges=np.array([-1.0e-12, 1.0e-12]),
            detectors={"theta_x": math.pi, "theta_y": 0.0},
        )
    with pytest.raises(ValueError, match="inside"):
        angular_cell_measure(
            np.array([-math.pi, math.pi]),
            np.array([-0.1, 0.1]),
            "solid_angle",
        )


def test_boosted_event_power_position_and_worldline_interval(tmp_path):
    gamma_boost = 4.0
    u_lab = np.array([1.5, -0.8, 70.0])
    electric_lab = np.array([0.8e11, -0.4e11, 1.2e11])
    c_magnetic_lab = np.array([0.2e11, 0.7e11, -0.1e11])
    u_sim, gamma_sim, electric_sim, c_magnetic_sim, beta_boost = (
        _boost_lab_event(
            gamma_boost, u_lab, electric_lab, c_magnetic_lab)
    )
    ct_lab, x_lab, y_lab, z_lab = 8.0e-6, 1.1e-6, -0.7e-6, 2.5e-6
    ct_sim = gamma_boost * (ct_lab - beta_boost * z_lab)
    z_sim = gamma_boost * (z_lab - beta_boost * ct_lab)
    translation = np.array([0.3e-6, 0.2e-6, -0.1e-6, 0.4e-6])
    species = _species(
        u_sim, electric_sim, c_magnetic_sim, weight=2.5,
        positions=np.array([x_lab, y_lab, z_sim]),
    )
    species.inv_gamma[:] = 1.0 / gamma_sim
    boost = SimpleNamespace(gamma0=gamma_boost, beta0=beta_boost)
    radiator = _activate(species, boost=boost)
    diagnostic = _diagnostic(
        tmp_path, species,
        observer_translation=translation,
        source_coordinate_edges={
            "x": np.array([-2.0e-6, 2.0e-6]),
        },
        source_projections=[{"name": "x", "axes": ("x",)}],
    )
    accumulator = diagnostic.accumulators["electrons"]
    event = accumulator._observer_event(
        species, ct_sim / c, np, slice(None))
    perpendicular, parallel, omega_c, _, _, _ = _power_oracle(
        u_lab, electric_lab, c_magnetic_lab)
    gamma_lab = math.sqrt(1.0 + np.dot(u_lab, u_lab))

    assert event["gamma"][0] == pytest.approx(gamma_lab, rel=3.0e-14)
    assert np.array([
        event["ux"][0], event["uy"][0], event["uz"][0],
    ]) == pytest.approx(u_lab, rel=3.0e-14)
    assert event["dt_ratio"][0] == pytest.approx(
        gamma_lab / gamma_sim, rel=3.0e-14)
    assert event["p_perp"][0] == pytest.approx(perpendicular, rel=4.0e-14)
    assert event["p_parallel"][0] == pytest.approx(parallel, rel=4.0e-14)
    assert event["omega_c"][0] == pytest.approx(omega_c, rel=4.0e-14)
    assert event["x"][0] == pytest.approx(x_lab + translation[1])
    assert event["y"][0] == pytest.approx(y_lab + translation[2])
    assert event["z"][0] == pytest.approx(z_lab + translation[3])
    assert event["time"][0] == pytest.approx(
        (ct_lab + translation[0]) / c)

    radiator.handle_radiation(ct_sim / c)
    scale = species.w[0] * species.dt * gamma_lab / gamma_sim
    assert accumulator.accounting["transverse_energy"][0] == pytest.approx(
        perpendicular * scale, rel=4.0e-14)
    assert accumulator.accounting["longitudinal_energy"][0] == pytest.approx(
        parallel * scale, rel=4.0e-14)


def test_ultrarelativistic_on_axis_angular_power_is_stable():
    gamma = 1.0e12
    acceleration = 2.0e10
    event = {
        "beta_x": np.array([0.0]),
        "beta_y": np.array([0.0]),
        "beta_z": np.array([1.0]),
        "beta_abs": np.array([1.0]),
        "inv_gamma": np.array([1.0 / gamma]),
        "dot_beta_x": np.array([acceleration]),
        "dot_beta_y": np.array([0.0]),
        "dot_beta_z": np.array([0.0]),
        "dot_beta_perp_x": np.array([acceleration]),
        "dot_beta_perp_y": np.array([0.0]),
        "dot_beta_perp_z": np.array([0.0]),
    }
    value = ObserverFrameRadiationAccumulator._angular_power(
        np.array([0.0, 0.0, 1.0]), event, np)[0]
    one_minus_beta = 0.5 / gamma**2
    expected = (
        ANGULAR_POWER_FACTOR * acceleration**2 / one_minus_beta**3
    )
    assert np.isfinite(value)
    assert value == pytest.approx(expected, rel=3.0e-15)


def test_broadband_aperture_is_deterministic_and_band_is_labeled(tmp_path):
    gamma = 55.0
    u = np.array([
        1.2, -0.7,
        math.sqrt(gamma**2 - 1.0 - 1.2**2 - 0.7**2),
    ])
    electric = np.array([0.7e11, -0.3e11, 1.1e11])
    c_magnetic = np.array([0.2e11, 0.9e11, -0.1e11])
    direction = np.array([0.025, -0.012, 1.0])
    direction /= np.linalg.norm(direction)
    species = _species(
        u, electric, c_magnetic, dt=2.5e-18, weight=3.2)
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species,
        observer_time_edges=np.array([-1.0e-12, 1.0e-12]),
        detectors=[{
            "name": "selected",
            "direction": direction,
            "half_angle": 0.01,
            "aperture_quadrature": 5,
            "energy_bands": [{
                "name": "all", "energy_range": (0.0, 1.0),
            }],
        }],
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert not accumulator.needs_spectral_samples
    assert accumulator.angular_kernel_x is None
    assert "energy_outside_apertures" not in accumulator.accounting

    radiator.handle_radiation(0.0)
    _, _, _, beta, dot_beta, dot_beta_perp = _power_oracle(
        u, electric, c_magnetic)
    expected_broadband = (
        species.w[0] * species.dt
        * _angular_power_oracle(direction, beta, dot_beta)
    )
    direction_key = "detector/selected/direction"
    assert accumulator.data[direction_key].sum() == pytest.approx(
        expected_broadband, rel=3.0e-13)
    expected_band = (
        species.w[0] * species.dt
        * _angular_power_oracle(direction, beta, dot_beta_perp)
        * accumulator.spectral_cdf[-1]
    )
    band_key = "detector/selected/band/all/direction"
    assert accumulator.data[band_key].sum() == pytest.approx(
        expected_band, rel=3.0e-13)
    assert accumulator.data["detector/selected/aperture"].sum() > 0.0

    diagnostic.write_hdf5(1)
    with h5py.File(
            tmp_path / "hdf5" / "data00000001.h5", "r") as output:
        fields = output["data/1/fields"]
        name = next(
            value for value in fields
            if "ObserverTime_selected_band_all_direction" in value)
        record = fields[name]
        assert record.attrs["bandEnergyAngleCouplingRetained"] == 0
        assert "separable" in _attribute_text(
            record.attrs["bandSpectralAngularClosure"])
        assert "not_the_joint" in _attribute_text(
            record.attrs["bandClosureScope"])
        assert record.attrs[
            "spectralClosureTruncatedEnergyFraction"] == pytest.approx(
                radiator.spectral_truncated_fraction)
        assert "preceding_integer_step" in _attribute_text(
            record.attrs["picEventTimeStaggering"])


def test_position_only_source_moments_are_deterministic(tmp_path):
    gamma = 40.0
    positions = np.array([
        [-2.0e-6, 1.0e-6, 0.0],
        [1.0e-6, -1.0e-6, 2.0e-6],
        [3.0e-6, 2.0e-6, 4.0e-6],
    ])
    species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [1.0e11, 0.0, 0.0], [0.0, 0.0, 0.0],
        count=3, positions=positions,
    )
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species,
        source_moments=[{
            "name": "position", "quantities": "position",
        }],
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert not accumulator.needs_spectral_samples
    assert accumulator.angular_kernel_x is None
    radiator.handle_radiation(0.0)

    stats = accumulator.moment_stats["position"]
    components = source_moment_components(stats, ("position",))
    assert components["centroid_x"][0] == pytest.approx(
        positions[:, 0].mean())
    assert components["centroid_y"][0] == pytest.approx(
        positions[:, 1].mean())
    assert components["centroid_z"][0] == pytest.approx(
        positions[:, 2].mean())
    assert components["principal_axis_x"][0] >= 0.0
    assert not any("theta" in name for name in components)
    assert not any("observer_time" in name for name in components)


def test_schwinger_conditional_narrows_with_photon_energy(tmp_path):
    gamma = 80.0
    count = 6000
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    species = _species(
        u, [0.0, 0.0, 0.0], [0.0, 2.0e10, 0.0],
        count=count,
    )
    _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species,
        photon_energy_edges=np.array([0.0, 1.0e-12]),
        theta_x_edges=np.array([-0.2, 0.0, 0.2]),
        theta_y_edges=np.array([-0.2, 0.0, 0.2]),
    )
    accumulator = diagnostic.accumulators["electrons"]
    event = accumulator._observer_event(species, 0.0, np, slice(None))

    np.random.seed(10)
    low = accumulator._sample_direction(
        event, np.full(count, 0.02), np)
    np.random.seed(11)
    high = accumulator._sample_direction(
        event, np.full(count, 4.0), np)
    low_normal_angle = np.arctan2(low[1], low[2])
    high_normal_angle = np.arctan2(high[1], high[2])
    assert np.std(low_normal_angle) > 2.0 * np.std(high_normal_angle)
    assert abs(np.mean(low_normal_angle)) < 0.1 * np.std(low_normal_angle)


def test_source_time_metadata_and_interval_snapshots(tmp_path):
    gamma = 45.0
    species = _species(
        [0.5, -0.2, math.sqrt(gamma**2 - 1.0 - 0.5**2 - 0.2**2)],
        [0.8e11, -0.2e11, 0.4e11], [0.0, 0.6e11, 0.1e11],
        positions=np.array([0.2e-6, -0.3e-6, 0.4e-6]),
    )
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species,
        observer_time_edges=np.array([-2.0e-12, 0.0, 2.0e-12]),
        source_coordinate_edges={
            "x": np.array([-1.0e-6, 0.0, 1.0e-6]),
        },
        source_projections=[{
            "name": "x_time", "axes": ("x", "time"),
        }],
        source_moments=[{"name": "all"}],
        output_mode="both",
        samples_per_particle=2,
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert accumulator.needs_spectral_samples

    np.random.seed(4)
    radiator.handle_radiation(0.0)
    one_step_energy = float(
        accumulator.accounting["transverse_energy"][0])
    diagnostic.write_hdf5(1)
    np.random.seed(5)
    radiator.handle_radiation(species.dt)
    diagnostic.write_hdf5(2)

    with h5py.File(
            tmp_path / "hdf5" / "data00000002.h5", "r") as output:
        fields = output["data/2/fields"]
        source_name = next(
            name for name in fields
            if "Source_x_time" in name and name.endswith("_cumulative"))
        source = fields[source_name]
        assert "direction_conditioned" in _attribute_text(
            source.attrs["observerTimeConditioning"])
        assert "distinct_null_coordinates" in _attribute_text(
            source.attrs["observerTimeConditioning"])

        moment_name = next(
            name for name in fields
            if "SourceMoments_all" in name
            and name.endswith("centroid_observer_time"))
        moment = fields[moment_name]
        assert "direction_conditioned" in _attribute_text(
            moment.attrs["observerTimeConditioning"])

        cumulative_name = next(
            name for name in fields
            if "Accounting_electrons_cumulative_transverse_energy" in name)
        interval_name = next(
            name for name in fields
            if "Accounting_electrons_interval_transverse_energy" in name)
        assert fields[cumulative_name][0] == pytest.approx(
            2.0 * one_step_energy, rel=3.0e-14)
        assert fields[interval_name][0] == pytest.approx(
            one_step_energy, rel=3.0e-14)

        assert "radiationAxes" in output["data/2"]
        assert source.attrs["axisEdgePaths"].size == 2

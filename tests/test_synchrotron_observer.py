# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Focused contract tests for observer-frame synchrotron radiation."""

import json
import math
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, hbar, m_e

from fbpic.main import Simulation
from fbpic.openpmd_diag import (
    SynchrotronRadiationDiagnostic, merge_radiation_segments,
    radiation_segment_status, restart_from_checkpoint,
    set_periodic_checkpoint,
)
from fbpic.openpmd_diag.observer_radiation_diag import (
    angular_cell_measure,
    merge_source_moment_stats,
    source_moment_components,
    source_z_interval_components,
)
from fbpic.openpmd_diag.segment_checkpoint import (
    atomic_write_json, selected_checkpoint_manifest,
)
from fbpic.particles.elementary_process.synchrotron.observer import (
    ObserverFrameRadiationAccumulator,
    _RANDOM_STREAM_PHOTON_ENERGY,
    _event_uniform,
)
from fbpic.particles.elementary_process.synchrotron.radiator import (
    SynchrotronRadiator,
    _cached_spectral_cdf,
)
from fbpic.particles.particles import Particles
from fbpic.particles.tracking import ParticleTracker
from fbpic.particles.push.numba_methods import push_p_numba
from fbpic.utils.cuda import cuda_installed


FOUR_ACCELERATION_POWER_FACTOR = (
    e**2 / (6.0 * np.pi * epsilon_0 * c**3))
ANGULAR_POWER_FACTOR = e**2 / (16.0 * np.pi**2 * epsilon_0 * c)


def _species(u, electric, c_magnetic, dt=2.0e-18, count=1, weight=1.0,
             positions=None, charge=-e, mass=m_e, particle_ids=None):
    u = np.asarray(u, dtype=np.float64)
    electric = np.asarray(electric, dtype=np.float64)
    c_magnetic = np.asarray(c_magnetic, dtype=np.float64)
    gamma = math.sqrt(1.0 + np.dot(u, u))
    if positions is None:
        positions = np.zeros((count, 3))
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape == (3,):
        positions = np.repeat(positions[None, :], count, axis=0)


    if particle_ids is None:
        particle_ids = np.arange(count, dtype=np.uint64)
    particle_ids = np.asarray(particle_ids, dtype=np.uint64)
    if particle_ids.shape != (count,):
        raise ValueError("particle_ids must match the local particle count")
    tracker = SimpleNamespace(id=particle_ids.copy())
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
        injector=None, ionizer=None, tracker=tracker,
    )


def _activate(species, boost=None, gamma_cutoff=2.0, x_max=8.0):
    radiator = SynchrotronRadiator(
        species, gamma_cutoff=gamma_cutoff, x_max=x_max,
        n_samples=64, boost=boost,
    )
    species.synchrotron_radiator = radiator
    return radiator


def _diagnostic(tmp_path, species, **kwargs):
    kwargs.setdefault(
        "particle_batch_size", max(1, min(int(species.Ntot), 256)))
    return SynchrotronRadiationDiagnostic(
        period=1,
        species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1),
        write_dir=str(tmp_path),
        **kwargs,
    )


def _complete_impulse(radiator, species, simulation_time):
    # Compatibility path for callers that already own complete endpoints;
    # production Particles.push_p uses the bounded pusher-owned buffer.
    lower = tuple(
        component.copy()
        for component in (species.ux, species.uy, species.uz))
    push_p_numba(
        species.ux, species.uy, species.uz, species.inv_gamma,
        species.Ex, species.Ey, species.Ez,
        species.Bx, species.By, species.Bz,
        species.q, species.m, species.Ntot, species.dt,
    )
    radiator.end_momentum_push(lower, simulation_time)
    return lower


def _impulse_oracle(lower, upper, dt, gamma_boost=1.0, beta_boost=0.0):
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    gamma_minus = math.sqrt(1.0 + np.dot(lower, lower))
    gamma_plus = math.sqrt(1.0 + np.dot(upper, upper))
    delta_u = upper - lower
    gamma_sum = gamma_plus + gamma_minus
    delta_gamma = np.dot(delta_u, upper + lower) / gamma_sum
    norm = math.sqrt(4.0 + np.dot(delta_u, delta_u) - delta_gamma**2)
    four_u_sim = np.concatenate((
        [(gamma_plus + gamma_minus) / norm],
        (upper + lower) / norm,
    ))
    delta_tau = dt / four_u_sim[0]
    four_a_sim = c * np.concatenate(([delta_gamma], delta_u)) / delta_tau
    matrix = np.array([
        [gamma_boost, 0.0, 0.0, gamma_boost * beta_boost],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [gamma_boost * beta_boost, 0.0, 0.0, gamma_boost],
    ])
    four_u = matrix @ four_u_sim
    four_a = matrix @ four_a_sim
    gamma = four_u[0]
    beta = four_u[1:] / gamma
    beta2 = np.dot(beta, beta)
    cross = np.cross(beta, four_a[1:])
    perpendicular = (
        FOUR_ACCELERATION_POWER_FACTOR * np.dot(cross, cross) / beta2)
    parallel = (
        FOUR_ACCELERATION_POWER_FACTOR * four_a[0]**2
        / (gamma**2 * beta2))
    omega_c = (
        1.5 * gamma * np.linalg.norm(cross)
        / (c * beta2**1.5))
    dot_beta = (
        four_a[1:] - beta * four_a[0]) / (c * gamma**2)
    a_parallel = beta * four_a[0] / beta2
    dot_beta_perp = (four_a[1:] - a_parallel) / (c * gamma**2)
    return {
        "four_u_sim": four_u_sim,
        "four_a_sim": four_a_sim,
        "four_u": four_u,
        "four_a": four_a,
        "delta_tau": delta_tau,
        "dt_observer": gamma * delta_tau,
        "p_perp": perpendicular,
        "p_parallel": parallel,
        "omega_c": omega_c,
        "beta": beta,
        "dot_beta": dot_beta,
        "dot_beta_perp": dot_beta_perp,
    }


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


def test_activation_validates_species_cutoff_and_spectral_tail(tmp_path):
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

    tiny_species = _species(u, fields, fields)
    tiny_radiator = _activate(tiny_species, x_max=1.0e-10)
    tiny_diagnostic = _diagnostic(
        tmp_path / "tiny_kernel", tiny_species,
        photon_energy_edges=np.array([0.0, 1.0e-20]),
        theta_x_edges=np.array([-0.1, 0.1]),
        theta_y_edges=np.array([-0.1, 0.1]),
    )
    tiny_kernel = tiny_diagnostic.accumulators[
        "electrons"].angular_kernel_x
    assert tiny_kernel[-1] == pytest.approx(1.0e-10)
    assert np.all(np.diff(tiny_kernel) > 0.0)
    assert tiny_radiator.observer_accumulator is not None


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
    lower = _complete_impulse(radiator, species, ct_sim / c)
    lower_vector = np.array([component[0] for component in lower])
    upper_vector = np.array([
        species.ux[0], species.uy[0], species.uz[0]])
    oracle = _impulse_oracle(
        lower_vector, upper_vector, species.dt,
        gamma_boost, beta_boost)
    event = accumulator._observer_event(
        species, lower, ct_sim / c, np, slice(None))

    assert event["gamma"][0] == pytest.approx(
        oracle["four_u"][0], rel=4.0e-14)
    assert np.array([
        event["ux"][0], event["uy"][0], event["uz"][0],
    ]) == pytest.approx(oracle["four_u"][1:], rel=4.0e-14)
    assert event["dt_observer"][0] == pytest.approx(
        oracle["dt_observer"], rel=4.0e-14)
    assert event["p_perp"][0] == pytest.approx(
        oracle["p_perp"], rel=8.0e-13)
    assert event["p_parallel"][0] == pytest.approx(
        oracle["p_parallel"], rel=8.0e-13)
    assert event["omega_c"][0] == pytest.approx(
        oracle["omega_c"], rel=8.0e-13)
    assert event["x"][0] == pytest.approx(x_lab + translation[1])
    assert event["y"][0] == pytest.approx(y_lab + translation[2])
    assert event["z"][0] == pytest.approx(z_lab + translation[3])
    assert event["time"][0] == pytest.approx(
        (ct_lab + translation[0]) / c)

    minkowski_u2 = (
        oracle["four_u_sim"][0]**2
        - np.dot(oracle["four_u_sim"][1:],
                 oracle["four_u_sim"][1:]))
    u_dot_a = (
        oracle["four_u_sim"][0] * oracle["four_a_sim"][0]
        - np.dot(oracle["four_u_sim"][1:],
                 oracle["four_a_sim"][1:]))
    invariant_scale = (
        np.linalg.norm(oracle["four_u_sim"])
        * np.linalg.norm(oracle["four_a_sim"]))
    assert minkowski_u2 == pytest.approx(1.0, rel=5.0e-13)
    assert abs(u_dot_a) <= 2.0e-14 * invariant_scale
    assert event["orthogonality_relative_error"][0] < 2.0e-14
    assert event["p_perp"][0] + event["p_parallel"][0] == pytest.approx(
        event["invariant_power"][0], rel=2.0e-11)

    scale = species.w[0] * oracle["dt_observer"]
    assert accumulator.accounting["transverse_energy"][0] == pytest.approx(
        oracle["p_perp"] * scale, rel=8.0e-13)
    assert accumulator.accounting["longitudinal_energy"][0] == pytest.approx(
        oracle["p_parallel"] * scale, rel=8.0e-13)


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


def test_retarded_time_uses_stable_observer_light_front_coordinates():
    # Reconstructing ct and z separately erases this small null coordinate.
    event = {
        "ct_plus_z": np.array([2.0e20]),
        "ct_minus_z": np.array([3.0e-6]),
        "x": np.array([0.0]),
        "y": np.array([0.0]),
    }
    tau = ObserverFrameRadiationAccumulator._retarded_time(
        event, np.array([0.0, 0.0, 1.0]))
    assert tau[0] == pytest.approx(3.0e-6 / c, rel=2.0e-16)


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
    assert accumulator.needs_spectral_samples
    assert accumulator.needs_joint_band_model
    assert accumulator.angular_kernel_x is not None
    assert "energy_outside_apertures" not in accumulator.accounting

    lower = _complete_impulse(radiator, species, 0.0)
    lower_vector = np.array([component[0] for component in lower])
    upper_vector = np.array([
        species.ux[0], species.uy[0], species.uz[0]])
    oracle = _impulse_oracle(lower_vector, upper_vector, species.dt)
    event = accumulator._observer_event(
        species, lower, 0.0, np, slice(None))
    expected_broadband = (
        species.w[0] * oracle["dt_observer"]
        * _angular_power_oracle(
            direction, oracle["beta"], oracle["dot_beta"])
    )
    direction_key = "detector/selected/direction"
    assert accumulator.data[direction_key].sum() == pytest.approx(
        expected_broadband, rel=3.0e-13)
    band_fraction = accumulator._joint_band_fraction(
        direction, event, (0.0, 1.0), np)[0]
    expected_band = (
        species.w[0] * oracle["dt_observer"]
        * _angular_power_oracle(
            direction, oracle["beta"], oracle["dot_beta_perp"])
        * band_fraction
    )
    band_key = "detector/selected/band/all/direction"
    assert accumulator.data[band_key].sum() == pytest.approx(
        expected_band, rel=3.0e-13)
    assert accumulator.data["detector/selected/aperture"].sum() > 0.0
    deterministic = accumulator.accounting[
        "deterministic_broadband_aperture_energy/selected"][0]
    stochastic_inside = accumulator.accounting[
        "stochastic_spectral_angular_aperture_energy/selected"][0]
    stochastic_outside = accumulator.accounting[
        "stochastic_spectral_angular_outside_aperture_energy/selected"][0]
    stochastic_partition = accumulator.accounting[
        "stochastic_spectral_angular_partition_energy/selected"][0]
    assert deterministic > 0.0
    assert stochastic_inside + stochastic_outside == pytest.approx(
        stochastic_partition, rel=2.0e-15)
    assert stochastic_partition == pytest.approx(
        accumulator.accounting["transverse_energy"][0]
        * accumulator.spectral_cdf[-1], rel=2.0e-15)
    assert accumulator.sampling[
        "stochastic_spectral_angular_aperture_energy_sampling_variance/selected"
    ][0] > 0.0

    diagnostic.write_hdf5(1)
    with h5py.File(
            tmp_path / "hdf5" / "data00000000.h5", "r") as output:
        fields = output["data/0/fields"]
        name = next(
            value for value in fields
            if "ObserverTime_selected_band_all_direction" in value)
        record = fields[name]
        assert record.attrs["bandEnergyAngleCouplingRetained"] == 1
        assert "angle_conditioned" in _attribute_text(
            record.attrs["bandSpectralAngularClosure"])
        assert _attribute_text(record.attrs["bandMode"]) == "joint"
        assert record.attrs[
            "spectralClosureTruncatedEnergyFraction"] == pytest.approx(
                radiator.spectral_truncated_fraction)
        assert "integer_time_center" in _attribute_text(
            record.attrs["picEventTimeStaggering"])
        assert _attribute_text(record.attrs["eventModel"]) == (
            "centered_covariant_pusher_impulse_v1")


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
    _complete_impulse(radiator, species, 0.0)

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
    positions = np.zeros((count, 3))
    positions[:, 0] = np.linspace(-1.0e-6, 1.0e-6, count)
    species = _species(
        u, [0.0, 0.0, 0.0], [0.0, 2.0e10, 0.0],
        count=count, positions=positions,
    )
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species,
        photon_energy_edges=np.array([0.0, 1.0e-12]),
        theta_x_edges=np.array([-0.2, 0.0, 0.2]),
        theta_y_edges=np.array([-0.2, 0.0, 0.2]),
    )
    accumulator = diagnostic.accumulators["electrons"]
    lower = _complete_impulse(radiator, species, 0.0)
    event = accumulator._observer_event(
        species, lower, 0.0, np, slice(None))

    low = accumulator._sample_direction(
        event, np.full(count, 0.02), np, sample_index=0)
    high = accumulator._sample_direction(
        event, np.full(count, 4.0), np, sample_index=1)
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

    _complete_impulse(radiator, species, 0.0)
    first_step_energy = float(
        accumulator.accounting["transverse_energy"][0])
    diagnostic.write_hdf5(1)
    _complete_impulse(radiator, species, species.dt)
    cumulative_energy = float(
        accumulator.accounting["transverse_energy"][0])
    second_step_energy = cumulative_energy - first_step_energy
    diagnostic.write_hdf5(2)

    with h5py.File(
            tmp_path / "hdf5" / "data00000001.h5", "r") as output:
        fields = output["data/1/fields"]
        source_name = next(
            name for name in fields
            if "Source_x_time" in name and name.endswith("_cumulative"))
        source = fields[source_name]
        assert "direction_conditioned" in _attribute_text(
            source.attrs["observerTimeConditioning"])
        assert "distinct_null_coordinates" in _attribute_text(
            source.attrs["observerTimeConditioning"])
        assert _attribute_text(source.attrs["sourceTimeConvention"]) == (
            "sampled_photon_direction_source_time")

        moment_name = next(
            name for name in fields
            if "SourceMoments_all" in name
            and name.endswith("centroid_observer_time"))
        moment = fields[moment_name]
        assert "direction_conditioned" in _attribute_text(
            moment.attrs["observerTimeConditioning"])
        assert _attribute_text(moment.attrs["sourceTimeConvention"]) == (
            "sampled_photon_direction_source_time")
        assert any(
            "SourceMoments_all" in name
            and name.endswith("covariance_theta_x_observer_time")
            for name in fields)
        assert any(
            "SourceMoments_all" in name
            and name.endswith("correlation_theta_y_observer_time")
            for name in fields)

        cumulative_name = next(
            name for name in fields
            if "Accounting_electrons_cumulative_transverse_energy" in name)
        interval_name = next(
            name for name in fields
            if "Accounting_electrons_interval_transverse_energy" in name)
        assert fields[cumulative_name][0] == pytest.approx(
            cumulative_energy, rel=3.0e-14)
        assert fields[interval_name][0] == pytest.approx(
            second_step_energy, rel=3.0e-14)

        assert "radiationAxes" in output["data/1"]
        assert source.attrs["axisEdgePaths"].size == 2


def test_particles_push_hook_is_centered_and_excludes_later_jumps(tmp_path):
    gamma = 35.0
    species = _species(
        [0.3, -0.1, math.sqrt(gamma**2 - 1.0 - 0.3**2 - 0.1**2)],
        [0.9e11, -0.2e11, 0.3e11], [0.0, 0.5e11, 0.0],
    )
    _activate(species)
    diagnostic = _diagnostic(tmp_path, species, channels=["accounting"])
    accumulator = diagnostic.accumulators["electrons"]

    lower = np.array([
        species.ux[0], species.uy[0], species.uz[0]])
    species._push_p_with_radiation = lambda t, z_plane, event_index: (
        Particles._push_p_with_radiation(species, t, z_plane, event_index))
    Particles.push_p(species, 0.5 * species.dt)
    upper = np.array([
        species.ux[0], species.uy[0], species.uz[0]])
    oracle = _impulse_oracle(lower, upper, species.dt)
    expected = species.w[0] * oracle["p_perp"] * oracle["dt_observer"]
    assert accumulator.accounting["transverse_energy"][0] == pytest.approx(
        expected, rel=8.0e-13)
    assert accumulator.completed_event_count == 1
    assert accumulator.cumulative_timing["last_event_center"] == 0.0

    # A discrete process after the push must not be folded into this event.
    recorded = float(accumulator.accounting["transverse_energy"][0])
    species.ux += 5.0
    species.inv_gamma[:] = 1.0 / np.sqrt(
        1.0 + species.ux**2 + species.uy**2 + species.uz**2)
    assert accumulator.accounting["transverse_energy"][0] == recorded


def test_nonfinite_impulse_is_counted_and_never_accumulated(tmp_path):
    gamma = 30.0
    species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species, channels=["accounting"])
    accumulator = diagnostic.accumulators["electrons"]

    lower = tuple(
        component.copy() for component in (species.ux, species.uy, species.uz))
    species.ux[0] = np.nan
    radiator.end_momentum_push(lower, 0.0)
    assert accumulator.completed_event_count == 1
    assert accumulator.sampling["invalid_pusher_events"][0] == 1
    assert accumulator.accounting["transverse_energy"][0] == 0.0
    assert accumulator.accounting["longitudinal_energy"][0] == 0.0
    assert all(np.isfinite(value[0]) for value in accumulator.quality.values())


def test_empty_population_records_interval_and_new_particle_waits_for_push(
        tmp_path):
    gamma = 30.0
    momentum = np.array([
        0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    species = _species(
        momentum, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], count=0)
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species, channels=["accounting"])
    accumulator = diagnostic.accumulators["electrons"]

    radiator.end_momentum_push((species.ux, species.uy, species.uz), 0.0)
    assert accumulator.completed_event_count == 1
    assert accumulator.accounting["transverse_energy"][0] == 0.0

    # Simulate creation after that interval. Merely adding the particle does
    # not radiate; only its next explicitly bracketed pusher impulse does.
    species.Ntot = 1
    species.x = np.array([0.0])
    species.y = np.array([0.0])
    species.z = np.array([0.0])
    species.ux = np.array([momentum[0]])
    species.uy = np.array([momentum[1]])
    species.uz = np.array([momentum[2]])
    species.w = np.array([1.0])
    species.tracker.id = np.array([17], dtype=np.uint64)
    lower = tuple(
        component.copy() for component in (species.ux, species.uy, species.uz))
    assert accumulator.accounting["transverse_energy"][0] == 0.0
    species.ux += 0.02
    radiator.end_momentum_push(lower, species.dt)
    assert accumulator.completed_event_count == 2
    assert accumulator.accounting["transverse_energy"][0] > 0.0


def test_stateless_packets_ignore_batching_and_particle_sorting(tmp_path):
    count = 257
    gamma = 60.0
    u = np.array([0.2, -0.1, math.sqrt(
        gamma**2 - 1.0 - 0.2**2 - 0.1**2)])
    positions = np.zeros((count, 3))
    positions[:, 0] = np.linspace(-3.0e-6, 3.0e-6, count)

    def run(name, batch_size, order):
        ordered_positions = positions[order]
        local_count = len(order)
        species = _species(
            u, [0.8e11, -0.1e11, 0.2e11], [0.0, 0.7e11, 0.0],
            count=local_count, positions=ordered_positions, particle_ids=order,
        )
        radiator = _activate(species)
        diagnostic = _diagnostic(
            tmp_path / name, species,
            photon_energy_edges=np.array([0.0, 1.0e-16, 1.0e-14]),
            theta_x_edges=np.array([-0.2, -0.01, 0.01, 0.2]),
            theta_y_edges=np.array([-0.2, -0.01, 0.01, 0.2]),
            samples_per_particle=3,
            particle_batch_size=batch_size,
            random_seed=7341,
        )
        _complete_impulse(radiator, species, 0.0)
        accumulator = diagnostic.accumulators["electrons"]
        return (
            accumulator.data["angular_spectral"].copy(),
            float(accumulator.accounting["transverse_energy"][0]),
        )

    identity = np.arange(count)
    reverse = identity[::-1]
    reference, reference_energy = run("reference", count, identity)
    batched, batched_energy = run("batched", 7, identity)
    sorted_result, sorted_energy = run("sorted", 13, reverse)
    split = count // 2
    first_partition, first_energy = run(
        "partition_0", 11, identity[:split])
    second_partition, second_energy = run(
        "partition_1", 17, identity[split:])
    partitioned = first_partition + second_partition
    partitioned_energy = first_energy + second_energy
    assert batched == pytest.approx(reference, rel=2.0e-15, abs=0.0)
    assert sorted_result == pytest.approx(reference, rel=2.0e-15, abs=0.0)
    # The packet realization is identical; the final two-rank-style sum
    # differs only by ordinary floating-point reduction order.
    assert partitioned == pytest.approx(reference, rel=2.0e-14, abs=0.0)
    assert batched_energy == pytest.approx(reference_energy, rel=2.0e-15)
    assert sorted_energy == pytest.approx(reference_energy, rel=2.0e-15)
    assert partitioned_energy == pytest.approx(
        reference_energy, rel=2.0e-14)


@pytest.mark.skipif(not cuda_installed, reason="CUDA is not available")
def test_stateless_packets_match_for_identical_cpu_and_gpu_events():
    import cupy

    count = 129
    positions = np.zeros((count, 3))
    positions[:, 0] = np.linspace(-4.0e-6, 4.0e-6, count)
    lower_u = np.array([0.2, -0.1, math.sqrt(45.0**2 - 1.05)])

    def run(use_cuda):
        species = _species(
            lower_u, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
            count=count, positions=positions)
        species.use_cuda = use_cuda
        if use_cuda:
            for name in (
                    "x", "y", "z", "ux", "uy", "uz", "inv_gamma",
                    "w"):
                setattr(species, name, cupy.asarray(getattr(species, name)))
            species.tracker.id = cupy.asarray(species.tracker.id)
        radiator = _activate(species)
        accumulator = radiator.configure_observer_diagnostic(
            observer_frame="simulation",
            enabled_channels=["angular_spectral"],
            photon_energy_edges=np.array([0.0, 1.0e-17, 1.0e-14]),
            theta_x_edges=np.array([-0.2, -0.01, 0.01, 0.2]),
            theta_y_edges=np.array([-0.2, -0.01, 0.01, 0.2]),
            samples_per_particle=3, particle_batch_size=19,
            random_seed=8091,
        )
        lower = tuple(
            component.copy() for component in (species.ux, species.uy, species.uz))
        species.ux += 0.03
        species.uy -= 0.02
        radiator.end_momentum_push(lower, 0.0)
        radiator.receive_from_gpu()
        return accumulator.data["angular_spectral"].copy()

    cpu = run(False)
    gpu = run(True)
    assert np.array_equal(cpu > 0.0, gpu > 0.0)
    assert gpu == pytest.approx(cpu, rel=3.0e-13, abs=0.0)


def test_diagnostic_thinning_uses_unbiased_effective_weights(tmp_path):
    count = 1000
    probability = 0.2
    gamma = 30.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    positions = np.zeros((count, 3))
    positions[:, 0] = np.linspace(-2.0e-6, 2.0e-6, count)

    def run(name, fraction):
        species = _species(
            u, [1.0e11, 0.0, 0.0], [0.0, 0.0, 0.0],
            count=count, positions=positions,
        )
        radiator = _activate(species)
        diagnostic = _diagnostic(
            tmp_path / name, species, channels=["accounting"],
            particle_sampling_fraction=fraction, random_seed=919,
            particle_batch_size=31,
        )
        _complete_impulse(radiator, species, 0.0)
        return diagnostic.accumulators["electrons"]

    full = run("full", 1.0)
    thinned = run("thin", probability)
    sampled = thinned.sampling["sampled_macroparticle_events"][0]
    full_energy = full.accounting["transverse_energy"][0]
    expected_thinned = full_energy * sampled / (count * probability)
    assert thinned.accounting["transverse_energy"][0] == pytest.approx(
        expected_thinned, rel=3.0e-14)
    assert thinned.sampling["eligible_macroparticle_events"][0] == count
    assert thinned.sampling["effective_sampled_weight"][0] == pytest.approx(
        sampled / probability)
    assert thinned.sampling[
        "transverse_energy_sampling_variance"][0] > 0.0


def test_source_moments_remain_stable_at_large_coordinate_offset(tmp_path):
    offsets = np.array([-2.0, -1.0, 1.0, 2.0]) * 1.0e-3
    positions = np.zeros((offsets.size, 3))
    positions[:, 0] = 1.0e12 + offsets
    gamma = 25.0
    species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [1.0e11, 0.0, 0.0], [0.0, 0.0, 0.0],
        count=offsets.size, positions=positions,
    )
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species,
        source_moments=[{
            "name": "stable", "quantities": "position",
        }],
        particle_batch_size=2,
    )
    _complete_impulse(radiator, species, 0.0)
    stats = diagnostic.accumulators[
        "electrons"].moment_stats["stable"]
    components = source_moment_components(stats, ("position",))
    actual_x = positions[:, 0]
    expected_mean = actual_x[0] + np.mean(actual_x - actual_x[0])
    expected_rms = np.sqrt(np.mean((actual_x - expected_mean)**2))
    assert components["centroid_x"][0] == pytest.approx(expected_mean)
    assert components["rms_x"][0] == pytest.approx(
        expected_rms, rel=2.0e-14, abs=0.0)
    assert components["rms_x"][0] > 0.0


def test_complete_allocation_limit_is_enforced_and_exposed(tmp_path):
    gamma = 20.0
    species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [1.0e11, 0.0, 0.0], [0.0, 0.0, 0.0],
    )
    radiator = _activate(species)
    edges = np.linspace(-1.0, 1.0, 2001)
    with pytest.raises(MemoryError, match="Breakdown"):
        _diagnostic(
            tmp_path / "limited", species,
            source_coordinate_edges={"x": edges, "y": edges},
            source_projections=[{"name": "xy", "axes": ("x", "y")}],
            max_allocation_bytes=1024 * 1024,
        )
    assert radiator.observer_accumulator is None

    with pytest.raises(MemoryError, match="total radiation diagnostic"):
        _diagnostic(
            tmp_path / "complete_limit", species,
            source_coordinate_edges={
                "x": np.array([-1.0, 0.0, 1.0]),
            },
            source_projections=[{"name": "x", "axes": ("x",)}],
            max_allocation_bytes=1024,
        )
    assert radiator.observer_accumulator is None

    # A pathological aperture request must be rejected from its integer size
    # estimate without first allocating the quadrature rays.
    with pytest.raises(MemoryError, match="detector_aperture_quadrature"):
        _diagnostic(
            tmp_path / "quadrature_limit", species,
            observer_time_edges=np.array([-1.0, 1.0]),
            detectors=[{
                "name": "huge", "direction": [0.0, 0.0, 1.0],
                "half_angle": 1.0e-3,
                "aperture_quadrature": 100_000_000,
            }],
            max_allocation_bytes=1024 * 1024,
        )
    assert radiator.observer_accumulator is None

    diagnostic = _diagnostic(
        tmp_path / "small", species,
        source_coordinate_edges={
            "x": np.array([-1.0, 0.0, 1.0]),
        },
        source_projections=[{"name": "x", "axes": ("x",)}],
        max_allocation_bytes=64 * 1024,
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert accumulator.estimated_dense_product_bytes == 16
    assert accumulator.allocation_breakdown["source/x"] == 16
    assert accumulator.estimated_total_allocation_bytes > 16
    assert "pusher_radiation_endpoint_buffer" in (
        accumulator.allocation_breakdown)
    assert "temporary_writer_snapshot_copies" in (
        accumulator.allocation_breakdown)
    assert "radiator_activation_spectral_table_copies" in (
        accumulator.allocation_breakdown)
    estimate = diagnostic.get_memory_estimate()["electrons"]
    assert estimate["total_bytes"] == (
        accumulator.estimated_total_allocation_bytes)


def test_memory_preflight_precedes_automatic_identity_allocation(tmp_path):
    species = _species(
        [0.0, 0.0, 20.0], [1.0e11, 0.0, 0.0],
        [0.0, 0.0, 0.0], count=3)
    species.tracker = None
    species.n_integer_quantities = 0
    track_calls = []

    def track(comm):
        track_calls.append((comm.size, comm.rank))
        species.tracker = SimpleNamespace(
            id=np.arange(species.Ntot, dtype=np.uint64))
        species.n_integer_quantities += 1

    species.track = track
    radiator = _activate(species)
    with pytest.raises(MemoryError, match="total radiation diagnostic"):
        _diagnostic(
            tmp_path / "rejected", species, channels=["accounting"],
            max_allocation_bytes=1)
    assert track_calls == []
    assert species.tracker is None
    assert species.n_integer_quantities == 0
    assert radiator.observer_accumulator is None

    diagnostic = _diagnostic(
        tmp_path / "accepted", species, channels=["accounting"],
        max_allocation_bytes=64 * 1024)
    assert track_calls == [(1, 0)]
    assert species.n_integer_quantities == 1
    assert species.tracker.id.tolist() == [0, 1, 2]
    estimate = diagnostic.get_memory_estimate()["electrons"]["components"]
    assert estimate["persistent_particle_identity"] == 3 * 8


def test_detector_referenced_source_time_is_deterministic(tmp_path):
    gamma = 32.0
    position = np.array([0.4e-6, -0.2e-6, 0.7e-6])
    species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [0.8e11, 0.0, 0.0], [0.0, 0.0, 0.0],
        positions=position,
    )
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species,
        observer_time_edges=np.array([-2.0e-12, 0.0, 2.0e-12]),
        detectors=[{"name": "D", "direction": [0.0, 0.0, 1.0]}],
        source_projections=[{
            "name": "detector_time", "axes": ("time",),
            "time_reference": "D",
        }],
        source_moments=[{
            "name": "detector", "quantities": ("position", "time"),
            "time_reference": "D",
        }],
    )
    accumulator = diagnostic.accumulators["electrons"]
    assert not accumulator.needs_spectral_samples
    lower = _complete_impulse(radiator, species, 0.0)
    event = accumulator._observer_event(
        species, lower, 0.0, np, slice(None))
    expected_tau = event["time"][0] - event["z"][0] / c
    components = source_moment_components(
        accumulator.moment_stats["detector"], ("position", "time"))
    assert components["centroid_observer_time"][0] == pytest.approx(
        expected_tau)
    assert accumulator.data["source/detector_time"].sum() > 0.0

    diagnostic.write_hdf5(1)
    with h5py.File(
            tmp_path / "hdf5" / "data00000000.h5", "r") as output:
        fields = output["data/0/fields"]
        record = next(
            fields[name] for name in fields
            if "Source_detector_time" in name)
        assert _attribute_text(record.attrs["sourceTimeReference"]) == "D"
        assert _attribute_text(record.attrs["sourceTimeConvention"]) == (
            "fixed_detector_referenced_source_time")
        assert "tau_D" in _attribute_text(
            record.attrs["observerTimeDefinition"])


def test_joint_band_is_angle_conditioned_and_fast_mode_is_explicit(tmp_path):
    gamma = 55.0
    species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [1.0e11, 0.0, 0.0], [0.0, 0.0, 0.0],
    )
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path / "joint", species,
        observer_time_edges=np.array([-1.0e-12, 1.0e-12]),
        detectors=[{
            "name": "D", "direction": [0.0, 0.0, 1.0],
            "energy_bands": [{"name": "low", "energy_range": (0.0, 1.0)}],
        }],
    )
    accumulator = diagnostic.accumulators["electrons"]
    lower = _complete_impulse(radiator, species, 0.0)
    event = accumulator._observer_event(
        species, lower, 0.0, np, slice(None))
    cutoff = 0.2 * hbar * event["omega_c"][0]
    axis = np.array([0.0, 0.0, 1.0])
    off_plane = np.array([0.0, 0.03, 1.0])
    off_plane /= np.linalg.norm(off_plane)
    axis_fraction = accumulator._joint_band_fraction(
        axis, event, (0.0, cutoff), np)[0]
    off_plane_fraction = accumulator._joint_band_fraction(
        off_plane, event, (0.0, cutoff), np)[0]
    assert off_plane_fraction > axis_fraction

    fast_species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [1.0e11, 0.0, 0.0], [0.0, 0.0, 0.0],
    )
    fast_radiator = _activate(fast_species)
    fast_diagnostic = _diagnostic(
        tmp_path / "fast", fast_species,
        observer_time_edges=np.array([-1.0e-12, 1.0e-12]),
        energy_band_mode="separable",
        detectors=[{
            "name": "D", "direction": [0.0, 0.0, 1.0],
            "energy_bands": [{"name": "low", "energy_range": (0.0, cutoff)}],
        }],
    )
    assert fast_diagnostic.accumulators[
        "electrons"].detectors[0]["energy_band_mode"] == "separable"
    _complete_impulse(fast_radiator, fast_species, 0.0)
    fast_diagnostic.write_hdf5(1)
    with h5py.File(
            tmp_path / "fast" / "hdf5" / "data00000000.h5", "r") as output:
        fields = output["data/0/fields"]
        record = next(
            fields[name] for name in fields
            if "ObserverTime_D_band_low_direction" in name)
        assert record.attrs["bandEnergyAngleCouplingRetained"] == 0
        assert "explicit_fast" in _attribute_text(
            record.attrs["bandClosureScope"])


def test_off_cadence_final_flush_writes_current_event(tmp_path):
    gamma = 28.0
    species = _species(
        [0.0, 0.0, math.sqrt(gamma**2 - 1.0)],
        [1.0e11, 0.0, 0.0], [0.0, 0.0, 0.0],
    )
    radiator = _activate(species)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=10, species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1),
        write_dir=str(tmp_path), channels=["accounting"],
        particle_batch_size=1,
    )
    _complete_impulse(radiator, species, species.dt)
    diagnostic.write(1)
    assert not (tmp_path / "hdf5" / "data00000001.h5").exists()
    assert diagnostic.observer_writer.has_unwritten_events()
    assert diagnostic.finalize()
    assert not diagnostic.finalize()
    assert not diagnostic.flush(999)
    assert not diagnostic.observer_writer.has_unwritten_events()

    with h5py.File(
            tmp_path / "hdf5" / "data00000001.h5", "r") as output:
        iteration = output["data/1"]
        assert iteration.attrs["radiationFinalFlush"] == 1
        assert _attribute_text(
            iteration.attrs["radiationWriteTrigger"]) == "explicit_finalization"
        assert iteration.attrs[
            "radiationLastIncludedEventIteration"] == 1
        fields = iteration["fields"]
        record = next(iter(fields.values()))
        assert record.attrs["finalFlush"] == 1
        assert record.attrs["representedEventCount"] == 1
        assert record.attrs[
            "lastEventCenterTimeSimulation"] == species.dt


def test_simulation_post_impulse_phase_and_resumed_steps_do_not_collide(
        tmp_path):
    dt = 0.5e-6 / c
    simulation = Simulation(
        8, 8.0e-6, 2, 2.0e-6, 1, dt, zmin=0.0,
        boundaries={"z": "periodic", "r": "reflective"},
        verbose_level=0,
    )
    species = simulation.add_new_species(
        -e, m_e, n=1.0e18, p_nz=1, p_nr=1, p_nt=1,
        p_rmax=1.5e-6, uz_m=12.0, continuous_injection=False,
    )
    species.activate_synchrotron(
        gamma_cutoff=2.0, x_max=4.0, n_samples=32)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=2, species={"electrons": species},
        comm=simulation.comm, write_dir=str(tmp_path),
        channels=["accounting"],
    )
    simulation.diags = [diagnostic]

    simulation.step(1, show_progress=False)
    simulation.step(1, show_progress=False)
    assert simulation.iteration == 2
    assert diagnostic.accumulators[
        "electrons"].completed_event_count == 2
    assert (tmp_path / "hdf5" / "data00000000.h5").exists()
    # Returning from either step(1) call does not force an off-cadence file.
    assert diagnostic.observer_writer.has_unwritten_events()
    assert not (tmp_path / "hdf5" / "data00000001.h5").exists()
    assert simulation.finalize_diagnostics() == 1
    assert simulation.finalize_diagnostics() == 0
    assert (tmp_path / "hdf5" / "data00000001.h5").exists()

    # Resuming starts with event center 2. Its scheduled output has a distinct
    # iteration and must not collide with the preceding final flush.
    simulation.step(1, show_progress=False)
    assert simulation.iteration == 3
    assert diagnostic.accumulators[
        "electrons"].completed_event_count == 3
    assert (tmp_path / "hdf5" / "data00000002.h5").exists()

def test_resolution_indicators_and_energy_weighted_statistics(tmp_path):
    lower_vector = np.array([0.0, 0.0, 20.0])
    upper = np.array([
        [0.15, 0.02, 20.0],
        [0.8, -0.25, 19.7],
    ])
    gamma_minus = math.sqrt(1.0 + np.dot(lower_vector, lower_vector))
    gamma_plus = np.sqrt(1.0 + np.sum(upper**2, axis=1))
    lower = np.repeat(lower_vector[None, :], upper.shape[0], axis=0)
    endpoint_dot = gamma_minus * gamma_plus - np.sum(lower * upper, axis=1)
    expected_eta = np.arccosh(np.maximum(endpoint_dot, 1.0))
    lower_norm = np.linalg.norm(lower, axis=1)
    upper_norm = np.linalg.norm(upper, axis=1)
    expected_theta = np.arccos(np.clip(
        np.sum(lower * upper, axis=1) / (lower_norm * upper_norm),
        -1.0, 1.0))
    delta_u = upper - lower
    delta_gamma = gamma_plus - gamma_minus
    center_norm = np.sqrt(
        4.0 + np.sum(delta_u**2, axis=1) - delta_gamma**2)
    gamma_center = (gamma_plus + gamma_minus) / center_norm
    expected_chi = gamma_center * expected_theta
    expected = {
        "delta_eta": expected_eta,
        "delta_theta_u": expected_theta,
        "chi_turn": expected_chi,
    }
    thresholds = {
        name: 0.5 * (values.min() + values.max())
        for name, values in expected.items()}

    species = _species(
        lower_vector, [0.0, 0.0, 0.0], [0.0, 4.0e10, 0.0],
        count=2, particle_ids=np.array([101, 202], dtype=np.uint64))
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species, channels=["accounting"], output_mode="both",
        resolution_warning_thresholds=thresholds)
    accumulator = diagnostic.accumulators["electrons"]
    lower_arrays = (
        species.ux.copy(), species.uy.copy(), species.uz.copy())
    species.ux[:] = upper[:, 0]
    species.uy[:] = upper[:, 1]
    species.uz[:] = upper[:, 2]
    species.inv_gamma[:] = 1.0 / gamma_plus
    event = accumulator._observer_event(
        species, lower_arrays, 0.0, np, slice(None), event_index=0)
    radiator.end_momentum_push(lower_arrays, 0.0)

    resolution_energy = (
        species.w * event["p_perp"] * event["dt_observer"])
    denominator = resolution_energy.sum()
    assert denominator > 0.0
    for name, values in expected.items():
        assert event[name] == pytest.approx(values, rel=2.0e-12, abs=1.0e-15)
        state = accumulator.resolution_stats[name]
        assert state[0] == pytest.approx(values.max(), rel=2.0e-12)
        assert state[1] == pytest.approx(denominator, rel=2.0e-14)
        assert state[2] / denominator == pytest.approx(
            np.sum(resolution_energy * values) / denominator, rel=2.0e-12)
        assert math.sqrt(state[3] / denominator) == pytest.approx(
            math.sqrt(np.sum(resolution_energy * values**2) / denominator),
            rel=2.0e-12)
        assert state[4] / denominator == pytest.approx(
            resolution_energy[values > thresholds[name]].sum() / denominator,
            rel=2.0e-14)

    diagnostic.write_hdf5(0)
    with h5py.File(
            tmp_path / "hdf5" / "data00000000.h5", "r") as output:
        fields = output["data/0/fields"]
        for mode in ("cumulative", "interval"):
            prefix = "radiationEventQuality_electrons_%s_" % mode
            assert fields[prefix + "max_delta_eta"][0] == pytest.approx(
                expected_eta.max(), rel=2.0e-12)
            assert fields[
                prefix + "transverse_energy_weighted_rms_chi_turn"
            ][0] > 0.0
            fraction = fields[
                prefix
                + "transverse_energy_fraction_above_delta_theta_u_"
                + "warning_threshold"
            ][0]
            expected_fraction = (
                resolution_energy[
                    expected_theta > thresholds["delta_theta_u"]].sum()
                / denominator)
            assert fraction == pytest.approx(expected_fraction, rel=2.0e-14)
        quality = fields[
            "radiationEventQuality_electrons_cumulative_max_delta_eta"]
        assert quality.attrs["resolutionIndicatorsModifyEvents"] == 0
        assert "exact_unthinned" in _attribute_text(
            quality.attrs["resolutionWeighting"])


def test_source_z_central_energy_intervals_are_mergeable_products(tmp_path):
    count = 100
    edges = np.linspace(-5.0, 5.0, count + 1)
    positions = np.zeros((count, 3))
    positions[:, 2] = 0.5 * (edges[:-1] + edges[1:])
    species = _species(
        [0.0, 0.0, 25.0], [1.0e11, 0.0, 0.0],
        [0.0, 0.0, 0.0], count=count, positions=positions)
    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species,
        source_coordinate_edges={"z": edges},
        source_moments=[{
            "name": "all", "quantities": "position",
        }],
        source_z_intervals=(0.5, 0.9),
        output_mode="both")
    accumulator = diagnostic.accumulators["electrons"]
    _complete_impulse(radiator, species, 0.0)

    histogram = accumulator.source_z_histograms["all"]
    moment_energy = accumulator.moment_stats["all"][0]
    assert histogram.sum() == pytest.approx(moment_energy, rel=2.0e-14)
    assert histogram[0] == 0.0
    assert histogram[-1] == 0.0

    diagnostic.write_hdf5(0)
    with h5py.File(
            tmp_path / "hdf5" / "data00000000.h5", "r") as output:
        fields = output["data/0/fields"]
        prefix = "radiationSourceMoments_all_electrons_cumulative_"
        assert fields[prefix + "central_50_percent_z_start"][0] == (
            pytest.approx(-2.5, abs=2.0e-14))
        assert fields[prefix + "central_50_percent_z_end"][0] == (
            pytest.approx(2.5, abs=2.0e-14))
        assert fields[prefix + "central_90_percent_z_width"][0] == (
            pytest.approx(9.0, abs=3.0e-14))
        assert fields[prefix + "source_z_interval_contained_fraction"][0] == (
            pytest.approx(1.0))
        record = fields[prefix + "central_90_percent_z_width"]
        assert np.asarray(
            record.attrs["sourceZCentralIntervalFractions"]) == (
                pytest.approx(np.array([0.5, 0.9])))
        assert "equal_tail" in _attribute_text(
            record.attrs["sourceZIntervalDefinition"])
        interval_prefix = "radiationSourceMoments_all_electrons_interval_"
        assert fields[
            interval_prefix + "central_50_percent_z_width"
        ][0] == pytest.approx(5.0, abs=3.0e-14)


def test_scheduled_cadence_window_and_explicit_finalization_are_distinct(
        tmp_path):
    species = _species(
        [0.0, 0.0, 24.0], [1.0e11, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    radiator = _activate(species)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=2, iteration_min=2, iteration_max=4,
        species={"electrons": species},
        comm=SimpleNamespace(rank=0, size=1),
        write_dir=str(tmp_path), channels=["accounting"],
        output_mode="interval", particle_batch_size=1)

    for event_index in range(5):
        _complete_impulse(radiator, species, event_index * species.dt)
        diagnostic.write(event_index)

    output_dir = tmp_path / "hdf5"
    assert not (output_dir / "data00000000.h5").exists()
    assert not (output_dir / "data00000001.h5").exists()
    assert (output_dir / "data00000002.h5").exists()
    assert not (output_dir / "data00000003.h5").exists()
    assert not (output_dir / "data00000004.h5").exists()
    assert diagnostic.observer_writer.has_unwritten_events()

    assert diagnostic.finalize()
    assert not diagnostic.finalize()
    assert (output_dir / "data00000004.h5").exists()
    with h5py.File(output_dir / "data00000002.h5", "r") as output:
        record = next(iter(output["data/2/fields"].values()))
        assert record.attrs["representedEventCount"] == 3
        assert _attribute_text(
            output["data/2"].attrs["radiationWriteTrigger"]
        ) == "scheduled_cadence"
    with h5py.File(output_dir / "data00000004.h5", "r") as output:
        record = next(iter(output["data/4/fields"].values()))
        assert record.attrs["representedEventCount"] == 2
        assert _attribute_text(
            output["data/4"].attrs["radiationWriteTrigger"]
        ) == "explicit_finalization"


def test_bounded_endpoint_pusher_matches_unmodified_vay_push(tmp_path):
    count = 9
    batch_size = 3
    species = _species(
        [0.3, -0.2, 30.0], [0.8e11, -0.4e11, 0.2e11],
        [0.1e11, 0.5e11, -0.2e11], count=count)
    expected_ux = species.ux.copy()
    expected_uy = species.uy.copy()
    expected_uz = species.uz.copy()
    expected_inv_gamma = species.inv_gamma.copy()
    push_p_numba(
        expected_ux, expected_uy, expected_uz, expected_inv_gamma,
        species.Ex, species.Ey, species.Ez,
        species.Bx, species.By, species.Bz,
        species.q, species.m, species.Ntot, species.dt)

    radiator = _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species, channels=["accounting"],
        particle_batch_size=batch_size)
    accumulator = diagnostic.accumulators["electrons"]
    assert all(
        array.size == batch_size
        for array in accumulator.pusher_endpoint_buffer)
    with pytest.raises(RuntimeError, match="full-species"):
        radiator.begin_momentum_push()

    species._push_p_with_radiation = lambda t, z_plane, event_index: (
        Particles._push_p_with_radiation(
            species, t, z_plane, event_index))
    Particles.push_p(species, 0.5 * species.dt, event_index=0)
    assert np.array_equal(species.ux, expected_ux)
    assert np.array_equal(species.uy, expected_uy)
    assert np.array_equal(species.uz, expected_uz)
    assert np.array_equal(species.inv_gamma, expected_inv_gamma)
    assert accumulator.completed_event_count == 1
    assert accumulator.last_completed_event_index == 0

def test_species_local_particle_ids_use_independent_random_namespaces(tmp_path):
    first = _species(
        [0.0, 0.0, 20.0], [1.0e11, 0.0, 0.0],
        [0.0, 0.0, 0.0], particle_ids=np.array([7], dtype=np.uint64))
    second = _species(
        [0.0, 0.0, 20.0], [1.0e11, 0.0, 0.0],
        [0.0, 0.0, 0.0], particle_ids=np.array([7], dtype=np.uint64))
    _activate(first)
    _activate(second)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1,
        species={"electrons_a": first, "electrons_b": second},
        comm=SimpleNamespace(rank=0, size=1),
        write_dir=str(tmp_path), channels=["accounting"],
        particle_batch_size=1, random_seed=41,
        max_allocation_bytes=128 * 1024)
    first_accumulator = diagnostic.accumulators["electrons_a"]
    second_accumulator = diagnostic.accumulators["electrons_b"]
    assert first_accumulator.random_namespace != (
        second_accumulator.random_namespace)

    particle_id = np.array([7], dtype=np.uint64)
    event_index = np.array([0], dtype=np.uint64)
    first_sample = _event_uniform(
        41, particle_id, event_index, _RANDOM_STREAM_PHOTON_ENERGY,
        0, np, first_accumulator.random_namespace)
    second_sample = _event_uniform(
        41, particle_id, event_index, _RANDOM_STREAM_PHOTON_ENERGY,
        0, np, second_accumulator.random_namespace)
    assert first_sample[0] != second_sample[0]
    assert diagnostic.estimated_total_allocation_bytes == sum(
        item["total_bytes"] for item in diagnostic.memory_estimate.values())


def test_restored_particle_ids_advance_the_correct_global_counter():
    class GatheredMaxima(object):
        def allgather(self, value):
            assert value == 7
            return [4, value, 14]

    communicator = SimpleNamespace(
        rank=1, size=3, mpi_comm=GatheredMaxima())
    tracker = ParticleTracker(3, 1, 0)
    restored = np.array([1, 7], dtype=np.uint64)
    tracker.overwrite_ids(restored, communicator)

    assert np.array_equal(tracker.id, restored)
    assert tracker.next_attributed_id == 16
    assert not hasattr(tracker, "next_attibuted_id")
    generated = tracker.generate_new_ids(3)
    assert np.array_equal(generated, np.array([16, 19, 22]))
    assert not set(generated.tolist()).intersection({1, 4, 7, 14})

    exhausted = ParticleTracker(1, 0, 0)
    exhausted.overwrite_ids(
        np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
        SimpleNamespace(rank=0, size=1, mpi_comm=None))
    with pytest.raises(OverflowError, match="namespace is exhausted"):
        exhausted.generate_new_ids(1)


def test_legacy_restart_requires_explicit_discontinuous_segment(tmp_path):
    species = _species(
        [0.0, 0.0, 20.0], [1.0e11, 0.0, 0.0],
        [0.0, 0.0, 0.0])
    _activate(species)
    diagnostic = _diagnostic(
        tmp_path, species, channels=["accounting"])
    context = {
        "initial": True,
        "restart": True,
        "legacy": True,
        "run_id": "simulation-run",
        "checkpoint_id": None,
        "checkpoint_iteration": 5,
        "iteration": 5,
        "segments": [],
    }

    with pytest.raises(RuntimeError, match="restart_policy='new_segment'"):
        diagnostic.start_segment(context)
    diagnostic.restart_policy = "new_segment"
    assert diagnostic.start_segment(context)
    assert diagnostic.segment_status == "open"
    assert diagnostic._segment_event_begin == 5
    assert diagnostic._segment_run_id != context["run_id"]
    assert diagnostic._segment_continuity == (
        "discontinuous_legacy_restart")

    missing_manifest = tmp_path / "missing-commit.json"
    reference = diagnostic.close_segment({
        "checkpoint_id": None,
        "checkpoint_iteration": None,
        "event_end_exclusive": 5,
        "close_reason": "finalize",
        "commit_manifest": str(missing_manifest),
    })
    assert diagnostic.segment_status == "closed"
    assert radiation_segment_status(reference["path"]) == "orphaned"
    with pytest.raises(RuntimeError, match="not committed"):
        merge_radiation_segments(
            [reference["path"]], tmp_path / "must-not-merge.h5")


def test_preparing_checkpoint_manifest_is_never_accepted_as_legacy(tmp_path):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    atomic_write_json(
        str(manifest_dir / "checkpoint00000004.json"), {
            "checkpointManifestSchemaVersion": 1,
            "checkpointStatus": "preparing",
            "iteration": 4,
        })
    with pytest.raises(RuntimeError, match="not committed"):
        selected_checkpoint_manifest(str(tmp_path), 4)


def test_checkpoint_restart_creates_exact_mergeable_radiation_segments(
        tmp_path):
    def build_simulation():
        simulation = Simulation(
            8, 8.0e-6, 2, 2.0e-6, 1, 0.5e-6 / c, zmin=0.0,
            boundaries={"z": "periodic", "r": "reflective"},
            verbose_level=0)
        particles = simulation.add_new_species(
            -e, m_e, n=1.0e18, p_nz=1, p_nr=1, p_nt=1,
            p_rmax=1.5e-6, uz_m=12.0,
            continuous_injection=False)
        return simulation, particles

    checkpoint_dir = tmp_path / "checkpoints"
    radiation_dir = tmp_path / "radiation"
    simulation, species = build_simulation()

    # Registering checkpoints before the radiation diagnostic exercises the
    # late tracker-activation path: IDs still have to enter the checkpoint.
    set_periodic_checkpoint(
        simulation, 2, checkpoint_dir=str(checkpoint_dir))
    species.activate_synchrotron(
        gamma_cutoff=2.0, x_max=4.0, n_samples=32)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=7, species={"electrons": species}, comm=simulation.comm,
        write_dir=str(radiation_dir), channels=["accounting"])
    simulation.diags = [diagnostic]
    simulation.step(2, show_progress=False)

    manifest_path = (
        checkpoint_dir / "manifests" / "checkpoint00000002.json")
    with open(manifest_path, "r") as source:
        checkpoint_manifest = json.load(source)
    assert checkpoint_manifest["checkpointStatus"] == "committed"
    assert checkpoint_manifest["eventEndExclusive"] == 2
    assert len(checkpoint_manifest["segments"]) == 1
    first_reference = checkpoint_manifest["segments"][0]
    assert (first_reference["eventBegin"],
            first_reference["eventEndExclusive"]) == (0, 2)
    assert radiation_segment_status(first_reference["path"]) == "committed"
    with h5py.File(first_reference["path"], "r") as first_segment:
        segment_metadata = json.loads(
            first_segment["metadata/json"][()].decode("utf-8"))
    frame_times = segment_metadata["frameTimes"]["electrons"]
    assert frame_times["simulationEventCenterStart"] == pytest.approx(
        0.5 * simulation.dt)
    assert frame_times["simulationEventCenterEnd"] == pytest.approx(
        1.5 * simulation.dt)
    assert frame_times["simulationImpulseIntervalStart"] == pytest.approx(0.0)
    assert frame_times["simulationImpulseIntervalEnd"] == pytest.approx(
        2.0 * simulation.dt)
    assert diagnostic.get_segment_status()["eventBegin"] == 2
    assert diagnostic.segment_status == "open"

    restored_ids = species.tracker.id.copy()
    checkpoint_file = (
        checkpoint_dir / "proc0" / "hdf5" / "data00000002.h5")
    with h5py.File(checkpoint_file, "r") as checkpoint:
        particle_group = checkpoint["data/2/particles/species 0"]
        assert particle_group["id"].shape == restored_ids.shape
        assert tuple(particle_group["mass"].attrs["shape"]) == (
            restored_ids.size,)
        assert "radiationState" not in checkpoint

    restarted, restarted_species = build_simulation()
    restart_from_checkpoint(
        restarted, iteration=2, checkpoint_dir=str(checkpoint_dir))
    assert restarted_species._persistent_ids_restored
    assert np.array_equal(restarted_species.tracker.id, restored_ids)
    assert restarted_species.tracker.next_attributed_id > int(
        restored_ids.max())

    restarted_species.activate_synchrotron(
        gamma_cutoff=2.0, x_max=4.0, n_samples=32)
    restarted_diagnostic = SynchrotronRadiationDiagnostic(
        period=7, species={"electrons": restarted_species},
        comm=restarted.comm, write_dir=str(radiation_dir),
        channels=["accounting"])
    assert restarted_diagnostic.configuration_fingerprint == (
        diagnostic.configuration_fingerprint)
    restarted.diags = [restarted_diagnostic]
    restarted.step(1, show_progress=False)
    assert restarted.finalize_diagnostics() == 1

    second_reference = restarted_diagnostic.get_segment_status()[
        "lastClosed"]
    assert (second_reference["eventBegin"],
            second_reference["eventEndExclusive"]) == (2, 3)
    assert second_reference["parentCheckpointId"] == (
        first_reference["closingCheckpointId"])
    assert second_reference["runId"] == first_reference["runId"]
    assert radiation_segment_status(second_reference["path"]) == "committed"

    merged_path = tmp_path / "whole-run-radiation.h5"
    merge_radiation_segments(
        [second_reference["path"], first_reference["path"]], merged_path)
    with h5py.File(merged_path, "r") as merged:
        assert _attribute_text(merged.attrs["artifactType"]) == (
            "radiation_merged_whole_run")
        assert merged.attrs["mergedWholeRun"] == 1
        assert merged["radiationState/electrons/timing"].attrs[
            "event_count"] == 3
        metadata = json.loads(
            merged["metadata/json"][()].decode("utf-8"))
        assert [item["eventBegin"] for item in metadata["segments"]] == [0, 2]
        merged_transverse = merged[
            "radiationState/electrons/accounting/transverse_energy"][()]
    with h5py.File(first_reference["path"], "r") as first_segment:
        first_transverse = first_segment[
            "radiationState/electrons/accounting/transverse_energy"][()]
    with h5py.File(second_reference["path"], "r") as second_segment:
        second_transverse = second_segment[
            "radiationState/electrons/accounting/transverse_energy"][()]
    assert np.allclose(
        merged_transverse, first_transverse + second_transverse,
        rtol=0.0, atol=0.0)

    suffix_path = tmp_path / "selected-suffix.h5"
    merge_radiation_segments([second_reference["path"]], suffix_path)
    with h5py.File(suffix_path, "r") as suffix:
        assert _attribute_text(suffix.attrs["artifactType"]) == (
            "radiation_merged_lineage_selection")
        assert suffix.attrs["mergedWholeRun"] == 0

    with pytest.raises(ValueError, match="Duplicate radiation segment ID"):
        merge_radiation_segments(
            [first_reference["path"], first_reference["path"]],
            tmp_path / "duplicate.h5")




def test_older_checkpoint_restart_orphans_the_abandoned_branch(tmp_path):
    def configured_simulation():
        simulation = Simulation(
            8, 8.0e-6, 2, 2.0e-6, 1, 0.5e-6 / c, zmin=0.0,
            boundaries={"z": "periodic", "r": "reflective"},
            verbose_level=0)
        particles = simulation.add_new_species(
            -e, m_e, n=1.0e18, p_nz=1, p_nr=1, p_nt=1,
            p_rmax=1.5e-6, uz_m=12.0,
            continuous_injection=False)
        return simulation, particles

    checkpoint_dir = tmp_path / "checkpoints"
    radiation_dir = tmp_path / "radiation"
    simulation, species = configured_simulation()
    species.activate_synchrotron(
        gamma_cutoff=2.0, x_max=4.0, n_samples=32)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=9, species={"electrons": species}, comm=simulation.comm,
        write_dir=str(radiation_dir), channels=["accounting"])
    simulation.diags = [diagnostic]
    set_periodic_checkpoint(
        simulation, 1, checkpoint_dir=str(checkpoint_dir))
    simulation.step(2, show_progress=False)

    with open(
            checkpoint_dir / "manifests" / "checkpoint00000001.json",
            "r") as source:
        first_checkpoint = json.load(source)
    with open(
            checkpoint_dir / "manifests" / "checkpoint00000002.json",
            "r") as source:
        abandoned_checkpoint = json.load(source)
    first = first_checkpoint["segments"][0]
    abandoned = abandoned_checkpoint["segments"][0]
    assert radiation_segment_status(first["path"]) == "committed"
    assert radiation_segment_status(abandoned["path"]) == "committed"

    restarted, restarted_species = configured_simulation()
    restart_from_checkpoint(
        restarted, iteration=1, checkpoint_dir=str(checkpoint_dir))
    restarted_species.activate_synchrotron(
        gamma_cutoff=2.0, x_max=4.0, n_samples=32)
    restarted_diagnostic = SynchrotronRadiationDiagnostic(
        period=9, species={"electrons": restarted_species},
        comm=restarted.comm, write_dir=str(radiation_dir),
        channels=["accounting"])
    restarted.diags = [restarted_diagnostic]
    set_periodic_checkpoint(
        restarted, 1, checkpoint_dir=str(checkpoint_dir))
    restarted.step(1, show_progress=False)

    with open(
            checkpoint_dir / "manifests" / "checkpoint00000002.json",
            "r") as source:
        replacement_checkpoint = json.load(source)
    replacement = replacement_checkpoint["segments"][0]
    assert replacement["segmentId"] != abandoned["segmentId"]
    assert replacement["parentCheckpointId"] == first[
        "closingCheckpointId"]
    assert replacement_checkpoint["parentCheckpointId"] == first[
        "closingCheckpointId"]
    assert radiation_segment_status(first["path"]) == "committed"
    assert radiation_segment_status(replacement["path"]) == "committed"
    assert radiation_segment_status(abandoned["path"]) == "orphaned"

    assert restarted.finalize_diagnostics() == 1
    terminal = restarted_diagnostic.get_segment_status()["lastClosed"]
    merged_path = tmp_path / "replacement-branch.h5"
    merge_radiation_segments(
        [terminal["path"], replacement["path"], first["path"]],
        merged_path)
    with h5py.File(merged_path, "r") as merged:
        assert _attribute_text(merged.attrs["artifactType"]) == (
            "radiation_merged_whole_run")
        assert merged["radiationState/electrons/timing"].attrs[
            "event_count"] == 2
def test_merger_combines_sufficient_statistics_and_interval_histograms(
        tmp_path):
    species = _species(
        [0.0, 0.0, 25.0], [1.0e11, 0.0, 0.0],
        [0.0, 0.0, 0.0], positions=[0.0, 0.0, 2.0e-6])
    radiator = _activate(species)
    z_edges = np.linspace(-5.0e-6, 5.0e-6, 11)
    diagnostic = _diagnostic(
        tmp_path / "radiation", species,
        observer_time_edges=np.linspace(-2.0e-12, 2.0e-12, 9),
        detectors=[{"name": "D", "direction": [0.0, 0.0, 1.0]}],
        source_coordinate_edges={"z": z_edges},
        source_moments=[{"name": "all"}],
        source_z_intervals=(0.5, 0.9),
        channels=["observer_time", "source_moments", "accounting"])
    diagnostic.start_segment({
        "initial": True, "restart": False, "legacy": False,
        "run_id": "moment-run", "checkpoint_id": None,
        "checkpoint_iteration": None, "iteration": 0, "segments": [],
    })

    _complete_impulse(radiator, species, 0.0)
    checkpoint_manifest = tmp_path / "moment-checkpoint.json"
    first_reference = diagnostic.close_segment({
        "checkpoint_id": "moment-checkpoint",
        "checkpoint_iteration": 1,
        "event_end_exclusive": 1,
        "close_reason": "checkpoint",
        "commit_manifest": str(checkpoint_manifest),
    })
    assert radiation_segment_status(first_reference["path"]) == "orphaned"
    atomic_write_json(str(checkpoint_manifest), {
        "checkpointStatus": "committed",
        "checkpointManifestSchemaVersion": 1,
        "checkpointId": "moment-checkpoint",
        "parentCheckpointId": None,
        "parentCheckpointIteration": None,
        "iteration": 1,
        "eventEndExclusive": 1,
        "segments": [first_reference],
    })
    diagnostic.start_segment({
        "run_id": "moment-run",
        "checkpoint_id": "moment-checkpoint",
        "checkpoint_iteration": 1,
        "iteration": 1,
        "restart": False,
        "segments": [first_reference],
    })

    _complete_impulse(radiator, species, species.dt)
    assert diagnostic.finalize_segment()
    second_reference = diagnostic.get_segment_status()["lastClosed"]
    merged_path = tmp_path / "moments-merged.h5"
    merge_radiation_segments(
        [first_reference["path"], second_reference["path"]], merged_path)

    with h5py.File(first_reference["path"], "r") as first_file:
        first_moments = first_file[
            "radiationState/electrons/moments/all"][()]
        first_histogram = first_file[
            "radiationState/electrons/source_z/all"][()]
        first_detector = first_file[
            "radiationState/electrons/data/detector/D/direction"][()]
    with h5py.File(second_reference["path"], "r") as second_file:
        second_moments = second_file[
            "radiationState/electrons/moments/all"][()]
        second_histogram = second_file[
            "radiationState/electrons/source_z/all"][()]
        second_detector = second_file[
            "radiationState/electrons/data/detector/D/direction"][()]

    expected_moments = merge_source_moment_stats(
        first_moments, second_moments)
    expected_histogram = first_histogram + second_histogram
    expected_detector = first_detector + second_detector
    expected_components = source_moment_components(
        expected_moments, ("position", "angle", "time"))
    expected_intervals = source_z_interval_components(
        expected_histogram, z_edges, (0.5, 0.9))

    with h5py.File(merged_path, "r") as merged:
        assert np.array_equal(
            merged["radiationState/electrons/moments/all"][()],
            expected_moments)
        assert np.array_equal(
            merged["radiationState/electrons/source_z/all"][()],
            expected_histogram)
        assert np.array_equal(
            merged[
                "radiationState/electrons/data/detector/D/direction"][()],
            expected_detector)
        derived = merged["derived/electrons/sourceMoments/all"]
        assert derived["centroid_observer_time"][0] == pytest.approx(
            expected_components["centroid_observer_time"][0])
        assert derived["central_50_percent_z_start"][0] == pytest.approx(
            expected_intervals["central_50_percent_z_start"][0])
        assert "pulseMetrics/D/direction/interval_duration" in merged[
            "derived/electrons"]


def test_merger_rejects_gaps_overlaps_and_incompatible_configuration(
        tmp_path):
    def committed_segment(name, begin, seed=19):
        particles = _species(
            [0.0, 0.0, 20.0], [1.0e11, 0.0, 0.0],
            [0.0, 0.0, 0.0])
        radiator = _activate(particles)
        diagnostic = _diagnostic(
            tmp_path / name, particles, channels=["accounting"],
            random_seed=seed)
        diagnostic.start_segment({
            "initial": True,
            "restart": False,
            "legacy": False,
            "run_id": "range-run",
            "checkpoint_id": None,
            "checkpoint_iteration": None,
            "iteration": begin,
            "segments": [],
        })
        _complete_impulse(radiator, particles, begin * particles.dt)
        diagnostic.finalize_segment()
        return diagnostic.get_segment_status()["lastClosed"]

    first = committed_segment("first", 0)
    gap = committed_segment("gap", 2)
    overlap = committed_segment("overlap", 0)
    incompatible = committed_segment("incompatible", 1, seed=20)

    with pytest.raises(ValueError, match="Gap between radiation segments"):
        merge_radiation_segments(
            [first["path"], gap["path"]], tmp_path / "gap.h5")
    with pytest.raises(ValueError, match="Overlap between radiation segments"):
        merge_radiation_segments(
            [first["path"], overlap["path"]], tmp_path / "overlap.h5")
    with pytest.raises(ValueError, match="Incompatible radiation configuration"):
        merge_radiation_segments(
            [first["path"], incompatible["path"]],
            tmp_path / "incompatible.h5")

    shape_mismatch = committed_segment("shape-mismatch", 1)
    accounting_path = (
        "radiationState/electrons/accounting/transverse_energy")
    with h5py.File(shape_mismatch["path"], "r+") as segment:
        original = np.atleast_1d(np.asarray(segment[accounting_path][()]))
        del segment[accounting_path]
        segment.create_dataset(
            accounting_path, data=np.concatenate((original, original)))
        commit_manifest = _attribute_text(segment.attrs["commitManifest"])
    with pytest.raises(ValueError, match="accounting shape"):
        merge_radiation_segments(
            [first["path"], shape_mismatch["path"]],
            tmp_path / "shape-mismatch.h5")

    with open(commit_manifest, "r") as source:
        corrupted_manifest = json.load(source)
    corrupted_manifest["segments"][0]["eventEndExclusive"] = 99
    atomic_write_json(commit_manifest, corrupted_manifest)
    assert radiation_segment_status(shape_mismatch["path"]) == "orphaned"


def test_configuration_fingerprint_excludes_runtime_decomposition_details(
        tmp_path):
    def configured(name, batch_size, communicator_size, seed=13):
        particles = _species(
            [0.0, 0.0, 20.0], [1.0e11, 0.0, 0.0],
            [0.0, 0.0, 0.0])
        _activate(particles)
        return SynchrotronRadiationDiagnostic(
            period=1, species={"electrons": particles},
            comm=SimpleNamespace(rank=0, size=communicator_size),
            write_dir=str(tmp_path / name), channels=["accounting"],
            particle_batch_size=batch_size, random_seed=seed)

    serial = configured("serial", 1, 1)
    decomposed = configured("decomposed", 64, 4)
    different_seed = configured("different-seed", 1, 1, seed=14)

    assert serial.configuration_fingerprint == (
        decomposed.configuration_fingerprint)
    assert serial.configuration_fingerprint != (
        different_seed.configuration_fingerprint)

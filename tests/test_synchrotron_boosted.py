# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Tests for laboratory-frame synchrotron radiation in boosted simulations.

The event-level comparisons use independently transformed four-vectors and
field tensors. Their tolerances cover floating-point Lorentz reconstruction
error, while the one-percent spectral tolerance is set by interpolation of the
tabulated synchrotron profile. Statistical bounds follow from sample counts.
"""

import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, hbar, m_e
from scipy.integrate import quad, trapezoid
from scipy.special import kv
from openpmd_viewer import OpenPMDTimeSeries

from fbpic.main import Simulation
from fbpic.lpa_utils.boosted_frame import BoostConverter
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.openpmd_diag import SynchrotronRadiationDiagnostic
from fbpic.particles.elementary_process.synchrotron.inline_functions import (
    get_fields_lab_frame, get_particle_lab_frame
)
from fbpic.particles.elementary_process.synchrotron.radiator import (
    SynchrotronRadiator, _get_cuda_rng_seed
)
from fbpic.particles.particles import Particles
from fbpic.utils.cuda import cuda_installed
from fbpic.utils.random_seed import set_random_seed, _synchrotron_random

if cuda_installed:
    import cupy


E_MC = e / (m_e * c)


def _constant_external_field(F, x, y, z, t, amplitude, length_scale):
    """Add a spatially and temporally constant external field."""
    return F + amplitude


def _lorentz_tensor_boost(gamma_boost, four_vector, field_tensor):
    """Boost independent four-vector/tensor oracles from lab to simulation."""
    beta_boost = math.sqrt(1.0 - gamma_boost**-2)
    transform = np.eye(4)
    transform[0, 0] = gamma_boost
    transform[3, 3] = gamma_boost
    transform[0, 3] = -gamma_boost * beta_boost
    transform[3, 0] = -gamma_boost * beta_boost
    return (
        transform @ four_vector,
        transform @ field_tensor @ transform.T
    )


def _field_tensor(E, cB):
    """Return F in E-field units, with coordinates ordered ct, x, y, z."""
    Ex, Ey, Ez = E
    cBx, cBy, cBz = cB
    return np.array([
        [0.0, -Ex, -Ey, -Ez],
        [Ex, 0.0, -cBz, cBy],
        [Ey, cBz, 0.0, -cBx],
        [Ez, -cBy, cBx, 0.0],
    ])


def _fields_from_tensor(tensor):
    E = np.array([-tensor[0, 1], -tensor[0, 2], -tensor[0, 3]])
    cB = np.array([-tensor[2, 3], tensor[1, 3], -tensor[1, 2]])
    return E, cB


def _boost_lab_event(gamma_boost, u_lab, E_lab, cB_lab):
    """Generate simulation-frame event data using a tensor oracle."""
    gamma_lab = math.sqrt(1.0 + np.dot(u_lab, u_lab))
    four_momentum = np.r_[gamma_lab, u_lab]
    momentum_sim, tensor_sim = _lorentz_tensor_boost(
        gamma_boost, four_momentum, _field_tensor(E_lab, cB_lab)
    )
    E_sim, cB_sim = _fields_from_tensor(tensor_sim)
    return momentum_sim[1:], 1.0 / momentum_sim[0], E_sim, cB_sim


def _power_and_critical_frequency(u, E, cB):
    """Independent rest-frame-power and trajectory-curvature oracle."""
    gamma = math.sqrt(1.0 + np.dot(u, u))
    beta = u / gamma

    # Field seen in the instantaneous particle rest frame.
    E_rest = gamma * (E + np.cross(beta, cB))
    E_rest -= gamma**2 / (gamma + 1.0) * beta * np.dot(beta, E)
    proper_accel2 = E_MC**2 * np.dot(E_rest, E_rest)
    power = e**2 * proper_accel2 / (6.0 * np.pi * epsilon_0 * c)

    du_dt = -E_MC * (E + np.cross(beta, cB))
    u_abs = np.linalg.norm(u)
    cross_accel = np.linalg.norm(np.cross(u, du_dt))
    if cross_accel == 0.0:
        omega_c = 0.0
    else:
        omega_c = 1.5 * gamma**4 * cross_accel / u_abs**3
    return power, omega_c


def _dummy_species(dt, u, E, cB, weight=1.0, count=1, use_cuda=False):
    """Create only the particle arrays needed by SynchrotronRadiator."""
    gamma = math.sqrt(1.0 + np.dot(u, u))

    def array(value):
        values = np.full(count, value, dtype=np.float64)
        if use_cuda:
            return cupy.asarray(values)
        return values

    return SimpleNamespace(
        use_cuda=use_cuda, dt=dt, Ntot=count,
        ux=array(u[0]), uy=array(u[1]), uz=array(u[2]),
        Ex=array(E[0]), Ey=array(E[1]), Ez=array(E[2]),
        Bx=array(cB[0] / c), By=array(cB[1] / c),
        Bz=array(cB[2] / c), w=array(weight),
        inv_gamma=array(1.0 / gamma)
    )


def _make_radiator(species, energy_axis, theta_x_axis, theta_y_axis,
                   gamma_cutoff=10.0, boost=None, n_samples=256):
    return SynchrotronRadiator(
        species, energy_axis, theta_x_axis, theta_y_axis,
        gamma_cutoff, False, 20.0, n_samples, boost
    )


def _to_numpy(array):
    if hasattr(array, "get"):
        return array.get()
    return array


@pytest.mark.parametrize(
    "gamma_boost", [1.0, 1.25, 2.0, 4.0, 8.0, 10.0]
)
@pytest.mark.parametrize(
    "u_lab,E_lab,cB_lab",
    [
        (
            np.array([0.8, -1.2, 39.968]),
            np.array([2.0e11, -1.0e11, 4.0e10]),
            np.array([9.0e10, -5.4e10, 2.25e10]),
        ),
        (
            np.array([8.0, -4.0, 30.0]),
            np.array([0.5e12, -1.2e12, 0.7e12]),
            np.array([-0.3e12, 0.8e12, 1.1e12]),
        ),
        (
            np.array([math.sqrt(63.0), 0.0, 0.0]),
            np.array([-3.0e10, 5.0e10, 2.0e10]),
            np.array([7.0e10, -4.0e10, 6.0e10]),
        ),
    ]
)
def test_lab_reconstruction_against_tensor_oracle(
        gamma_boost, u_lab, E_lab, cB_lab):
    """Momentum and fields use FBPIC's independently checked boost sign."""
    u_sim, gamma_inv_sim, E_sim, cB_sim = _boost_lab_event(
        gamma_boost, u_lab, E_lab, cB_lab
    )
    beta_boost = math.sqrt(1.0 - gamma_boost**-2)

    reconstructed = get_particle_lab_frame(
        *u_sim, gamma_inv_sim, gamma_boost, beta_boost
    )
    fields = get_fields_lab_frame(
        *E_sim, *cB_sim, gamma_boost, beta_boost
    )

    gamma_lab = math.sqrt(1.0 + np.dot(u_lab, u_lab))
    gamma_sim = 1.0 / gamma_inv_sim
    assert np.allclose(reconstructed[:3], u_lab, rtol=2.0e-13, atol=2.0e-13)
    assert np.isclose(
        1.0 / reconstructed[3], gamma_lab, rtol=2.0e-13, atol=0.0
    )
    assert np.isclose(reconstructed[4], gamma_lab / gamma_sim,
                      rtol=2.0e-13, atol=0.0)
    assert np.allclose(fields[:3], E_lab, rtol=3.0e-13, atol=3.0e-3)
    assert np.allclose(fields[3:], cB_lab, rtol=3.0e-13, atol=3.0e-3)

    # Independent electromagnetic invariants.
    assert np.isclose(
        np.dot(fields[:3], fields[:3]) - np.dot(fields[3:], fields[3:]),
        np.dot(E_lab, E_lab) - np.dot(cB_lab, cB_lab),
        rtol=5.0e-13, atol=1.0
    )
    assert np.isclose(
        np.dot(fields[:3], fields[3:]), np.dot(E_lab, cB_lab),
        rtol=5.0e-13, atol=1.0
    )


def test_closed_form_boost_sign_and_synchrotron_scalars():
    """Closed-form signs agree with a pure transverse magnetic field."""
    gamma_boost = 4.0
    beta_boost = math.sqrt(1.0 - gamma_boost**-2)
    gamma_lab = 100.0
    uz_lab = math.sqrt(gamma_lab**2 - 1.0)
    u_lab = np.array([0.0, 0.0, uz_lab])
    field_amplitude = c * 1.0e3
    E_lab = np.zeros(3)
    cB_lab = np.array([0.0, field_amplitude, 0.0])

    u_sim, gamma_inv_sim, E_sim, cB_sim = _boost_lab_event(
        gamma_boost, u_lab, E_lab, cB_lab
    )
    expected_gamma_sim = gamma_boost * (
        gamma_lab - beta_boost * uz_lab
    )
    expected_uz_sim = gamma_boost * (
        uz_lab - beta_boost * gamma_lab
    )
    assert np.isclose(1.0 / gamma_inv_sim, expected_gamma_sim,
                      rtol=2.0e-14, atol=0.0)
    assert np.isclose(
        u_sim[2], expected_uz_sim, rtol=2.0e-14, atol=0.0
    )
    assert np.isclose(E_sim[0],
                      -gamma_boost * beta_boost * field_amplitude,
                      rtol=2.0e-14, atol=0.0)
    assert np.isclose(cB_sim[1], gamma_boost * field_amplitude,
                      rtol=2.0e-14, atol=0.0)

    recovered_fields = get_fields_lab_frame(
        *E_sim, *cB_sim, gamma_boost, beta_boost
    )
    assert np.allclose(recovered_fields[:3], E_lab,
                       rtol=2.0e-14, atol=1.0e-3)
    assert np.allclose(recovered_fields[3:], cB_lab,
                       rtol=2.0e-14, atol=1.0e-3)

    power, omega_c = _power_and_critical_frequency(
        u_lab, E_lab, cB_lab
    )
    beta_particle = uz_lab / gamma_lab
    cyclotron_frequency = E_MC * field_amplitude
    expected_power = e**2 / (6.0 * np.pi * epsilon_0 * c) * (
        gamma_lab * beta_particle * cyclotron_frequency
    )**2
    expected_omega_c = 1.5 * gamma_lab**2 * \
        cyclotron_frequency / beta_particle
    assert np.isclose(power, expected_power, rtol=2.0e-14, atol=0.0)
    assert np.isclose(omega_c, expected_omega_c, rtol=2.0e-14, atol=0.0)


@pytest.mark.parametrize("gamma_boost", [1.0e2, 1.0e4, 1.0e6])
def test_lightfront_field_transform_avoids_cancellation(gamma_boost):
    """Large opposing simulation-frame fields reconstruct accurately."""
    beta_boost = math.sqrt(1.0 - gamma_boost**-2)
    lightfront_boost = gamma_boost * (1.0 + beta_boost)
    amplitude = 3.0e11
    fields = get_fields_lab_frame(
        lightfront_boost * amplitude,
        lightfront_boost * amplitude,
        0.25 * amplitude,
        lightfront_boost * amplitude,
        -lightfront_boost * amplitude,
        -0.5 * amplitude,
        gamma_boost, beta_boost
    )
    expected = np.array([
        amplitude, amplitude, 0.25 * amplitude,
        amplitude, -amplitude, -0.5 * amplitude
    ])
    assert np.allclose(fields, expected, rtol=5.0e-15, atol=1.0e-3)


@pytest.mark.parametrize("gamma_boost", [1.0e2, 1.0e4, 1.0e6])
def test_lightfront_transform_is_stable_for_lab_rest(gamma_boost):
    """Lab-rest momentum is an inverse-boost cancellation adversary."""
    beta_boost = math.sqrt(1.0 - gamma_boost**-2)
    recovered = get_particle_lab_frame(
        0.0, 0.0, -gamma_boost * beta_boost,
        1.0 / gamma_boost, gamma_boost, beta_boost
    )
    assert np.isclose(
        1.0 / recovered[3], 1.0, rtol=5.0e-11, atol=0.0
    )
    assert np.isclose(recovered[2], 0.0, atol=5.0e-11)
    assert np.isclose(
        recovered[4], 1.0 / gamma_boost, rtol=5.0e-11, atol=0.0
    )


@pytest.mark.parametrize(
    "gamma_boost", [1.0, 1.25, 2.0, 4.0, 8.0, 10.0]
)
def test_single_event_lab_boost_equivalence_cpu(gamma_boost):
    """The CPU path agrees for one event and lab worldline element."""
    u_lab = np.array([8.0, -4.0, 30.0])
    E_lab = np.array([0.5e12, -1.2e12, 0.7e12])
    cB_lab = np.array([-0.3e12, 0.8e12, 1.1e12])
    gamma_lab = math.sqrt(1.0 + np.dot(u_lab, u_lab))
    power, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)
    assert power > 0.0
    assert omega_c > 0.0

    u_sim, gamma_inv_sim, E_sim, cB_sim = _boost_lab_event(
        gamma_boost, u_lab, E_lab, cB_lab
    )
    dt_lab = 2.0e-18
    dt_ratio = gamma_lab * gamma_inv_sim
    energy_axis = (0.1 * hbar * omega_c, 3.0 * hbar * omega_c, 48)
    theta_x = (0.12, 0.40, 57)
    theta_y = (-0.28, 0.04, 65)

    lab_species = _dummy_species(dt_lab, u_lab, E_lab, cB_lab)
    sim_species = _dummy_species(
        dt_lab / dt_ratio, u_sim, E_sim, cB_sim
    )
    boost = BoostConverter(gamma_boost)

    set_random_seed(37)
    lab = _make_radiator(
        lab_species, energy_axis, theta_x, theta_y, 20.0, None
    )
    lab.handle_radiation()
    set_random_seed(37)
    boosted = _make_radiator(
        sim_species, energy_axis, theta_x, theta_y, 20.0, boost
    )
    boosted.handle_radiation()

    assert np.allclose(
        boosted.radiation_data, lab.radiation_data,
        rtol=2.0e-10, atol=1.0e-12 * lab.radiation_data.max()
    )


def test_lab_cutoff_overrides_simulation_gamma():
    """Both directions of a large lab/simulation gamma mismatch are tested."""
    boost = BoostConverter(4.0)

    # The requested high-sensitivity trap: lab gamma 200 maps to simulation
    # gamma 25.4. A lab cutoff of 100 must retain this particle.
    gamma_lab = 200.0
    u_lab = np.array([0.0, 0.0, math.sqrt(gamma_lab**2 - 1.0)])
    E_lab = np.zeros(3)
    cB_lab = np.array([0.0, c * 1.0e3, 0.0])
    u_sim, inv_gamma_sim, E_sim, cB_sim = _boost_lab_event(
        4.0, u_lab, E_lab, cB_lab
    )
    assert 1.0 / inv_gamma_sim < 100.0 < gamma_lab
    _, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)
    dt_ratio = gamma_lab * inv_gamma_sim
    species = _dummy_species(
        1.0e-18 / dt_ratio, u_sim, E_sim, cB_sim
    )
    energy = (0.1 * hbar * omega_c, 3.0 * hbar * omega_c, 32)
    radiator = _make_radiator(
        species, energy, (-0.05, 0.05, 21), (-0.05, 0.05, 23),
        100.0, boost
    )
    set_random_seed(4)
    radiator.handle_radiation()
    assert radiator.radiation_data.sum() > 0.0

    # Conversely, transverse lab gamma 40 maps to simulation gamma 160.
    # The same lab cutoff must still exclude it.
    gamma_lab = 40.0
    u_lab = np.array([math.sqrt(gamma_lab**2 - 1.0), 0.0, 0.0])
    cB_lab = np.array([0.0, 0.0, c * 1.0e3])
    u_sim, inv_gamma_sim, E_sim, cB_sim = _boost_lab_event(
        4.0, u_lab, E_lab, cB_lab
    )
    assert gamma_lab < 100.0 < 1.0 / inv_gamma_sim
    _, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)
    species = _dummy_species(
        1.0e-18 / (gamma_lab * inv_gamma_sim),
        u_sim, E_sim, cB_sim
    )
    energy = (0.1 * hbar * omega_c, 3.0 * hbar * omega_c, 32)
    radiator = _make_radiator(
        species, energy,
        (0.5 * np.pi - 0.1, 0.5 * np.pi + 0.1, 21),
        (-0.1, 0.1, 23), 100.0, boost
    )
    set_random_seed(4)
    radiator.handle_radiation()
    assert radiator.radiation_data.sum() == 0.0


def test_forward_worldline_time_conversion_trap():
    """A forward gamma-200 event requires nearly eight boosted timesteps."""
    gamma_lab = 200.0
    u_lab = np.array([0.0, 0.0, math.sqrt(gamma_lab**2 - 1.0)])
    E_lab = np.zeros(3)
    cB_lab = np.array([0.0, c * 1.0e3, 0.0])
    boost = BoostConverter(4.0)
    u_sim, inv_gamma_sim, E_sim, cB_sim = _boost_lab_event(
        4.0, u_lab, E_lab, cB_lab
    )
    dt_ratio = gamma_lab * inv_gamma_sim
    assert np.isclose(
        dt_ratio, 7.869983689729365, rtol=5.0e-14, atol=0.0
    )

    _, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)
    energy = (0.2 * hbar * omega_c, 2.0 * hbar * omega_c, 32)
    theta_x = (-0.02, 0.025, 35)
    theta_y = (-0.018, 0.022, 37)
    dt_lab = 2.0e-18

    def run(u, E, cB, dt, boost_arg):
        species = _dummy_species(dt, u, E, cB)
        radiator = _make_radiator(
            species, energy, theta_x, theta_y, 100.0,
            boost_arg, 256
        )
        set_random_seed(811)
        radiator.handle_radiation()
        return radiator.radiation_data

    lab = run(u_lab, E_lab, cB_lab, dt_lab, None)
    boosted = run(
        u_sim, E_sim, cB_sim, dt_lab / dt_ratio, boost
    )

    # Reference for an implementation that evaluates lab physics but uses
    # dt' as though it were dt. It is lower by exactly dt/dt'.
    omitted_time_factor = run(
        u_lab, E_lab, cB_lab, dt_lab / dt_ratio, None
    )
    scale = lab.max()
    assert scale > 0.0
    assert np.allclose(
        boosted, lab, rtol=2.0e-10, atol=1.0e-12 * scale
    )
    assert np.isclose(
        omitted_time_factor.sum() / lab.sum(),
        1.0 / dt_ratio, rtol=2.0e-13, atol=0.0
    )


def test_spectrum_power_critical_energy_and_weight_oracle():
    """Sampled spectrum follows an independent rest-frame/curvature oracle."""
    gamma = 60.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    E = np.zeros(3)
    cB = np.array([0.0, c * 2.0e3, 0.0])
    power, omega_c = _power_and_critical_frequency(u, E, cB)
    dt = 3.0e-18
    weight = 5.0
    xi = np.linspace(0.1, 3.0, 32)
    energy = (xi[0] * hbar * omega_c, xi[-1] * hbar * omega_c,
              xi.size)
    species = _dummy_species(dt, u, E, cB, weight=weight)
    radiator = _make_radiator(
        species, energy, (-0.1, 0.1, 41), (-0.08, 0.12, 45),
        10.0, None, 2048
    )
    set_random_seed(9)
    radiator.handle_radiation()

    spectrum = radiator.radiation_data.sum(axis=(0, 1))
    spectrum *= radiator.d_theta_x * radiator.d_theta_y

    profile = np.array([
        9.0 * math.sqrt(3.0) / (8.0 * np.pi) * x
        * quad(lambda value: kv(5.0 / 3.0, value), x, np.inf)[0]
        for x in xi
    ])
    expected = weight * power * dt / (hbar * omega_c) * profile
    assert np.allclose(spectrum, expected, rtol=1.0e-2, atol=0.0)

    emitted_on_grid = trapezoid(spectrum, radiator.omega_ax * hbar)
    expected_on_grid = weight * power * dt * trapezoid(profile, xi)
    assert np.isclose(
        emitted_on_grid, expected_on_grid, rtol=1.0e-2, atol=0.0
    )


def test_force_free_and_parallel_curvature_are_zero():
    """Vanishing curvature produces a clean zero spectrum without NaNs."""
    gamma = 50.0
    beta_z = math.sqrt(1.0 - gamma**-2)
    u = np.array([0.0, 0.0, gamma * beta_z])

    cases = [
        (np.zeros(3), np.zeros(3)),
        (np.array([beta_z * c * 1.0e3, 0.0, 0.0]),
         np.array([0.0, c * 1.0e3, 0.0])),
        (np.array([0.0, 0.0, 1.0e12]), np.zeros(3)),
        (np.zeros(3), np.array([0.0, 0.0, c * 1.0e3])),
    ]
    for E, cB in cases:
        for gamma_boost in (1.0, 4.0):
            if gamma_boost == 1.0:
                u_input, E_input, cB_input = u, E, cB
                inv_gamma_input = 1.0 / gamma
                boost = None
            else:
                u_input, inv_gamma_input, E_input, cB_input = \
                    _boost_lab_event(gamma_boost, u, E, cB)
                boost = BoostConverter(gamma_boost)
            dt_ratio = gamma * inv_gamma_input
            species = _dummy_species(
                1.0e-18 / dt_ratio, u_input, E_input, cB_input
            )
            radiator = _make_radiator(
                species, (hbar * 1.0e5, hbar * 1.0e7, 16),
                (-0.1, 0.1, 11), (-0.1, 0.1, 13), boost=boost
            )
            set_random_seed(2)
            radiator.handle_radiation()
            assert np.isfinite(radiator.radiation_data).all()
            assert radiator.radiation_data.sum() == 0.0


def test_boosted_angular_statistics_and_cpu_reproducibility():
    """The stochastic lab spread uses lab gamma and is seed-reproducible."""
    gamma_lab = 20.0
    u_lab = np.array([math.sqrt(gamma_lab**2 - 1.0), 0.0, 0.0])
    E_lab = np.zeros(3)
    cB_lab = np.array([0.0, 0.0, c * 1.0e3])
    boost = BoostConverter(4.0)
    u_sim, inv_gamma_sim, E_sim, cB_sim = _boost_lab_event(
        4.0, u_lab, E_lab, cB_lab
    )
    dt_ratio = gamma_lab * inv_gamma_sim
    _, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)
    count = 8192
    energy = (0.2 * hbar * omega_c, 2.0 * hbar * omega_c, 12)
    theta_x = (0.5 * np.pi - 0.12, 0.5 * np.pi + 0.12, 121)
    theta_y = (-0.12, 0.12, 121)

    def run(seed, boosted):
        if boosted:
            u, E, cB = u_sim, E_sim, cB_sim
            dt, boost_arg = 1.0e-18 / dt_ratio, boost
        else:
            u, E, cB = u_lab, E_lab, cB_lab
            dt, boost_arg = 1.0e-18, None
        species = _dummy_species(dt, u, E, cB, count=count)
        radiator = _make_radiator(
            species, energy, theta_x, theta_y, 10.0,
            boost_arg, 128
        )
        set_random_seed(seed)
        radiator.handle_radiation()
        return radiator

    lab = run(419, False)
    first = run(419, True)
    second = run(419, True)
    different = run(420, True)
    assert np.allclose(
        first.radiation_data, lab.radiation_data,
        rtol=2.0e-10, atol=1.0e-12 * lab.radiation_data.max()
    )
    assert np.array_equal(first.radiation_data, second.radiation_data)
    assert not np.array_equal(first.radiation_data, different.radiation_data)

    angular = first.radiation_data.sum(axis=2)
    x_axis = np.linspace(theta_x[0], theta_x[1], theta_x[2])
    y_axis = np.linspace(theta_y[0], theta_y[1], theta_y[2])
    x_weights = angular.sum(axis=1)
    y_weights = angular.sum(axis=0)
    x_mean = np.average(x_axis, weights=x_weights)
    y_mean = np.average(y_axis, weights=y_weights)
    x_var = np.average((x_axis - x_mean)**2, weights=x_weights)
    y_var = np.average((y_axis - y_mean)**2, weights=y_weights)
    sigma = 2.0**-1.5 / gamma_lab
    # Five standard errors bound the mean. The 5% variance bound is 3.2
    # sampling standard deviations at N=8192; bilinear-grid bias is below
    # 0.4% for this resolution.
    mean_tolerance = 5.0 * sigma / math.sqrt(count)
    assert abs(x_mean - 0.5 * np.pi) < mean_tolerance
    assert abs(y_mean) < mean_tolerance
    assert np.isclose(x_var, sigma**2, rtol=0.05, atol=0.0)
    assert np.isclose(y_var, sigma**2, rtol=0.05, atol=0.0)


def _run_uniform_bz_simulation(
        gamma_boost, enable_diagnostic=True, write_dir=None):
    """Run one controlled tracer helix through the actual PIC cycle."""
    gamma_lab = 30.0
    u_lab = np.array([
        8.0, 0.0, math.sqrt(gamma_lab**2 - 1.0 - 8.0**2)
    ])
    Bz_lab = 500.0
    E_lab = np.zeros(3)
    cB_lab = np.array([0.0, 0.0, c * Bz_lab])

    if gamma_boost == 1.0:
        boost = None
        u_sim = u_lab
        gamma_inv_sim = 1.0 / gamma_lab
    else:
        boost = BoostConverter(gamma_boost)
        u_sim, gamma_inv_sim, _, _ = _boost_lab_event(
            gamma_boost, u_lab, E_lab, cB_lab
        )

    # Match corresponding worldline intervals: dt_lab/dt_sim=gamma/gamma'.
    dt_lab = 3.125e-15
    dt_ratio = gamma_lab * gamma_inv_sim
    dt_sim = dt_lab / dt_ratio
    if boost is None:
        input_dt = dt_sim
    else:
        # Simulation converts its input as a copropagating length/time.
        input_dt = dt_sim / (boost.gamma0 * (1.0 + boost.beta0))

    sim = Simulation(
        8, 1.0e-5, 4, 5.0e-5, 1, input_dt, zmin=-1.0e-5,
        n_order=-1, use_cuda=False,
        boundaries={'z': 'periodic', 'r': 'reflective'},
        gamma_boost=None if boost is None else boost.gamma0,
        verbose_level=0
    )
    particle = Particles(
        -e, m_e, 1.0, 1, -1.0e-6, 1.0e-6,
        1, 0.0, 1.0e-6, 1, sim.dt,
        grid_shape=sim.grid_shape, use_cuda=False,
        continuous_injection=False, is_tracer=True
    )
    particle.x[:] = 1.0e-7
    particle.y[:] = 0.0
    particle.z[:] = 0.0
    particle.ux[:] = u_sim[0]
    particle.uy[:] = u_sim[1]
    particle.uz[:] = u_sim[2]
    particle.inv_gamma[:] = gamma_inv_sim
    particle.w[:] = 1.0
    sim.ptcl.append(particle)
    sim.external_fields = [
        ExternalField(
            _constant_external_field, 'Bz', Bz_lab, 0.0,
            gamma_boost=None if boost is None else boost.gamma0
        )
    ]

    _, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)
    axes = (
        (0.2 * hbar * omega_c, 2.0 * hbar * omega_c, 24),
        (0.15, 0.40, 81),
        (-0.12, 0.12, 79),
    )
    if enable_diagnostic:
        particle.activate_synchrotron(
            *axes, gamma_cutoff=10.0, nSamples=256, boost=boost
        )
        sim.diags = [
            SynchrotronRadiationDiagnostic(
                period=8, species={"electrons": particle},
                comm=sim.comm, write_dir=str(write_dir)
            )
        ]

    set_random_seed(181)
    # Diagnostics run before radiation accumulation. Step 17 therefore writes
    # cumulative snapshots containing 0, 8, and 16 particle events.
    sim.step(17, show_progress=False)
    output = None
    if enable_diagnostic:
        timeseries = OpenPMDTimeSeries(str(Path(write_dir) / "hdf5"))
        snapshots = {}
        info = None
        for iteration in timeseries.iterations:
            snapshots[iteration], info = timeseries.get_field(
                "radiation_electrons", iteration=iteration,
                slice_across=None
            )
        output = SimpleNamespace(
            snapshots=snapshots, info=info,
            iterations=timeseries.iterations.copy(),
            times=timeseries.t.copy(), dt=sim.dt,
            axes=axes, omega_c=omega_c
        )
    state = np.array([
        particle.x[0], particle.y[0], particle.z[0], particle.ux[0],
        particle.uy[0], particle.uz[0], particle.inv_gamma[0]
    ])
    lab_momentum = np.array(get_particle_lab_frame(
        particle.ux[0], particle.uy[0], particle.uz[0],
        particle.inv_gamma[0], gamma_boost,
        math.sqrt(1.0 - gamma_boost**-2)
    )[:4])
    return output, state, lab_momentum


def _radiation_observables(data, info, omega_c):
    """Return integrated energy, centroids, widths, and one finite ROI."""
    cell_volume = info.dx * info.dy * info.dz
    spectrum = data.sum(axis=(0, 1)) * info.dx * info.dy
    x_weights = data.sum(axis=(1, 2))
    y_weights = data.sum(axis=(0, 2))
    x_mean = np.average(info.x, weights=x_weights)
    y_mean = np.average(info.y, weights=y_weights)
    x_width = np.sqrt(np.average(
        (info.x - x_mean)**2, weights=x_weights
    ))
    y_width = np.sqrt(np.average(
        (info.y - y_mean)**2, weights=y_weights
    ))
    energy_centroid = np.average(info.z, weights=spectrum)

    x_mask = (info.x >= 0.22) & (info.x <= 0.31)
    y_mask = (info.y >= -0.04) & (info.y <= 0.05)
    energy_mask = (
        (info.z >= 0.6 * hbar * omega_c)
        & (info.z <= 1.4 * hbar * omega_c)
    )
    roi = data[np.ix_(x_mask, y_mask, energy_mask)].sum() * cell_volume
    return {
        "total": data.sum() * cell_volume,
        "energy_centroid": energy_centroid,
        "x_mean": x_mean, "y_mean": y_mean,
        "x_width": x_width, "y_width": y_width,
        "roi": roi,
    }


def test_equivalent_lab_and_boosted_simulations(tmp_path):
    """Equivalent PIC runs write the same cumulative lab radiation."""
    lab, lab_state, lab_momentum = _run_uniform_bz_simulation(
        1.0, write_dir=tmp_path / "lab"
    )
    boosted, boosted_state, boosted_lab_momentum = \
        _run_uniform_bz_simulation(
            4.0, write_dir=tmp_path / "boosted"
        )
    _, plain_lab_state, _ = _run_uniform_bz_simulation(
        1.0, enable_diagnostic=False
    )
    _, plain_boosted_state, _ = _run_uniform_bz_simulation(
        4.0, enable_diagnostic=False
    )

    for output in (lab, boosted):
        assert np.array_equal(output.iterations, [0, 8, 16])
        assert np.allclose(
            output.times, output.iterations * output.dt,
            rtol=2.0e-15, atol=0.0
        )
        assert output.snapshots[16].shape == (81, 79, 24)
        assert not np.any(output.snapshots[0])
        scale = output.snapshots[16].max()
        assert scale > 0.0
        assert np.all(
            output.snapshots[16]
            >= output.snapshots[8] - 2.0e-12 * scale
        )
        total_8 = output.snapshots[8].sum()
        total_16 = output.snapshots[16].sum()
        assert np.isclose(
            total_16, 2.0 * total_8, rtol=2.0e-10, atol=0.0
        )

        energy_axis, theta_x_axis, theta_y_axis = output.axes
        assert np.allclose(
            output.info.x, np.linspace(*theta_x_axis),
            rtol=2.0e-15, atol=0.0
        )
        assert np.allclose(
            output.info.y, np.linspace(*theta_y_axis),
            rtol=2.0e-15, atol=0.0
        )
        assert np.allclose(
            output.info.z, np.linspace(*energy_axis),
            rtol=2.0e-15, atol=0.0
        )
        assert output.info.axes == {0: "x", 1: "y", 2: "z"}

    lab_data = lab.snapshots[16]
    boosted_data = boosted.snapshots[16]
    scale = np.max(lab_data)
    assert np.array_equal(lab_state, plain_lab_state)
    assert np.array_equal(boosted_state, plain_boosted_state)
    assert np.allclose(
        boosted_lab_momentum, lab_momentum,
        rtol=2.0e-13, atol=2.0e-13
    )
    assert np.allclose(boosted_data, lab_data, rtol=2.0e-10,
                       atol=2.0e-12 * scale)
    for axes in ((0, 1), (0, 2), (1, 2)):
        assert np.allclose(boosted_data.sum(axis=axes),
                           lab_data.sum(axis=axes), rtol=2.0e-10,
                           atol=2.0e-12 * scale)

    lab_observables = _radiation_observables(
        lab_data, lab.info, lab.omega_c
    )
    boosted_observables = _radiation_observables(
        boosted_data, boosted.info, boosted.omega_c
    )
    assert lab_observables["roi"] > 0.05 * lab_observables["total"]
    for name in (
            "total", "energy_centroid", "x_width", "y_width", "roi"):
        assert np.isclose(
            boosted_observables[name], lab_observables[name],
            rtol=2.0e-10, atol=0.0
        )
    for name in ("x_mean", "y_mean"):
        assert np.isclose(
            boosted_observables[name], lab_observables[name],
            rtol=2.0e-10, atol=2.0e-12
        )


def test_boosted_radiation_reaction_is_rejected():
    """The unsupported boosted recoil cannot be enabled silently."""
    particles = Particles(
        -e, m_e, 0.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0,
        1.0e-18, continuous_injection=False
    )
    axes = ((1.0e-18, 2.0e-16, 8),
            (-0.1, 0.1, 5), (-0.1, 0.1, 7))
    with pytest.raises(NotImplementedError, match="not supported"):
        particles.activate_synchrotron(
            *axes, radiation_reaction=True, nSamples=32,
            boost=BoostConverter(4.0)
        )
    assert particles.synchrotron_radiator is None

    # An identity BoostConverter retains the legacy unboosted behavior.
    particles.activate_synchrotron(
        *axes, radiation_reaction=True, nSamples=32,
        boost=BoostConverter(1.0)
    )
    assert particles.synchrotron_radiator.radiation_reaction


def test_unboosted_radiation_reaction_none_and_identity():
    """Identity boost preserves the legacy laboratory recoil update."""
    gamma = 60.0
    u = np.array([
        3.0, -2.0, math.sqrt(gamma**2 - 1.0 - 3.0**2 - 2.0**2)
    ])
    E = np.zeros(3)
    cB = np.array([0.0, c * 2.0e3, 0.0])
    dt = 1.0e-14
    power, omega_c = _power_and_critical_frequency(u, E, cB)
    energy_axis = (
        0.1 * hbar * omega_c, 3.0 * hbar * omega_c, 32
    )
    theta_x_axis = (-0.20, 0.30, 41)
    theta_y_axis = (-0.25, 0.20, 43)

    def run(boost):
        species = _dummy_species(dt, u, E, cB)
        radiator = SynchrotronRadiator(
            species, energy_axis, theta_x_axis, theta_y_axis,
            10.0, True, 20.0, 256, boost
        )
        set_random_seed(613)
        radiator.handle_radiation()
        state = np.array([
            species.ux[0], species.uy[0], species.uz[0],
            species.inv_gamma[0]
        ])
        return state, radiator.radiation_data

    no_boost_state, no_boost_data = run(None)
    identity_state, identity_data = run(BoostConverter(1.0))

    # For the legacy collinear recoil, Delta|u| = P dt / (m c^2).
    recoil = power * dt / (m_e * c**2)
    expected_u = u - recoil * u / np.linalg.norm(u)
    expected_state = np.r_[
        expected_u, 1.0 / math.sqrt(1.0 + np.dot(expected_u, expected_u))
    ]
    assert no_boost_data.sum() > 0.0
    assert np.allclose(
        no_boost_state, expected_state, rtol=2.0e-12, atol=5.0e-13
    )
    assert np.array_equal(identity_state, no_boost_state)
    assert np.array_equal(identity_data, no_boost_data)


def test_cuda_seed_entropy_interval(monkeypatch):
    """CUDA seeds are drawn from much more than the former 8-bit interval."""
    calls = []

    def randrange(low, high):
        calls.append((low, high))
        return 17

    from fbpic.particles.elementary_process.synchrotron import radiator
    fake_random = SimpleNamespace(randrange=randrange)
    monkeypatch.setattr(radiator, "_synchrotron_random", fake_random)
    assert _get_cuda_rng_seed() == 17
    assert calls == [(0, 1 << 63)]


def test_synchrotron_rng_does_not_advance_particle_streams():
    """Angle sampling is isolated from NumPy and Python particle streams."""
    gamma = 20.0
    u = np.array([0.0, 0.0, math.sqrt(gamma**2 - 1.0)])
    E = np.zeros(3)
    cB = np.array([0.0, c * 1.0e3, 0.0])
    _, omega_c = _power_and_critical_frequency(u, E, cB)
    species = _dummy_species(1.0e-18, u, E, cB)
    radiator = _make_radiator(
        species, (0.2 * hbar * omega_c, 2.0 * hbar * omega_c, 12),
        (-0.1, 0.1, 13), (-0.12, 0.12, 15)
    )

    set_random_seed(271)
    expected_numpy = np.random.random()
    set_random_seed(271)
    python_state = random.getstate()
    radiator.handle_radiation()
    _get_cuda_rng_seed()
    assert np.random.random() == expected_numpy
    assert random.getstate() == python_state


def _get_ultrarelativistic_spectrum(use_cuda, boosted):
    """Evaluate a cancellation-sensitive gamma=1e8 transverse-E event."""
    gamma_lab = 1.0e8
    u_lab = np.array([
        0.0, 0.0, math.sqrt(gamma_lab**2 - 1.0)
    ])
    E_lab = np.array([1.0e12, 0.0, 0.0])
    cB_lab = np.zeros(3)
    power, omega_c = _power_and_critical_frequency(
        u_lab, E_lab, cB_lab
    )
    boost = BoostConverter(4.0)
    u_sim, gamma_inv_sim, E_sim, cB_sim = _boost_lab_event(
        4.0, u_lab, E_lab, cB_lab
    )
    dt_lab = 1.0e-18
    if boosted:
        dt_ratio = gamma_lab * gamma_inv_sim
        u_input, E_input, cB_input = u_sim, E_sim, cB_sim
        dt_input, boost_input = dt_lab / dt_ratio, boost
    else:
        u_input, E_input, cB_input = u_lab, E_lab, cB_lab
        dt_input, boost_input = dt_lab, None

    xi = np.linspace(0.2, 2.0, 24)
    energy = (
        xi[0] * hbar * omega_c, xi[-1] * hbar * omega_c, xi.size
    )
    theta_x = (-2.0e-5, 2.0e-5, 33)
    theta_y = (-1.8e-5, 2.2e-5, 35)
    count = 32
    species = _dummy_species(
        dt_input, u_input, E_input, cB_input,
        count=count, use_cuda=use_cuda
    )
    radiator = _make_radiator(
        species, energy, theta_x, theta_y, 10.0, boost_input, 256
    )
    set_random_seed(717)
    radiator.handle_radiation()
    if use_cuda:
        cupy.cuda.runtime.deviceSynchronize()
        radiator.receive_from_gpu()
    spectrum = radiator.radiation_data.sum(axis=(0, 1))
    spectrum *= radiator.d_theta_x * radiator.d_theta_y
    profile = np.array([
        9.0 * math.sqrt(3.0) / (8.0 * np.pi) * value
        * quad(lambda point: kv(5.0 / 3.0, point), value, np.inf)[0]
        for value in xi
    ])
    expected = count * power * dt_lab / (hbar * omega_c) * profile
    return spectrum, expected, radiator.omega_ax * hbar


def test_ultrarelativistic_power_stability_cpu():
    """Stable proper acceleration matches a rest-frame power oracle."""
    lab, expected, _ = _get_ultrarelativistic_spectrum(False, False)
    boosted, _, _ = _get_ultrarelativistic_spectrum(False, True)
    assert np.isfinite(lab).all()
    assert lab.sum() > 0.0
    assert np.allclose(lab, expected, rtol=1.0e-2, atol=0.0)
    assert np.allclose(boosted, lab, rtol=2.0e-11, atol=0.0)


@pytest.mark.skipif(not cuda_installed, reason="CUDA hardware is unavailable")
def test_ultrarelativistic_power_cpu_cuda():
    """CPU and CUDA agree on energy-resolved stable power and total energy."""
    cpu, _, energy = _get_ultrarelativistic_spectrum(False, True)
    gpu, _, _ = _get_ultrarelativistic_spectrum(True, True)
    assert np.allclose(gpu, cpu, rtol=2.0e-11, atol=0.0)
    assert np.isclose(trapezoid(gpu, energy), trapezoid(cpu, energy),
                      rtol=2.0e-11, atol=0.0)


@pytest.mark.skipif(not cuda_installed, reason="CUDA hardware is unavailable")
def test_single_event_lab_boost_equivalence_cuda():
    """H100 uses lab physics and does not mutate particle momentum."""
    gamma_lab = 100.0
    u_lab = np.array([0.0, 0.0, math.sqrt(gamma_lab**2 - 1.0)])
    E_lab = np.zeros(3)
    cB_lab = np.array([0.0, c * 1.0e3, 0.0])
    boost = BoostConverter(4.0)
    u_sim, inv_gamma_sim, E_sim, cB_sim = _boost_lab_event(
        4.0, u_lab, E_lab, cB_lab
    )
    dt_ratio = gamma_lab * inv_gamma_sim
    _, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)
    energy = (0.1 * hbar * omega_c, 3.0 * hbar * omega_c, 48)
    theta_x = (-0.05, 0.05, 33)
    theta_y = (-0.04, 0.06, 35)

    def run(u, E, cB, dt, boost_arg):
        species = _dummy_species(
            dt, u, E, cB, count=32, use_cuda=True
        )
        before = [
            value.copy() for value in
            (species.ux, species.uy, species.uz, species.inv_gamma)
        ]
        set_random_seed(91)
        radiator = _make_radiator(
            species, energy, theta_x, theta_y, 50.0, boost_arg, 256
        )
        assert radiator.use_cuda
        radiator.handle_radiation()
        cupy.cuda.runtime.deviceSynchronize()
        for old, new in zip(
                before,
                (species.ux, species.uy, species.uz, species.inv_gamma)):
            assert cupy.array_equal(old, new)
        radiator.receive_from_gpu()
        return radiator.radiation_data

    lab = run(u_lab, E_lab, cB_lab, 1.0e-18, None)
    boosted = run(
        u_sim, E_sim, cB_sim, 1.0e-18 / dt_ratio, boost
    )
    repeated = run(
        u_sim, E_sim, cB_sim, 1.0e-18 / dt_ratio, boost
    )
    assert np.allclose(
        boosted, repeated, rtol=5.0e-15,
        atol=5.0e-15 * max(boosted.max(), repeated.max())
    )
    assert np.allclose(
        boosted, lab, rtol=2.0e-10, atol=1.0e-12 * lab.max()
    )


def _angular_histogram_moments(data, theta_x, theta_y):
    """Return the two means and variances of an angular histogram."""
    angular = data.sum(axis=2)
    normalization = angular.sum()
    x_mean = np.sum(angular * theta_x[:, None]) / normalization
    y_mean = np.sum(angular * theta_y[None, :]) / normalization
    x_variance = np.sum(
        angular * (theta_x[:, None] - x_mean)**2
    ) / normalization
    y_variance = np.sum(
        angular * (theta_y[None, :] - y_mean)**2
    ) / normalization
    return np.array([x_mean, y_mean, x_variance, y_variance])


@pytest.mark.skipif(not cuda_installed, reason="CUDA hardware is unavailable")
def test_cuda_cpu_identity_and_boosted_population_parity():
    """CPU and CUDA agree on lab spectra and angular statistics."""
    gamma_lab = 200.0
    direction = np.array([0.18, -0.11, 1.0])
    direction /= np.linalg.norm(direction)
    u_lab = direction * math.sqrt(gamma_lab**2 - 1.0)
    E_lab = np.array([2.0e11, -3.0e11, 0.7e11])
    cB_lab = np.array([-1.4e11, 2.2e11, 0.4e11])
    _, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)

    energy = (0.08 * hbar * omega_c, 3.2 * hbar * omega_c, 24)
    theta_x = (0.14, 0.22, 121)
    theta_y = (-0.15, -0.065, 103)
    x_axis = np.linspace(*theta_x)
    y_axis = np.linspace(*theta_y)
    count = 32768
    dt_lab = 2.0e-18
    outputs = {}

    for gamma_boost in (1.0, 4.0):
        if gamma_boost == 1.0:
            u_input = u_lab
            inv_gamma_input = 1.0 / gamma_lab
            E_input, cB_input = E_lab, cB_lab
            dt_input = dt_lab
        else:
            u_input, inv_gamma_input, E_input, cB_input = \
                _boost_lab_event(
                    gamma_boost, u_lab, E_lab, cB_lab
                )
            assert 1.0 / inv_gamma_input < 100.0 < gamma_lab
            dt_input = dt_lab / (gamma_lab * inv_gamma_input)

        for use_cuda in (False, True):
            species = _dummy_species(
                dt_input, u_input, E_input, cB_input,
                count=count, use_cuda=use_cuda
            )
            set_random_seed(9401)
            radiator = _make_radiator(
                species, energy, theta_x, theta_y, 100.0,
                BoostConverter(gamma_boost), 256
            )
            radiator.handle_radiation()
            if use_cuda:
                cupy.cuda.runtime.deviceSynchronize()
                radiator.receive_from_gpu()

            data = np.asarray(radiator.radiation_data)
            spectrum = data.sum(axis=(0, 1)) \
                * radiator.d_theta_x * radiator.d_theta_y
            total = trapezoid(spectrum, np.linspace(*energy))
            moments = _angular_histogram_moments(
                data, x_axis, y_axis
            )
            outputs[gamma_boost, use_cuda] = (
                data, spectrum, total, moments
            )
            assert total > 0.0

    sigma = 2.0**-1.5 / gamma_lab
    expected_mean = np.arctan2(u_lab[:2], u_lab[2])
    mean_tolerance = 5.0 * sigma / math.sqrt(count)
    backend_mean_tolerance = math.sqrt(2.0) * mean_tolerance

    for gamma_boost in (1.0, 4.0):
        cpu = outputs[gamma_boost, False]
        gpu = outputs[gamma_boost, True]
        assert np.allclose(
            gpu[1], cpu[1], rtol=5.0e-12,
            atol=5.0e-14 * cpu[1].max()
        )
        assert np.isclose(gpu[2], cpu[2], rtol=5.0e-12, atol=0.0)
        assert np.all(
            np.abs(gpu[3][:2] - cpu[3][:2])
            < backend_mean_tolerance
        )
        assert np.allclose(
            gpu[3][2:], cpu[3][2:], rtol=0.06, atol=0.0
        )
        for output in (cpu, gpu):
            assert np.all(
                np.abs(output[3][:2] - expected_mean) < mean_tolerance
            )
            assert np.allclose(
                output[3][2:], [sigma**2, sigma**2],
                rtol=0.08, atol=0.0
            )

    # Resetting the seed supplies the same angular realization in the two
    # frames. This enables an energy-angle-resolved comparison in addition
    # to the backend comparison, whose independent generators are assessed
    # through the five-standard-error bounds above. The 6% variance bound is
    # five sampling standard deviations; 8% also covers linear-grid bias.
    for use_cuda in (False, True):
        lab = outputs[1.0, use_cuda][0]
        boosted = outputs[4.0, use_cuda][0]
        assert np.allclose(
            boosted, lab, rtol=2.0e-10, atol=1.0e-12 * lab.max()
        )


@pytest.mark.skipif(not cuda_installed, reason="CUDA hardware is unavailable")
def test_cuda_rng_state_persists_and_live_reseed_replays():
    """A live radiator retains CUDA streams and restarts them on reseed."""
    gamma_lab = 100.0
    direction = np.array([0.05, -0.03, 1.0])
    direction /= np.linalg.norm(direction)
    u_lab = direction * math.sqrt(gamma_lab**2 - 1.0)
    E_lab = np.zeros(3)
    cB_lab = np.array([0.0, c * 1.0e3, 0.0])
    boost = BoostConverter(4.0)
    u_sim, inv_gamma_sim, E_sim, cB_sim = _boost_lab_event(
        4.0, u_lab, E_lab, cB_lab
    )
    _, omega_c = _power_and_critical_frequency(u_lab, E_lab, cB_lab)
    dt_ratio = gamma_lab * inv_gamma_sim
    energy = (0.2 * hbar * omega_c, 2.0 * hbar * omega_c, 12)
    theta_x = (-0.01, 0.11, 41)
    theta_y = (-0.09, 0.03, 43)

    species = _dummy_species(
        1.0e-18 / dt_ratio, u_sim, E_sim, cB_sim,
        count=1024, use_cuda=True
    )
    set_random_seed(20260803)
    radiator = _make_radiator(
        species, energy, theta_x, theta_y,
        50.0, boost, 128
    )
    assert radiator.rng_states_batch is None

    radiator.handle_radiation()
    cupy.cuda.runtime.deviceSynchronize()
    first_total = _to_numpy(radiator.radiation_data).copy()
    first_states = radiator.rng_states_batch
    first_state_size = radiator.rng_states_size

    radiator.handle_radiation()
    cupy.cuda.runtime.deviceSynchronize()
    second_total = _to_numpy(radiator.radiation_data).copy()
    first_increment = first_total
    second_increment = second_total - first_total

    # A fixed population continues the existing xoroshiro streams rather
    # than allocating and reseeding them at every diagnostic event.
    assert radiator.rng_states_batch is first_states
    assert radiator.rng_states_size == first_state_size

    # Reseeding a live simulation invalidates persistent CUDA streams lazily:
    # the state object changes at the next radiation event, not before it.
    set_random_seed(20260803)
    assert radiator.rng_states_batch is first_states
    radiator.handle_radiation()
    cupy.cuda.runtime.deviceSynchronize()
    third_total = _to_numpy(radiator.radiation_data).copy()
    repeated_first = third_total - second_total
    reseeded_states = radiator.rng_states_batch
    assert reseeded_states is not first_states
    assert radiator.rng_states_size == first_state_size

    radiator.handle_radiation()
    cupy.cuda.runtime.deviceSynchronize()
    fourth_total = _to_numpy(radiator.radiation_data).copy()
    repeated_second = fourth_total - third_total
    assert radiator.rng_states_batch is reseeded_states
    assert radiator.rng_states_size == first_state_size

    scale = max(first_increment.max(), second_increment.max())
    assert scale > 0.0

    # Resetting FBPIC's seed replays both increments on the same radiator.
    assert np.allclose(
        repeated_first, first_increment,
        rtol=5.0e-15, atol=5.0e-15 * scale
    )
    assert np.allclose(
        repeated_second, second_increment,
        rtol=5.0e-15, atol=5.0e-15 * scale
    )

    # Successive increments have the same energy spectrum but independent
    # angular samples. Equality here would expose per-step state reset.
    assert np.allclose(
        second_increment.sum(axis=(0, 1)),
        first_increment.sum(axis=(0, 1)), rtol=2.0e-13, atol=0.0
    )
    assert not np.array_equal(second_increment, first_increment)


def test_openpmd_unequal_lab_axes_metadata(tmp_path):
    """The output records distinct lab theta_x/theta_y origins and spacings."""
    radiator = SimpleNamespace(
        use_cuda=False, dt=1.0e-18,
        N_theta_x=5, N_theta_y=7, N_omega=4,
        d_theta_x=0.05, d_theta_y=0.02,
        d_omega=(8.0e-16 - 2.0e-16) / (3.0 * hbar),
        theta_x_min=-0.12, theta_y_min=0.03,
        omega_min=2.0e-16 / hbar,
        radiation_data=np.arange(140, dtype=np.float64).reshape(5, 7, 4)
    )
    species = SimpleNamespace(synchrotron_radiator=radiator)
    comm = SimpleNamespace(rank=0, size=1)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1, species={"electrons": species}, comm=comm,
        write_dir=str(tmp_path)
    )
    first = radiator.radiation_data.copy()
    diagnostic.write_hdf5(3)
    increment = np.full_like(first, 17.0)
    radiator.radiation_data += increment
    second = radiator.radiation_data.copy()
    diagnostic.write_hdf5(6)

    path = tmp_path / "hdf5" / "data00000003.h5"
    with h5py.File(path, "r") as output:
        fields = output["/data/3/fields"]
        assert set(fields.keys()) == {"radiation_electrons"}
        dataset = fields["radiation_electrons"]
        assert dataset.shape == (5, 7, 4)
        assert np.allclose(
            dataset.attrs["gridGlobalOffset"], [-0.12, 0.03, 2.0e-16],
            rtol=2.0e-15, atol=0.0
        )
        assert np.allclose(
            dataset.attrs["gridSpacing"], [0.05, 0.02, 2.0e-16],
            rtol=2.0e-15, atol=0.0
        )
        assert np.array_equal(dataset[:], first)
        assert np.array_equal(
            dataset.attrs["unitDimension"], np.zeros(7)
        )
        assert dataset.attrs["unitSI"] == 1.0
        assert dataset.attrs["gridUnitSI"] == 1.0
        assert list(dataset.attrs["axisLabels"]) == [b"x", b"y", b"z"]
        assert dataset.attrs["dataOrder"] == np.bytes_("C")
        assert dataset.attrs["geometry"] == np.bytes_("cartesian")

    timeseries = OpenPMDTimeSeries(str(tmp_path / "hdf5"))
    assert np.array_equal(timeseries.iterations, [3, 6])
    assert np.allclose(
        timeseries.t, [3.0e-18, 6.0e-18], rtol=2.0e-15, atol=0.0
    )
    loaded_first, info = timeseries.get_field(
        "radiation_electrons", iteration=3, slice_across=None
    )
    loaded_second, second_info = timeseries.get_field(
        "radiation_electrons", iteration=6, slice_across=None
    )
    assert np.array_equal(loaded_first, first)
    assert np.array_equal(loaded_second, second)
    assert np.array_equal(loaded_second - loaded_first, increment)
    assert info.axes == {0: "x", 1: "y", 2: "z"}
    assert np.allclose(
        info.x, [-0.12, -0.07, -0.02, 0.03, 0.08],
        rtol=2.0e-15, atol=0.0
    )
    assert np.allclose(
        info.y, [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15],
        rtol=2.0e-15, atol=0.0
    )
    assert np.allclose(
        info.z, [2.0e-16, 4.0e-16, 6.0e-16, 8.0e-16],
        rtol=2.0e-15, atol=0.0
    )
    assert second_info.component_attrs["unitSI"] == 1.0
    assert second_info.component_attrs["gridUnitSI"] == 1.0
    assert second_info.component_attrs["axisLabels"] == ["x", "y", "z"]


def _run_mpi_openpmd_worker(write_dir):
    """Accumulate a partitioned physical population and reduce it."""
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    global_count = 4096
    lower = global_count * world.rank // world.size
    upper = global_count * (world.rank + 1) // world.size
    particle_index = np.arange(lower, upper, dtype=np.float64)

    # Each global macroparticle has distinct momentum, field, and weight.
    gamma_lab = 170.0 + 25.0 * np.sin(0.013 * particle_index)
    nx = 0.045 + 0.012 * np.sin(0.019 * particle_index)
    ny = -0.022 + 0.009 * np.cos(0.017 * particle_index)
    nz = np.sqrt(1.0 - nx**2 - ny**2)
    u_abs = np.sqrt(gamma_lab**2 - 1.0)
    ux_lab = u_abs * nx
    uy_lab = u_abs * ny
    uz_lab = u_abs * nz

    Ex_lab = 2.0e11 * (
        1.0 + 0.1 * np.sin(0.011 * particle_index)
    )
    Ey_lab = -3.0e11 * (
        1.0 + 0.1 * np.cos(0.007 * particle_index)
    )
    Ez_lab = np.full_like(particle_index, 0.7e11)
    cBx_lab = np.full_like(particle_index, -1.4e11)
    cBy_lab = 2.2e11 * (
        1.0 + 0.08 * np.sin(0.005 * particle_index)
    )
    cBz_lab = np.full_like(particle_index, 0.4e11)
    weight = 0.5 + (particle_index % 17.0) / 17.0

    boost = BoostConverter(4.0)
    gamma_boost, beta_boost = boost.gamma0, boost.beta0
    gamma_sim = gamma_boost * (
        gamma_lab - beta_boost * uz_lab
    )
    uz_sim = gamma_boost * (uz_lab - beta_boost * gamma_lab)
    Ex_sim = gamma_boost * (Ex_lab - beta_boost * cBy_lab)
    Ey_sim = gamma_boost * (Ey_lab + beta_boost * cBx_lab)
    cBx_sim = gamma_boost * (cBx_lab + beta_boost * Ey_lab)
    cBy_sim = gamma_boost * (cBy_lab - beta_boost * Ex_lab)
    assert np.all(gamma_lab > 100.0)
    assert np.all(gamma_sim < 100.0)

    def make_species():
        species = SimpleNamespace(
            use_cuda=False, dt=2.0e-18, Ntot=upper - lower,
            ux=ux_lab, uy=uy_lab, uz=uz_sim,
            Ex=Ex_sim, Ey=Ey_sim, Ez=Ez_lab,
            Bx=cBx_sim / c, By=cBy_sim / c, Bz=cBz_lab / c,
            w=weight, inv_gamma=1.0 / gamma_sim
        )
        species.synchrotron_radiator = _make_radiator(
            species, (1.0e-17, 5.0e-15, 18),
            (-0.03, 0.10, 31), (-0.09, 0.05, 35),
            100.0, boost, 256
        )
        return species

    aligned_species = make_species()
    normal_species = make_species()

    # Compare the same global stochastic realization across decompositions.
    # set_random_seed normally adds the MPI rank, so cancel that offset and
    # advance past the two Gaussian samples of earlier contiguous particles.
    set_random_seed(20260803 - world.rank)
    for _ in range(2 * lower):
        _synchrotron_random.gauss(0.0, 1.0)
    aligned_species.synchrotron_radiator.handle_radiation()

    # Exercise the public project convention without cancelling the rank
    # offset. This produces independent, reproducible streams on each rank.
    set_random_seed(20260803)
    normal_species.synchrotron_radiator.handle_radiation()

    comm = SimpleNamespace(rank=world.rank, size=world.size)
    diagnostic = SynchrotronRadiationDiagnostic(
        period=1,
        species={"aligned": aligned_species, "normal": normal_species},
        comm=comm,
        write_dir=write_dir
    )
    diagnostic.write_hdf5(5)
    world.Barrier()


def test_mpi_reduction_and_openpmd_output(tmp_path):
    """A physical boosted population is independent of MPI partitioning."""
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
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    if current_pythonpath:
        env["PYTHONPATH"] = repo_root + os.pathsep + current_pythonpath
    else:
        env["PYTHONPATH"] = repo_root
    output_dirs = {}
    for rank_count in (1, 2):
        output_dir = tmp_path / f"ranks-{rank_count}"
        output_dirs[rank_count] = output_dir
        command = [
            mpi_exec, "-n", str(rank_count), sys.executable,
            str(test_file), "--mpi-radiation-worker", str(output_dir)
        ]
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, (
            f"{rank_count}-rank MPI radiation worker failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    arrays = {"aligned": {}, "normal": {}}
    attributes = {}
    for rank_count, output_dir in output_dirs.items():
        path = output_dir / "hdf5" / "data00000005.h5"
        with h5py.File(path, "r") as output:
            fields = output["/data/5/fields"]
            assert set(fields.keys()) == {
                "radiation_aligned", "radiation_normal"
            }
            for stream in arrays:
                dataset = fields[f"radiation_{stream}"]
                arrays[stream][rank_count] = dataset[:]
                attributes[rank_count, stream] = {
                    key: dataset.attrs[key]
                    for key in (
                        "gridGlobalOffset", "gridSpacing", "axisLabels"
                    )
                }

    aligned_serial = arrays["aligned"][1]
    aligned_distributed = arrays["aligned"][2]
    normal_serial = arrays["normal"][1]
    normal_distributed = arrays["normal"][2]
    for data in (
            aligned_serial, aligned_distributed,
            normal_serial, normal_distributed):
        assert data.shape == (31, 35, 18)
        assert np.isfinite(data).all()
        assert data.sum() > 0.0
    assert np.array_equal(normal_serial, aligned_serial)
    assert np.allclose(
        aligned_distributed, aligned_serial, rtol=5.0e-13,
        atol=5.0e-15 * aligned_serial.max()
    )
    assert not np.array_equal(normal_distributed, aligned_distributed)

    expected_origin = [-0.03, -0.09, 1.0e-17]
    expected_spacing = [
        0.13 / 30.0, 0.14 / 34.0,
        (5.0e-15 - 1.0e-17) / 17.0
    ]
    for attrs in attributes.values():
        assert np.allclose(
            attrs["gridGlobalOffset"], expected_origin,
            rtol=2.0e-15, atol=0.0
        )
        assert np.allclose(
            attrs["gridSpacing"], expected_spacing,
            rtol=2.0e-15, atol=0.0
        )
        assert list(attrs["axisLabels"]) == [b"x", b"y", b"z"]

    x_axis = np.linspace(-0.03, 0.10, 31)
    y_axis = np.linspace(-0.09, 0.05, 35)
    energy_axis = np.linspace(1.0e-17, 5.0e-15, 18)
    cell_volume = np.prod(expected_spacing)
    moments = {}

    for stream in arrays:
        serial = arrays[stream][1]
        distributed = arrays[stream][2]
        serial_spectrum = serial.sum(axis=(0, 1))
        distributed_spectrum = distributed.sum(axis=(0, 1))
        assert np.allclose(
            distributed_spectrum, serial_spectrum, rtol=5.0e-13,
            atol=5.0e-15 * serial_spectrum.max()
        )
        serial_total = serial.sum() * cell_volume
        distributed_total = distributed.sum() * cell_volume
        assert np.isclose(
            distributed_total, serial_total, rtol=5.0e-13, atol=0.0
        )
        serial_centroid = np.average(
            energy_axis, weights=serial_spectrum
        )
        distributed_centroid = np.average(
            energy_axis, weights=distributed_spectrum
        )
        assert np.isclose(
            distributed_centroid, serial_centroid,
            rtol=5.0e-13, atol=1.0e-30
        )
        moments[stream, 1] = _angular_histogram_moments(
            serial, x_axis, y_axis
        )
        moments[stream, 2] = _angular_histogram_moments(
            distributed, x_axis, y_axis
        )

    assert np.allclose(
        moments["aligned", 2], moments["aligned", 1],
        rtol=5.0e-13, atol=1.0e-15
    )

    # For the normal rank-offset streams, use six-standard-error bounds.
    # N_eff=1024 conservatively allows a factor-four reduction from the 4096
    # physical particles for their nonuniform weights and emission strengths.
    effective_count = 1024.0
    normal_serial_moments = moments["normal", 1]
    normal_distributed_moments = moments["normal", 2]
    for axis in range(2):
        variance = max(
            normal_serial_moments[axis + 2],
            normal_distributed_moments[axis + 2]
        )
        mean_standard_error = math.sqrt(
            2.0 * variance / effective_count
        )
        assert abs(
            normal_distributed_moments[axis]
            - normal_serial_moments[axis]
        ) < 6.0 * mean_standard_error
        variance_standard_error = 2.0 * variance / math.sqrt(
            effective_count
        )
        assert abs(
            normal_distributed_moments[axis + 2]
            - normal_serial_moments[axis + 2]
        ) < 6.0 * variance_standard_error


if __name__ == "__main__" and len(sys.argv) == 3:
    if sys.argv[1] == "--mpi-radiation-worker":
        _run_mpi_openpmd_worker(sys.argv[2])

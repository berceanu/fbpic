# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""Observer-frame synchrotron-radiation activation and spectral tables."""

from functools import lru_cache
import math
import warnings

import numpy as np
from scipy.constants import e, m_e
from scipy.integrate import IntegrationWarning, quad
from scipy.special import gamma as gamma_function, kv

from fbpic.utils.printing import catch_gpu_memory_error


_SYNCHROTRON_NORMALIZATION = 9.0 * math.sqrt(3.0) / (8.0 * math.pi)
_SOFT_SPECTRUM_COEFFICIENT = (
    _SYNCHROTRON_NORMALIZATION
    * 2.0**(2.0 / 3.0) * gamma_function(2.0 / 3.0)
)


def _spectral_profile(scaled_energy):
    """Return the normalized classical synchrotron power profile ``S(x)``."""
    if scaled_energy <= 0.0:
        return 0.0
    if scaled_energy < 1.0e-4:
        return _SOFT_SPECTRUM_COEFFICIENT * scaled_energy**(1.0 / 3.0)
    integral = quad(
        lambda value: kv(5.0 / 3.0, value),
        scaled_energy, np.inf,
    )[0]
    return _SYNCHROTRON_NORMALIZATION * scaled_energy * integral


def _spectral_tail_fraction(x_max):
    """Return the analytic-profile energy fraction above ``x_max``.

    Reversing the order of the two synchrotron-profile integrals gives a
    single, well-conditioned quadrature for the omitted tail.
    """
    integral = quad(
        lambda value: (value**2 - x_max**2) * kv(5.0 / 3.0, value),
        x_max, np.inf,
    )[0]
    fraction = 0.5 * _SYNCHROTRON_NORMALIZATION * integral
    return float(np.clip(fraction, 0.0, 1.0))


@lru_cache(maxsize=8)
def _cached_spectral_cdf(x_max, n_samples):
    """Build and cache a low-energy-resolving synchrotron CDF table."""
    x_max = float(x_max)
    n_samples = int(n_samples)
    if not math.isfinite(x_max) or x_max <= 0.0:
        raise ValueError("`x_max` must be a finite positive number.")
    if n_samples < 16:
        raise ValueError("`n_samples` must be at least 16.")

    # A logarithmic positive grid resolves S(x) ~ x**(1/3) without spending
    # most entries in the exponentially small high-energy tail. The origin
    # remains explicit because both the profile and its CDF vanish there.
    x_min = min(1.0e-6, x_max * 1.0e-4)
    positive_x = np.geomspace(x_min, x_max, n_samples - 1)
    spectral_x = np.concatenate(([0.0], positive_x))
    profile = np.empty_like(spectral_x)
    profile[0] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=IntegrationWarning)
        profile[1:] = np.array([
            _spectral_profile(value) for value in positive_x
        ])
        tail_fraction = _spectral_tail_fraction(x_max)

    cdf = np.zeros_like(spectral_x)
    cdf[1:] = np.cumsum(
        0.5 * (profile[:-1] + profile[1:]) * np.diff(spectral_x)
    )
    retained_fraction = 1.0 - tail_fraction
    if not cdf[-1] > 0.0:
        raise RuntimeError("Could not normalize the synchrotron CDF table.")
    # Use the exact retained integral for normalization. The unrepresented
    # tail is intentionally not folded back into the sampled spectrum.
    cdf *= retained_fraction / cdf[-1]
    cdf[0] = 0.0
    cdf[-1] = retained_fraction
    spectral_x.setflags(write=False)
    cdf.setflags(write=False)
    return spectral_x, cdf, tail_fraction


class SynchrotronRadiator(object):
    """Own the passive observer-frame accumulator for one lepton species."""

    def __init__(self, radiating_species, gamma_cutoff=10.0, x_max=20.0,
                 n_samples=2048, boost=None):
        charge = getattr(radiating_species, "q", None)
        mass = getattr(radiating_species, "m", None)
        if charge is None or mass is None:
            raise TypeError(
                "Synchrotron radiation requires a species with charge and "
                "mass attributes.")
        if not math.isclose(abs(float(charge)), e, rel_tol=1.0e-12,
                            abs_tol=0.0) or not math.isclose(
                                float(mass), m_e, rel_tol=1.0e-12,
                                abs_tol=0.0):
            raise ValueError(
                "The observer synchrotron diagnostic supports only "
                "electrons and positrons (|q| = e and m = m_e).")

        gamma_cutoff = float(gamma_cutoff)
        if not math.isfinite(gamma_cutoff) or gamma_cutoff <= 1.0:
            raise ValueError(
                "`gamma_cutoff` must be greater than one; the local "
                "synchrotron closure is relativistic.")

        self.use_cuda = radiating_species.use_cuda
        self.eon = radiating_species
        self.dt = radiating_species.dt
        self.gamma_cutoff = gamma_cutoff
        self.gamma_cutoff_inv = 1.0 / gamma_cutoff
        if boost is None:
            self.gamma_boost = 1.0
            self.beta_boost = 0.0
        else:
            self.gamma_boost = boost.gamma0
            self.beta_boost = boost.beta0

        table = _cached_spectral_cdf(float(x_max), int(n_samples))
        self.spectral_x = table[0].copy()
        self.spectral_cdf = table[1].copy()
        self.spectral_truncated_fraction = table[2]
        self.observer_accumulator = None

    def configure_observer_diagnostic(self, **configuration):
        """Configure the independently selectable observer products."""
        if self.observer_accumulator is not None:
            raise RuntimeError(
                "Only one synchrotron diagnostic may configure a species "
                "at a time.")
        from .observer import ObserverFrameRadiationAccumulator
        configuration.setdefault("gamma_boost", self.gamma_boost)
        configuration.setdefault("beta_boost", self.beta_boost)
        configuration.setdefault("gamma_cutoff", self.gamma_cutoff)
        configuration.setdefault(
            "spectral_truncated_fraction",
            self.spectral_truncated_fraction,
        )
        self.observer_accumulator = ObserverFrameRadiationAccumulator(
            self.eon, self.dt, self.spectral_x, self.spectral_cdf,
            **configuration
        )
        if self.use_cuda:
            self.observer_accumulator.send_to_gpu()
        return self.observer_accumulator

    @property
    def has_observer_diagnostic(self):
        """Whether the bounded pusher/radiation coupling path is active."""
        return self.observer_accumulator is not None

    def pusher_endpoint_buffer(self):
        """Return the fixed-capacity lower-endpoint coupling buffer."""
        if self.observer_accumulator is None:
            return None
        return self.observer_accumulator.pusher_endpoint_buffer

    @catch_gpu_memory_error
    def accumulate_pusher_batch(
            self, particle_slice, batch_count, simulation_time, event_index):
        """Consume one batch immediately after its momentum push."""
        if self.observer_accumulator is None:
            return
        self.observer_accumulator.accumulate_impulse_batch(
            self.observer_accumulator.pusher_endpoint_buffer,
            particle_slice, int(batch_count), float(simulation_time),
            int(event_index))

    def complete_momentum_push(self, simulation_time, event_index):
        """Mark one integer-centered pusher event complete exactly once."""
        if self.observer_accumulator is None:
            return
        self.observer_accumulator.complete_impulse(
            float(simulation_time), int(event_index))

    def begin_momentum_push(self):
        """Reject the removed full-species endpoint snapshot interface."""
        raise RuntimeError(
            "The full-species momentum snapshot interface was removed. "
            "Use Particles.push_p, which streams bounded endpoint batches "
            "directly from the pusher into the radiation accumulator.")

    def end_momentum_push(self, lower_momentum, simulation_time):
        """Accept explicitly supplied endpoints only for compatibility tests.

        Production stepping never calls this path and never allocates complete
        endpoint copies. Callers that already own lower endpoints may still
        feed them to the accumulator without changing the physical event.
        """
        if self.observer_accumulator is None:
            return
        event_index = int(round(float(simulation_time) / self.dt))
        self.observer_accumulator.accumulate_impulse(
            lower_momentum, float(simulation_time), event_index=event_index)

    def handle_radiation(self, simulation_time=0.0):
        """Reject the former mixed-time gathered-field event entry point.

        Radiation is now accumulated automatically around Particles.push_p.
        Keeping this method with an explicit error makes stale integrations
        fail loudly instead of silently reverting to the old field-derived
        acceleration model.
        """
        raise RuntimeError(
            "Synchrotron radiation is accumulated from completed momentum "
            "pushes; handle_radiation no longer accepts gathered-field "
            "events.")

    def send_to_gpu(self):
        """Move configured accumulator state to the particle backend."""
        if self.observer_accumulator is not None:
            self.observer_accumulator.send_to_gpu()

    def receive_from_gpu(self):
        """Move configured accumulator state to the host."""
        if self.observer_accumulator is not None:
            self.observer_accumulator.receive_from_gpu()

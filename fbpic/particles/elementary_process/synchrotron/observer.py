# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Fast observer-frame accumulation for incoherent betatron radiation.

The accumulator uses a fixed number of energy-angle samples from the local
synchrotron closure. Its normal per-step cost is consequently independent of
the number of requested photon-energy bins, and it never allocates an
``N_particle x N_energy`` array.

All accumulated arrays contain *integrated energy per bin*.  Conversion to a
density with respect to the selected bin measures is deliberately left to the
openPMD writer.  This makes energy accounting independent of bin spacing.
"""

import math
from contextlib import nullcontext
import re
from functools import lru_cache

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar
from scipy.special import kv

from fbpic.utils.cuda import cuda_installed

if cuda_installed:
    import cupy


_FOUR_ACCELERATION_POWER_FACTOR = (
    e**2 / (6.0 * math.pi * epsilon_0 * c**3)
)
_ANGULAR_POWER_FACTOR = e**2 / (16.0 * math.pi**2 * epsilon_0 * c)
_PROJECTED_ANGLE_LIMIT = 0.5 * math.pi

# Mergeable source moments store total weight, six local references, six
# small mean offsets (x, y, z, theta_x, theta_y, retarded time), and the
# packed upper triangle of the centered second-moment matrix. Separating the
# reference from the offset preserves small source sizes at large coordinates.
_SOURCE_VARIABLE_COUNT = 6
_SOURCE_STAT_SIZE = 1 + 2 * _SOURCE_VARIABLE_COUNT + (
    _SOURCE_VARIABLE_COUNT * (_SOURCE_VARIABLE_COUNT + 1) // 2)
_UINT64_MASK = (1 << 64) - 1

_DEFAULT_RESOLUTION_WARNING_THRESHOLDS = {
    "delta_eta": 0.1,
    "delta_theta_u": 0.01,
    "chi_turn": 1.0,
}
_RESOLUTION_QUANTITIES = tuple(_DEFAULT_RESOLUTION_WARNING_THRESHOLDS)


def _splitmix64(value, xp):
    """Vectorized SplitMix64 finalizer, identical with NumPy and CuPy."""
    overflow_context = np.errstate(over="ignore") if xp is np else nullcontext()
    with overflow_context:
        value = value + xp.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> xp.uint64(30))) \
            * xp.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> xp.uint64(27))) \
            * xp.uint64(0x94D049BB133111EB)
    return value ^ (value >> xp.uint64(31))


# Stream identifiers are part of the public reproducibility contract. New
# stochastic channels must receive a distinct identifier and never reuse one.
_RANDOM_STREAM_PARTICLE_THINNING = 0x5458494E
_RANDOM_STREAM_PHOTON_ENERGY = 0x454E4552
_RANDOM_STREAM_ANGULAR_MAGNITUDE = 0x414E474D
_RANDOM_STREAM_ANGULAR_SIGN = 0x414E4753


def _event_uniform(
        seed, particle_id, event_index, stream_id, sample_index, xp,
        namespace=0):
    """Return a stateless variate keyed only by persistent integer identity."""
    key = xp.asarray(particle_id, dtype=xp.uint64)
    fields = (seed, namespace, event_index, stream_id, sample_index)
    salts = (
        0x243F6A8885A308D3, 0x13198A2E03707344,
        0xA4093822299F31D0, 0x082EFA98EC4E6C89,
        0x452821E638D01377)
    for value, salt in zip(fields, salts):
        encoded = (
            xp.asarray(value, dtype=xp.uint64)
            + xp.uint64(salt & _UINT64_MASK))
        key = _splitmix64(key ^ _splitmix64(encoded, xp), xp)
    return (key >> xp.uint64(11)).astype(xp.float64) * (1.0 / 2.0**53)


def _packed_upper_index(first, second, count=_SOURCE_VARIABLE_COUNT):
    """Index of one upper-triangular entry in the packed moment state."""
    if second < first:
        first, second = second, first
    return first * count - first * (first - 1) // 2 + second - first

_PARTICLE_SELECTION_ALIASES = {
    "gamma": "gamma", "x": "x", "y": "y", "z": "z",
    "ux": "ux", "uy": "uy", "uz": "uz",
    "theta_x": "particle_theta_x", "theta_y": "particle_theta_y",
    "weight": "weight", "w": "weight",
}

_RADIATION_SELECTION_RANGES = {
    "energy_range": "energy", "energy_band": "energy",
    "observer_time_range": "time", "time_range": "time",
    "x_range": "x", "y_range": "y", "z_range": "z",
    "theta_x_range": "theta_x", "theta_y_range": "theta_y",
}


@lru_cache(maxsize=8)
def _cached_angular_kernel_data(maximum_scaled_energy):
    """Tabulate Schwinger inverse-CDF data and its angle normalization."""
    maximum_scaled_energy = float(maximum_scaled_energy)
    if (not math.isfinite(maximum_scaled_energy)
            or maximum_scaled_energy <= 0.0):
        raise ValueError(
            "maximum_scaled_energy must be finite and positive.")
    minimum_scaled_energy = min(
        1.0e-8, maximum_scaled_energy * 1.0e-4)
    x_grid = np.geomspace(
        minimum_scaled_energy, maximum_scaled_energy, 96)
    probability = np.linspace(0.0, 1.0, 257)
    q_grid = np.linspace(0.0, 8.0, 1025)
    inverse = np.empty((x_grid.size, probability.size), dtype=np.float64)
    normalization = np.empty(x_grid.size, dtype=np.float64)
    for index, scaled_energy in enumerate(x_grid):
        x_third = scaled_energy**(1.0 / 3.0)
        y = q_grid / x_third
        one_plus_y2 = 1.0 + y**2
        xi = 0.5 * scaled_energy * one_plus_y2**1.5
        density = one_plus_y2**2 * (
            kv(2.0 / 3.0, xi)**2
            + y**2 / one_plus_y2 * kv(1.0 / 3.0, xi)**2
        ) / x_third
        density[~np.isfinite(density)] = 0.0
        cdf = np.zeros_like(q_grid)
        cdf[1:] = np.cumsum(
            0.5 * (density[:-1] + density[1:]) * np.diff(q_grid))
        if not cdf[-1] > 0.0:
            raise RuntimeError(
                "Could not normalize the synchrotron angular kernel.")
        normalization[index] = cdf[-1]
        cdf /= normalization[index]
        inverse[index] = np.interp(probability, cdf, q_grid)
    log_x = np.log(x_grid)
    for array in (x_grid, log_x, probability, inverse, normalization):
        array.setflags(write=False)
    return x_grid, log_x, probability, inverse, normalization


@lru_cache(maxsize=8)
def _cached_angular_kernel(maximum_scaled_energy):
    """Return the established four-array angular-kernel interface."""
    return _cached_angular_kernel_data(maximum_scaled_energy)[:4]


def _safe_name(value):
    """Return a stable HDF5-compatible fragment."""
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return name or "channel"


def _require_unique(items, label):
    names = [item["name"] for item in items]
    if len(names) != len(set(names)):
        raise ValueError("%s names must be unique." % label)


def _as_edges(values, name):
    """Validate and copy monotonically increasing bin edges."""
    edges = np.asarray(values, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("`%s` must be a one-dimensional array of bin edges."
                         % name)
    if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError("`%s` must contain finite, strictly increasing edges."
                         % name)
    return edges.copy()


def _as_projected_angle_edges(values, name):
    """Validate edges in the one-to-one forward projected-angle chart."""
    edges = _as_edges(values, name)
    if (edges[0] <= -_PROJECTED_ANGLE_LIMIT
            or edges[-1] >= _PROJECTED_ANGLE_LIMIT):
        raise ValueError(
            "`%s` must lie strictly inside (-pi/2, pi/2); projected "
            "angles represent only the forward hemisphere." % name)
    return edges


def _as_range(values, name, nonnegative=False):
    """Validate a selectable half-open observer-frame interval."""
    if values is None or len(values) != 2:
        raise ValueError("`%s` must contain exactly two bounds." % name)
    lower, upper = float(values[0]), float(values[1])
    if np.isnan(lower) or np.isnan(upper) or not lower < upper:
        raise ValueError("`%s` must contain two ordered, non-NaN bounds."
                         % name)
    if nonnegative and lower < 0.0:
        raise ValueError("`%s` cannot start below zero." % name)
    return (lower, upper)


def _as_projected_angle_range(values, name):
    bounds = _as_range(values, name)
    if (bounds[0] <= -_PROJECTED_ANGLE_LIMIT
            or bounds[1] >= _PROJECTED_ANGLE_LIMIT):
        raise ValueError(
            "`%s` must lie strictly inside (-pi/2, pi/2)." % name)
    return bounds


def _unit_vector(value, name="direction"):
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("`%s` must be a finite three-vector." % name)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("`%s` must be nonzero." % name)
    return vector / norm


def _cone_quadrature(direction, half_angle, count):
    """Return equal-solid-angle Fibonacci points in a circular aperture."""
    direction = _unit_vector(direction)
    if half_angle <= 0.0:
        return direction.reshape(1, 3), np.zeros(1)
    if not (half_angle < math.pi):
        raise ValueError("Detector aperture half angles must be below pi.")
    count = max(1, int(count))

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, direction)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    basis_1 = np.cross(helper, direction)
    basis_1 /= np.linalg.norm(basis_1)
    basis_2 = np.cross(direction, basis_1)

    cap = 1.0 - math.cos(half_angle)
    solid_angle = 2.0 * math.pi * cap
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    rays = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        cos_angle = 1.0 - cap * (index + 0.5) / count
        sin_angle = math.sqrt(max(0.0, 1.0 - cos_angle**2))
        azimuth = index * golden_angle
        rays[index] = (
            cos_angle * direction
            + sin_angle * math.cos(azimuth) * basis_1
            + sin_angle * math.sin(azimuth) * basis_2
        )
    return rays, np.full(count, solid_angle / count)


def _normalize_band(band, index):
    if isinstance(band, dict):
        name = _safe_name(band.get("name", "band_%d" % index))
        bounds = band.get("energy_range", band.get("range"))
    else:
        name = "band_%d" % index
        bounds = band
    if bounds is None or len(bounds) != 2:
        raise ValueError("Each detector energy band needs two energy bounds.")
    lower, upper = float(bounds[0]), float(bounds[1])
    if np.isnan(lower) or np.isnan(upper) or not (0.0 <= lower < upper):
        raise ValueError("Detector energy bands must obey 0 <= min < max.")
    return {"name": name, "energy_range": (lower, upper)}


def _normalize_radiation_selection(selection, label):
    """Validate an energy/angle/source/time packet selection."""
    normalized = dict(selection or {})
    allowed = set(_RADIATION_SELECTION_RANGES) | {
        "angular_range", "angular_region", "direction", "half_angle"
    }
    unknown = set(normalized) - allowed
    if unknown:
        raise ValueError("Unknown %s keys: %s" % (
            label, ", ".join(sorted(unknown))))
    for key in _RADIATION_SELECTION_RANGES:
        if key in normalized:
            range_name = "%s %s" % (label, key)
            if key in ("theta_x_range", "theta_y_range"):
                normalized[key] = _as_projected_angle_range(
                    normalized[key], range_name)
            else:
                normalized[key] = _as_range(
                    normalized[key], range_name,
                    nonnegative=key in ("energy_range", "energy_band"))
    angular_key = (
        "angular_range" if "angular_range" in normalized
        else "angular_region" if "angular_region" in normalized else None)
    if angular_key is not None:
        angular_range = normalized[angular_key]
        if angular_range is None or len(angular_range) != 2:
            raise ValueError(
                "`%s %s` needs theta_x and theta_y ranges."
                % (label, angular_key))
        normalized[angular_key] = (
            _as_projected_angle_range(
                angular_range[0], "%s theta_x" % label),
            _as_projected_angle_range(
                angular_range[1], "%s theta_y" % label),
        )
    if "direction" in normalized:
        normalized["direction"] = _unit_vector(
            normalized["direction"], "%s direction" % label)
        half_angle = float(normalized.get("half_angle", 0.0))
        if not (0.0 <= half_angle < math.pi):
            raise ValueError("`%s half_angle` must obey 0 <= value < pi."
                             % label)
        normalized["half_angle"] = half_angle
    elif "half_angle" in normalized:
        raise ValueError("`%s half_angle` requires a direction." % label)
    return normalized


def _normalize_detector(
        detector, index, common_time_edges, default_energy_band_mode):
    if isinstance(detector, dict):
        item = dict(detector)
    else:
        item = {"direction": detector}
    if "direction" not in item and "theta_x" in item and "theta_y" in item:
        theta_x = float(item["theta_x"])
        theta_y = float(item["theta_y"])
        if not (-_PROJECTED_ANGLE_LIMIT < theta_x <
                _PROJECTED_ANGLE_LIMIT) or not (
                    -_PROJECTED_ANGLE_LIMIT < theta_y <
                    _PROJECTED_ANGLE_LIMIT):
            raise ValueError(
                "Detector projected angles must lie inside "
                "(-pi/2, pi/2).")
        tangent_x = math.tan(theta_x)
        tangent_y = math.tan(theta_y)
        item["direction"] = (tangent_x, tangent_y, 1.0)
    if "direction" not in item:
        raise ValueError("Every detector channel requires a `direction`.")
    name = _safe_name(item.get("name", "detector_%d" % index))
    direction = _unit_vector(item["direction"], "detector direction")
    time_values = item.get("observer_time_edges",
                           item.get("time_edges", common_time_edges))
    if time_values is None:
        raise ValueError(
            "Detector `%s` requires observer-time bin edges." % name)
    time_edges = _as_edges(time_values, "observer_time_edges")
    half_angle = float(item.get(
        "half_angle", item.get("aperture_half_angle", 0.0)))
    if half_angle < 0.0:
        raise ValueError("Detector aperture half angles cannot be negative.")
    quadrature = int(item.get("aperture_quadrature", 12))
    if quadrature < 1:
        raise ValueError("`aperture_quadrature` must be a positive integer.")
    band_values = item.get("energy_bands", [])
    if isinstance(band_values, dict):
        band_values = [band_values]
    bands = [
        _normalize_band(band, band_index)
        for band_index, band in enumerate(band_values)
    ]
    _require_unique(bands, "Detector energy-band")
    energy_band_mode = item.get(
        "energy_band_mode", item.get("band_mode", default_energy_band_mode))
    if energy_band_mode not in ("joint", "separable"):
        raise ValueError(
            "Detector energy_band_mode must be 'joint' or 'separable'.")
    interval = item.get("pulse_interval", (0.05, 0.95))
    if len(interval) != 2 or not (0.0 <= interval[0] < interval[1] <= 1.0):
        raise ValueError("`pulse_interval` must contain two ordered fractions.")
    return {
        "name": name,
        "direction": direction,
        "time_edges": time_edges,
        "half_angle": half_angle,
        "aperture_quadrature": quadrature,
        "rays": None,
        "ray_weights": None,
        "energy_bands": bands,
        "energy_band_mode": energy_band_mode,
        "pulse_interval": (float(interval[0]), float(interval[1])),
    }


def _normalize_projection(projection, index, available_edges):
    if isinstance(projection, str):
        if projection != "full":
            axes = (projection,)
        else:
            axes = tuple(axis for axis in (
                "x", "y", "z", "theta_x", "theta_y", "energy", "time"
            ) if axis in available_edges)
        item = {"axes": axes, "name": projection}
    elif isinstance(projection, dict):
        item = dict(projection)
    else:
        item = {"axes": tuple(projection)}
    axes = tuple(item.get("axes", ()))
    if not axes:
        raise ValueError("Every source projection needs at least one axis.")
    if len(set(axes)) != len(axes):
        raise ValueError("Source projection axes cannot be repeated.")
    unknown = [axis for axis in axes if axis not in available_edges]
    if unknown:
        raise ValueError(
            "Missing bin edges for source projection axes: %s"
            % ", ".join(unknown))
    name = _safe_name(item.get("name", "_".join(axes) or "projection_%d" % index))
    selection = _normalize_radiation_selection(
        item.get("selection", {}), "source projection selection")
    return {
        "name": name,
        "axes": axes,
        "edges": tuple(available_edges[axis] for axis in axes),
        "selection": selection,
        "time_reference": item.get(
            "time_reference", item.get(
                "observer_time_reference", "photon_direction")),
    }


def _expand_moment_selections(selections):
    """Expand convenience energy/angle-bin requests into named selections."""
    if isinstance(selections, (dict, str)):
        selections = [selections]
    expanded = []
    if selections is None:
        selections = []
    for index, selection in enumerate(selections):
        if isinstance(selection, str):
            item = {"name": selection}
        else:
            item = dict(selection)
        base_name = _safe_name(item.pop("name", "selection_%d" % index))
        energy_bins = item.pop("energy_bins", None)
        angle_x_bins = item.pop("theta_x_bins", None)
        angle_y_bins = item.pop("theta_y_bins", None)
        if energy_bins is not None:
            edges = _as_edges(energy_bins, "source moment energy_bins")
            for bin_index in range(edges.size - 1):
                child = dict(item)
                child["name"] = "%s_energy_%d" % (base_name, bin_index)
                child["energy_range"] = (edges[bin_index], edges[bin_index + 1])
                expanded.append(child)
        elif angle_x_bins is not None or angle_y_bins is not None:
            if angle_x_bins is None or angle_y_bins is None:
                raise ValueError(
                    "Source-moment angular bins require both theta axes.")
            x_edges = _as_projected_angle_edges(
                angle_x_bins, "source moment theta_x_bins")
            y_edges = _as_projected_angle_edges(
                angle_y_bins, "source moment theta_y_bins")
            for ix in range(x_edges.size - 1):
                for iy in range(y_edges.size - 1):
                    child = dict(item)
                    child["name"] = "%s_angle_%d_%d" % (base_name, ix, iy)
                    child["angular_range"] = (
                        (x_edges[ix], x_edges[ix + 1]),
                        (y_edges[iy], y_edges[iy + 1]),
                    )
                    expanded.append(child)
        else:
            item["name"] = base_name
            expanded.append(item)
    return expanded


class ObserverFrameRadiationAccumulator(object):
    """Accumulate independently selectable observer-frame radiation products."""

    def __init__(
            self, radiating_species, dt_sim, spectral_x, spectral_cdf,
            spectral_truncated_fraction=None,
            gamma_boost=1.0, beta_boost=0.0,
            observer_frame="laboratory", observer_translation=None,
            enabled_channels=None, photon_energy_edges=None,
            theta_x_edges=None, theta_y_edges=None,
            angular_measure="solid_angle", detectors=None,
            observer_time_edges=None, source_coordinate_edges=None,
            source_projections=None, source_moments=None,
            source_z_intervals=None, source_z_interval_edges=None,
            samples_per_particle=1, gamma_cutoff=10.0, particle_selection=None,
            particle_batch_size=262144, energy_band_mode="joint",
            random_seed=0, random_namespace=0, particle_sampling_fraction=1.0,
            resolution_warning_thresholds=None,
            output_mode="cumulative", mpi_size=1,
            max_allocation_bytes=1073741824,
            allocation_budget_bytes=None,
            identity_tracking_will_be_activated=False,
            identity_sorting_buffer_will_be_allocated=False):
        self.eon = radiating_species
        tracker = getattr(radiating_species, "tracker", None)
        if (tracker is None
                and not identity_tracking_will_be_activated):
            raise ValueError(
                "Observer radiation requires persistent particle IDs. "
                "Configure the diagnostic with a Particles species so that "
                "tracking can be activated before accumulation.")
        if (tracker is not None
                and (not hasattr(tracker, "id")
                     or int(tracker.id.size) != int(
                         radiating_species.Ntot))):
            raise ValueError(
                "Persistent particle-ID and particle arrays have unequal sizes.")
        self.identity_tracking_will_be_activated = bool(
            identity_tracking_will_be_activated)
        self.identity_sorting_buffer_will_be_allocated = bool(
            identity_sorting_buffer_will_be_allocated)
        self.use_cuda = radiating_species.use_cuda
        self.dt_sim = float(dt_sim)
        if output_mode not in ("cumulative", "interval", "both"):
            raise ValueError("Unknown radiation output mode.")
        self.output_mode = output_mode
        self.mpi_size = max(1, int(mpi_size))
        self.observer_frame = str(observer_frame)
        if self.observer_frame == "lab":
            self.observer_frame = "laboratory"
        if self.observer_frame not in ("laboratory", "simulation"):
            raise ValueError(
                "`observer_frame` must be 'laboratory' or 'simulation'.")
        if self.observer_frame == "simulation":
            gamma_boost, beta_boost = 1.0, 0.0
        self.gamma_boost = float(gamma_boost)
        self.beta_boost = float(beta_boost)
        if self.gamma_boost < 1.0 or abs(self.beta_boost) >= 1.0:
            raise ValueError("Invalid observer-frame Lorentz boost.")
        expected_beta2 = 1.0 - self.gamma_boost**-2
        if not math.isclose(
                self.beta_boost**2, expected_beta2, rel_tol=2.e-13,
                abs_tol=2.e-15):
            raise ValueError(
                "Observer-frame boost gamma and beta are inconsistent.")

        if observer_translation is None:
            observer_translation = np.zeros(4)
        translation = np.asarray(observer_translation, dtype=np.float64)
        if translation.shape != (4,) or not np.all(np.isfinite(translation)):
            raise ValueError(
                "`observer_translation` must be (ct, x, y, z) in SI units.")
        self.observer_translation = translation

        self.spectral_x = _as_edges_like_grid(spectral_x, "spectral_x")
        self.spectral_cdf = np.asarray(spectral_cdf, dtype=np.float64).copy()
        if self.spectral_cdf.shape != self.spectral_x.shape:
            raise ValueError("Synchrotron CDF and x grid have unequal shapes.")
        if (not np.all(np.isfinite(self.spectral_cdf))
                or np.any(np.diff(self.spectral_cdf) < 0.0)
                or self.spectral_cdf[-1] <= 0.0
                or self.spectral_cdf[-1] > 1.0 + 2.0e-14):
            raise ValueError("The synchrotron CDF must be finite and monotonic.")
        self.spectral_cdf[0] = 0.0
        self.spectral_cdf[-1] = min(self.spectral_cdf[-1], 1.0)
        inferred_truncation = 1.0 - self.spectral_cdf[-1]
        if spectral_truncated_fraction is None:
            spectral_truncated_fraction = inferred_truncation
        self.spectral_truncated_fraction = float(
            spectral_truncated_fraction)
        if (not (0.0 <= self.spectral_truncated_fraction < 1.0)
                or not math.isclose(
                    self.spectral_truncated_fraction, inferred_truncation,
                    rel_tol=2.0e-12, abs_tol=2.0e-14)):
            raise ValueError(
                "The spectral CDF endpoint and truncated fraction disagree.")

        self.gamma_cutoff = float(gamma_cutoff)
        if self.gamma_cutoff <= 1.0:
            raise ValueError("`gamma_cutoff` must be greater than one.")
        self.particle_selection = {}
        for key, bounds in dict(particle_selection or {}).items():
            if key not in _PARTICLE_SELECTION_ALIASES:
                raise ValueError(
                    "Unsupported observer-frame particle selection `%s`."
                    % key)
            if key in ("theta_x", "theta_y"):
                self.particle_selection[key] = _as_projected_angle_range(
                    bounds, "particle_selection %s" % key)
            else:
                self.particle_selection[key] = _as_range(
                    bounds, "particle_selection %s" % key,
                    nonnegative=key in ("gamma", "weight", "w"))
        self.samples_per_particle = int(samples_per_particle)
        if self.samples_per_particle < 1:
            raise ValueError("`samples_per_particle` must be a positive integer.")
        self.particle_batch_size = int(particle_batch_size)
        if self.particle_batch_size < 1:
            raise ValueError("`particle_batch_size` must be a positive integer.")
        if energy_band_mode not in ("joint", "separable"):
            raise ValueError(
                "energy_band_mode must be 'joint' or 'separable'.")
        self.energy_band_mode = energy_band_mode
        try:
            seed_integer = int(random_seed)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("random_seed must be an integer.")
        if isinstance(random_seed, (float, np.floating)) \
                and random_seed != seed_integer:
            raise ValueError("random_seed must be an integer.")
        self.random_seed = seed_integer & _UINT64_MASK
        self.random_namespace = int(random_namespace) & _UINT64_MASK
        self.random_stream_ids = {
            "particle_thinning": _RANDOM_STREAM_PARTICLE_THINNING,
            "photon_energy": _RANDOM_STREAM_PHOTON_ENERGY,
            "angular_magnitude": _RANDOM_STREAM_ANGULAR_MAGNITUDE,
            "angular_sign": _RANDOM_STREAM_ANGULAR_SIGN,
        }
        self.particle_sampling_fraction = float(
            particle_sampling_fraction)
        if (not math.isfinite(self.particle_sampling_fraction)
                or not (0.0 < self.particle_sampling_fraction <= 1.0)):
            raise ValueError(
                "particle_sampling_fraction must obey 0 < value <= 1.")

        thresholds = dict(_DEFAULT_RESOLUTION_WARNING_THRESHOLDS)
        if resolution_warning_thresholds is not None:
            try:
                supplied_thresholds = dict(resolution_warning_thresholds)
            except (TypeError, ValueError):
                raise ValueError(
                    "resolution_warning_thresholds must be a mapping.")
            unknown_thresholds = (
                set(supplied_thresholds) - set(_RESOLUTION_QUANTITIES))
            if unknown_thresholds:
                raise ValueError(
                    "Unknown pusher-resolution thresholds: %s"
                    % ", ".join(sorted(unknown_thresholds)))
            thresholds.update(supplied_thresholds)
        for name, value in thresholds.items():
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "Resolution warning thresholds must be finite and "
                    "nonnegative.")
            thresholds[name] = value
        self.resolution_warning_thresholds = thresholds

        if max_allocation_bytes is None:
            self.max_allocation_bytes = None
        else:
            self.max_allocation_bytes = int(max_allocation_bytes)
            if self.max_allocation_bytes < 1:
                raise ValueError(
                    "max_allocation_bytes must be positive or None.")
        if allocation_budget_bytes is None:
            self.allocation_budget_bytes = self.max_allocation_bytes
        else:
            self.allocation_budget_bytes = int(allocation_budget_bytes)
            if self.allocation_budget_bytes < 1:
                raise ValueError(
                    "allocation_budget_bytes must be positive or None.")
        if angular_measure not in ("solid_angle", "projected_angles"):
            raise ValueError(
                "`angular_measure` must be 'solid_angle' or "
                "'projected_angles'.")
        self.angular_measure = angular_measure

        if enabled_channels is None:
            enabled_channels = []
            if (photon_energy_edges is not None
                    and theta_x_edges is not None
                    and theta_y_edges is not None):
                enabled_channels.append("angular_spectral")
            if detectors:
                enabled_channels.append("observer_time")
            if source_projections:
                enabled_channels.append("source")
            if source_moments or source_z_intervals:
                enabled_channels.append("source_moments")
            enabled_channels.append("accounting")
        if isinstance(enabled_channels, str):
            enabled_channels = [enabled_channels]
        aliases = {
            "spectrum": "angular_spectral",
            "detectors": "observer_time",
            "time": "observer_time",
            "moments": "source_moments",
        }
        self.enabled_channels = {
            aliases.get(channel, channel) for channel in enabled_channels
        }
        allowed = {
            "angular_spectral", "observer_time", "source",
            "source_moments", "accounting"
        }
        unknown = self.enabled_channels - allowed
        if unknown:
            raise ValueError("Unknown radiation output channels: %s"
                             % ", ".join(sorted(unknown)))
        # Accounting is a handful of scalars and is always maintained so that
        # no enabled product can silently lose normalization information.
        self.enabled_channels.add("accounting")

        self.energy_edges = None
        self.theta_x_edges = None
        self.theta_y_edges = None
        if photon_energy_edges is not None:
            self.energy_edges = _as_edges(
                photon_energy_edges, "photon_energy_edges")
            if self.energy_edges[0] < 0.0:
                raise ValueError("Photon-energy edges cannot be negative.")
        if theta_x_edges is not None:
            self.theta_x_edges = _as_projected_angle_edges(
                theta_x_edges, "theta_x_edges")
        if theta_y_edges is not None:
            self.theta_y_edges = _as_projected_angle_edges(
                theta_y_edges, "theta_y_edges")
        if "angular_spectral" in self.enabled_channels:
            if (self.energy_edges is None or self.theta_x_edges is None
                    or self.theta_y_edges is None):
                raise ValueError(
                    "The angular-spectral channel requires photon-energy, "
                    "theta_x, and theta_y bin edges.")

        self.common_time_edges = None
        if observer_time_edges is not None:
            self.common_time_edges = _as_edges(
                observer_time_edges, "observer_time_edges")
        detector_values = [] if detectors is None else detectors
        if isinstance(detector_values, dict):
            detector_values = [detector_values]
        self.detectors = [
            _normalize_detector(
                detector, index, self.common_time_edges,
                self.energy_band_mode)
            for index, detector in enumerate(detector_values)
        ]
        _require_unique(self.detectors, "Detector")
        if "observer_time" in self.enabled_channels and not self.detectors:
            raise ValueError(
                "The observer-time channel requires at least one detector.")

        coordinate_edges = dict(source_coordinate_edges or {})
        unknown_coordinates = set(coordinate_edges) - {"x", "y", "z"}
        if unknown_coordinates:
            raise ValueError("Unknown source coordinate axes: %s" %
                             ", ".join(sorted(unknown_coordinates)))
        available_edges = {}
        for axis in ("x", "y", "z"):
            if axis in coordinate_edges:
                available_edges[axis] = _as_edges(
                    coordinate_edges[axis], "source_%s_edges" % axis)
        if self.theta_x_edges is not None:
            available_edges["theta_x"] = self.theta_x_edges
        if self.theta_y_edges is not None:
            available_edges["theta_y"] = self.theta_y_edges
        if self.energy_edges is not None:
            available_edges["energy"] = self.energy_edges
        if self.common_time_edges is not None:
            available_edges["time"] = self.common_time_edges
        self.source_axis_edges = available_edges
        if source_z_intervals is None:
            source_z_intervals = ()
        elif np.isscalar(source_z_intervals):
            source_z_intervals = (source_z_intervals,)
        normalized_intervals = tuple(
            sorted(set(float(value) for value in source_z_intervals)))
        if any(
                not math.isfinite(value) or not (0.0 < value < 1.0)
                for value in normalized_intervals):
            raise ValueError(
                "source_z_intervals must contain fractions strictly between "
                "zero and one.")
        if source_z_interval_edges is None:
            source_z_interval_edges = available_edges.get("z")
        self.source_z_intervals = normalized_intervals
        self.source_z_interval_edges = (
            None if source_z_interval_edges is None
            else _as_edges(
                source_z_interval_edges, "source_z_interval_edges"))
        if self.source_z_intervals and self.source_z_interval_edges is None:
            raise ValueError(
                "Central source-z intervals require source_z_interval_edges "
                "or source_coordinate_edges['z'].")

        projection_values = [] if source_projections is None \
            else source_projections
        if isinstance(projection_values, (dict, str)):
            projection_values = [projection_values]
        self.source_projections = [
            _normalize_projection(projection, index, available_edges)
            for index, projection in enumerate(projection_values)
        ]
        coordinate_selection_keys = {
            "x_range", "y_range", "z_range"
        }
        time_selection_keys = {"observer_time_range", "time_range"}
        detector_directions = {
            detector["name"]: detector["direction"]
            for detector in self.detectors}
        for projection in self.source_projections:
            reference = projection["time_reference"]
            if reference in (None, "photon", "photon_direction"):
                reference = "photon_direction"
                direction = None
            elif reference in detector_directions:
                direction = detector_directions[reference]
            else:
                raise ValueError(
                    "Source projection time_reference must be "
                    "'photon_direction' or a configured detector name.")
            projection["time_reference"] = reference
            projection["time_direction"] = direction
            has_time_selection = bool(
                set(projection["selection"]) & time_selection_keys)
            position_only = (
                set(projection["axes"]) <= {"x", "y", "z"}
                and (not has_time_selection
                     or reference != "photon_direction"))
            detector_timed = (
                reference != "photon_direction"
                and set(projection["axes"]) <= {"x", "y", "z", "time"}
                and "time" in projection["axes"])
            projection["deterministic"] = bool(
                (position_only or detector_timed)
                and set(projection["selection"]) <= (
                    coordinate_selection_keys | time_selection_keys))
        _require_unique(self.source_projections, "Source projection")
        if "source" in self.enabled_channels and not self.source_projections:
            raise ValueError(
                "The source channel requires `source_projections`.")

        self.source_moment_selections = _expand_moment_selections(
            source_moments)
        if "source_moments" in self.enabled_channels \
                and not self.source_moment_selections:
            # An unfiltered selection is the useful, inexpensive default.
            self.source_moment_selections = [{"name": "all"}]
        normalized_selections = []
        for index, selection in enumerate(self.source_moment_selections):
            selection = dict(selection)
            name = _safe_name(
                selection.pop("name", "selection_%d" % index))
            quantities = selection.pop(
                "quantities", ("position", "angle", "time"))
            time_reference = selection.pop(
                "time_reference", selection.pop(
                    "observer_time_reference", "photon_direction"))
            if time_reference in (None, "photon", "photon_direction"):
                time_reference = "photon_direction"
                time_direction = None
            elif time_reference in detector_directions:
                time_direction = detector_directions[time_reference]
            else:
                raise ValueError(
                    "Source-moment time_reference must be "
                    "'photon_direction' or a configured detector name.")
            if isinstance(quantities, str):
                quantities = (quantities,)
            quantities = tuple(quantities)
            allowed_quantities = {"position", "angle", "time"}
            if (not quantities or len(set(quantities)) != len(quantities)
                    or set(quantities) - allowed_quantities):
                raise ValueError(
                    "Source-moment quantities must be a nonempty subset of "
                    "'position', 'angle', and 'time'.")
            selection = _normalize_radiation_selection(
                selection, "source moment selection")
            selection["name"] = name
            selection["quantities"] = quantities
            selection["time_reference"] = time_reference
            selection["time_direction"] = time_direction
            has_time_selection = bool(
                set(selection) & time_selection_keys)
            position_only = (
                set(quantities) == {"position"}
                and (not has_time_selection
                     or time_reference != "photon_direction"))
            detector_timed = (
                "time" in quantities and "angle" not in quantities
                and time_reference != "photon_direction")
            selection["deterministic"] = bool(
                (position_only or detector_timed)
                and set(selection) <= (
                    {"name", "quantities", "time_reference",
                     "time_direction"}
                    | coordinate_selection_keys | time_selection_keys))
            normalized_selections.append(selection)
        self.source_moment_selections = normalized_selections
        _require_unique(self.source_moment_selections, "Source moment selection")

        self.data = {}
        self.data_axes = {}
        self.data_kinds = {}
        self.accounting = {}
        self.sampling = {}
        self.quality = {}
        self.interval_quality = {}
        self.resolution_stats = {}
        self.interval_resolution_stats = {}
        self.moment_stats = {}
        self.interval_moment_stats = {}
        self.source_z_histograms = {}

        dense_breakdown = self._estimate_dense_allocation()
        self.estimated_dense_product_bytes = sum(dense_breakdown.values())
        self.allocation_breakdown = self._estimate_total_allocation(
            dense_breakdown)
        self.estimated_total_allocation_bytes = sum(
            self.allocation_breakdown.values())
        if (self.allocation_budget_bytes is not None
                and self.estimated_total_allocation_bytes
                > self.allocation_budget_bytes):
            raise MemoryError(
                "Requested total radiation diagnostic footprint is %d bytes, "
                "exceeding the available max_allocation_bytes budget of %d. "
                "Breakdown: %s"
                % (self.estimated_total_allocation_bytes,
                   self.allocation_budget_bytes,
                   ", ".join("%s=%d" % item
                             for item in sorted(
                                 self.allocation_breakdown.items()))))

        self.completed_event_count = 0
        self.last_completed_event_index = None
        self.cumulative_timing = self._empty_timing()
        self.interval_timing = self._empty_timing()
        self._initialize_detector_quadratures()
        self._initialize_storage()
        self.pusher_endpoint_buffer = tuple(
            np.empty(self.particle_batch_size, dtype=np.float64)
            for _ in range(3))
        self.angular_kernel_x = None
        self.angular_kernel_log_x = None
        self.angular_kernel_probability = None
        self.angular_kernel_inverse = None
        self.joint_band_x = None
        self.joint_band_log_x = None
        self.joint_band_y = None
        self.joint_band_y_coordinate = None
        self.joint_band_cdf = None
        if self.needs_angular_kernel:
            self._initialize_angular_kernel()
        if self.needs_joint_band_model:
            self._initialize_joint_band_kernel()
        self._runtime = self._all_runtime_arrays()
        self._on_gpu = False

    def _initialize_detector_quadratures(self):
        """Allocate deterministic aperture rays after memory preflight."""
        for detector in self.detectors:
            rays, ray_weights = _cone_quadrature(
                detector["direction"], detector["half_angle"],
                detector["aperture_quadrature"])
            detector["rays"] = rays
            detector["ray_weights"] = ray_weights

    def _initialize_angular_kernel(self):
        """Tabulate the Schwinger vertical-angle conditional distribution.

        At a fixed scaled photon energy ``x = omega/omega_c``, the
        polarization-summed spectral-angular distribution normal to the
        instantaneous orbit plane is proportional to

        ``(1+y^2)^2 [K_(2/3)(xi)^2 + y^2/(1+y^2) K_(1/3)(xi)^2]``,

        where ``y = gamma*psi`` and
        ``xi = x*(1+y^2)^(3/2)/2``.  Storing the inverse CDF in
        ``q=x^(1/3)*abs(y)`` resolves the broad low-energy tail with a small,
        fixed table.  The conditional is normalized independently for every
        x, so the prescribed one-dimensional synchrotron closure remains the
        exact energy marginal.
        """
        table = _cached_angular_kernel(float(self.spectral_x[-1]))
        self.angular_kernel_x = table[0]
        self.angular_kernel_log_x = table[1]
        self.angular_kernel_probability = table[2]
        self.angular_kernel_inverse = table[3]

    @staticmethod
    def _empty_timing():
        return {
            "event_count": 0,
            "first_event_index": np.iinfo(np.int64).max,
            "last_event_index": np.iinfo(np.int64).min,
            "first_event_center": math.inf,
            "last_event_center": -math.inf,
            "represented_interval_start": math.inf,
            "represented_interval_end": -math.inf,
        }

    def _record_event_timing(self, simulation_time, event_index):
        for timing in (self.cumulative_timing, self.interval_timing):
            timing["event_count"] += 1
            timing["first_event_index"] = min(
                timing["first_event_index"], event_index)
            timing["last_event_index"] = max(
                timing["last_event_index"], event_index)
            timing["first_event_center"] = min(
                timing["first_event_center"], simulation_time)
            timing["last_event_center"] = max(
                timing["last_event_center"], simulation_time)
            timing["represented_interval_start"] = min(
                timing["represented_interval_start"],
                simulation_time - 0.5*self.dt_sim)
            timing["represented_interval_end"] = max(
                timing["represented_interval_end"],
                simulation_time + 0.5*self.dt_sim)

    def _estimate_dense_allocation(self):
        """Return byte costs before allocating dense detector/source arrays."""
        costs = {}
        if "angular_spectral" in self.enabled_channels:
            shape = (
                self.theta_x_edges.size - 1,
                self.theta_y_edges.size - 1,
                self.energy_edges.size - 1,
            )
            costs["angular_spectral"] = (
                math.prod(int(value) for value in shape) * 8)
        if "observer_time" in self.enabled_channels:
            for detector in self.detectors:
                bins = int(detector["time_edges"].size - 1)
                prefix = "detector/%s" % detector["name"]
                costs[prefix + "/direction"] = bins * 8
                if detector["half_angle"] > 0.0:
                    costs[prefix + "/aperture"] = bins * 8
                for band in detector["energy_bands"]:
                    band_prefix = prefix + "/band/" + band["name"]
                    costs[band_prefix + "/direction"] = bins * 8
                    if detector["half_angle"] > 0.0:
                        costs[band_prefix + "/aperture"] = bins * 8
        if "source" in self.enabled_channels:
            for projection in self.source_projections:
                shape = tuple(
                    int(edges.size - 1) for edges in projection["edges"])
                costs["source/" + projection["name"]] = (
                    math.prod(shape) * 8)
        return costs

    def _estimate_total_allocation(self, dense_costs):
        """Estimate peak resident and writer memory before product allocation.

        The batch workspace count is deliberately conservative: vectorized
        event construction keeps many simultaneous four-vector, power,
        selection, histogram-index, and packet arrays.  Its size remains
        bounded by particle_batch_size and never by the full species size.
        """
        costs = dict(dense_costs)
        dense_bytes = sum(dense_costs.values())
        selection_count = len(self.source_moment_selections)
        moment_state_bytes = selection_count * _SOURCE_STAT_SIZE * 8
        if moment_state_bytes:
            costs["source_moment_cumulative_and_interval_state"] = (
                2 * moment_state_bytes)

        source_z_bytes = 0
        if self.source_z_intervals:
            source_z_bytes = (
                selection_count
                * int(self.source_z_interval_edges.size + 1) * 8)
            costs["source_z_interval_histograms"] = source_z_bytes

        aperture_count = sum(
            detector["half_angle"] > 0.0 for detector in self.detectors)
        accounting_slots = (
            3 + (3 if self.energy_edges is not None else 0)
            + (1 if "angular_spectral" in self.enabled_channels else 0)
            + 4 * aperture_count + len(dense_costs))
        sampling_slots = 8 + 2 * aperture_count
        quality_and_resolution_slots = 2 * 3 + 2 * 3 * 5
        timing_slots = 2 * 7
        scalar_slots = (
            accounting_slots + sampling_slots
            + quality_and_resolution_slots + timing_slots)
        costs["persistent_scalar_and_resolution_state"] = scalar_slots * 8
        costs["persistent_python_state_overhead_estimate"] = (
            1024 + 256 * (
                len(self.detectors) + selection_count
                + len(self.source_projections) + len(dense_costs)))

        spectral_lookup = (
            int(self.spectral_x.nbytes) + int(self.spectral_cdf.nbytes))
        angular_runtime_lookup = 0
        angular_cached_auxiliary = 0
        lookup_construction_workspace = 0
        if self.needs_angular_kernel:
            angular_runtime_lookup = (96 + 96 + 257 + 96 * 257) * 8
            angular_cached_auxiliary = 96 * 8
            # q-grid plus Bessel, density, CDF, and interpolation work arrays.
            lookup_construction_workspace += 16 * 1025 * 8
        joint_lookup = 0
        if self.needs_joint_band_model:
            joint_lookup = (96 + 96 + 129 + 129 + 129 * 96) * 8
            lookup_construction_workspace += (16 * 96 + 4 * 129) * 8
        device_lookup_bytes = (
            spectral_lookup + angular_runtime_lookup + joint_lookup)
        costs["lookup_tables"] = (
            device_lookup_bytes + angular_cached_auxiliary)
        if lookup_construction_workspace:
            costs["temporary_lookup_table_construction_workspace"] = (
                lookup_construction_workspace)
        radiator = getattr(self.eon, "synchrotron_radiator", None)
        if radiator is not None:
            costs["radiator_activation_spectral_table_copies"] = sum(
                int(array.nbytes) for array in (
                    radiator.spectral_x, radiator.spectral_cdf))

        optional_axis_arrays = (
            self.energy_edges, self.theta_x_edges, self.theta_y_edges,
            self.common_time_edges)
        configuration_arrays = [self.observer_translation]
        configuration_arrays.extend(
            array for array in optional_axis_arrays if array is not None)
        configuration_arrays.extend(self.source_axis_edges.values())
        if self.source_z_interval_edges is not None:
            configuration_arrays.append(self.source_z_interval_edges)
        detector_quadrature_bytes = 0
        for detector in self.detectors:
            configuration_arrays.extend((
                detector["direction"], detector["time_edges"]))
            ray_count = (
                detector["aperture_quadrature"]
                if detector["half_angle"] > 0.0 else 1)
            detector_quadrature_bytes += ray_count * 4 * 8
        # Several source axes can alias the primary axes on the host.
        costs["detector_aperture_quadrature"] = detector_quadrature_bytes
        configuration_bytes = 0
        seen_arrays = set()
        for array in configuration_arrays:
            marker = id(array)
            if marker not in seen_arrays:
                configuration_bytes += int(array.nbytes)
                seen_arrays.add(marker)
        costs["configuration_and_axis_arrays"] = configuration_bytes

        # Runtime dictionaries use distinct names for primary and source axes;
        # their GPU conversion materializes every named entry.
        runtime_configuration_bytes = sum(
            int(array.nbytes) for array in optional_axis_arrays
            if array is not None)
        runtime_configuration_bytes += sum(
            int(array.nbytes) for array in self.source_axis_edges.values())
        if self.source_z_interval_edges is not None:
            runtime_configuration_bytes += int(
                self.source_z_interval_edges.nbytes)
        runtime_configuration_bytes += detector_quadrature_bytes
        for detector in self.detectors:
            runtime_configuration_bytes += sum(
                int(array.nbytes) for array in (
                    detector["direction"], detector["time_edges"]))

        batch = self.particle_batch_size
        endpoint_buffer_bytes = 3 * batch * 8
        costs["pusher_radiation_endpoint_buffer"] = endpoint_buffer_bytes
        # 80 float, 12 integer, and 16 boolean vector slots cover the peak
        # event dictionary, selection masks, indexed histogram coordinates,
        # and reduction temporaries used by one vectorized batch.
        costs["particle_batch_event_workspace"] = (
            batch * (80 * 8 + 12 * 8 + 16))
        if self.needs_spectral_samples:
            packet_slots = 20 + max(1, aperture_count)
            costs["spectral_angular_packet_workspace"] = (
                batch * packet_slots * 8)
        identity_bytes = int(self.eon.Ntot) * 8
        costs["persistent_particle_identity"] = identity_bytes
        if self.identity_sorting_buffer_will_be_allocated:
            costs["particle_identity_sorting_buffer"] = identity_bytes

        snapshot_moment_copies = moment_state_bytes
        if self.output_mode in ("interval", "both"):
            snapshot_moment_copies += moment_state_bytes
        persistent_snapshot = (
            dense_bytes + snapshot_moment_copies + source_z_bytes
            + scalar_slots * 8)
        costs["temporary_writer_snapshot_copies"] = persistent_snapshot
        costs["temporary_writer_configuration_copies"] = configuration_bytes
        if self.output_mode in ("interval", "both"):
            costs["writer_interval_baseline"] = (
                dense_bytes + source_z_bytes + scalar_slots * 8)
            costs["temporary_writer_interval_difference"] = (
                dense_bytes + source_z_bytes + scalar_slots * 8)
        costs["temporary_writer_density_copy"] = (
            max(dense_costs.values()) if dense_costs else 0)
        if self.mpi_size > 1:
            output_mode_count = 2 if self.output_mode == "both" else 1
            reduced_snapshot = (
                dense_bytes + moment_state_bytes + source_z_bytes
                + scalar_slots * 8)
            costs["reduction_buffers"] = (
                output_mode_count * reduced_snapshot)
            # Maxima, centered moments, and timing metadata use object
            # gathers. Charge the root rank for the transient list and its
            # serialized Python/NumPy payload in addition to the retained
            # reduced snapshots above. One gather is live at a time.
            gather_item_bytes = max(
                _SOURCE_STAT_SIZE * 8 if selection_count else 0, 1024)
            costs["temporary_mpi_object_gather_workspace"] = (
                self.mpi_size * gather_item_bytes)
        if self.use_cuda:
            costs["device_runtime_array_copies"] = (
                device_lookup_bytes + runtime_configuration_bytes)
            costs["temporary_gpu_writer_transfer_copies"] = (
                dense_bytes + 2 * moment_state_bytes + source_z_bytes
                + scalar_slots * 8)
            costs["temporary_gpu_initialization_transfer_copies"] = (
                endpoint_buffer_bytes + identity_bytes)
        return {key: int(value) for key, value in costs.items() if value > 0}

    def _initialize_joint_band_kernel(self):
        """Tabulate energy CDFs conditioned on Schwinger vertical angle."""
        x_grid = self.angular_kernel_x
        finite_cdf = np.interp(x_grid, self.spectral_x, self.spectral_cdf)
        spectral_density = np.maximum(
            np.gradient(finite_cdf, x_grid, edge_order=2), 0.0)
        angular_normalization = _cached_angular_kernel_data(
            float(self.spectral_x[-1]))[4]
        y_grid = np.concatenate((
            np.array([0.0]), np.geomspace(1.0e-4, 64.0, 128)))
        conditional_cdf = np.zeros(
            (y_grid.size, x_grid.size), dtype=np.float64)
        for index, y_value in enumerate(y_grid):
            one_plus_y2 = 1.0 + y_value**2
            xi = 0.5 * x_grid * one_plus_y2**1.5
            angular_density = one_plus_y2**2 * (
                kv(2.0 / 3.0, xi)**2
                + y_value**2 / one_plus_y2
                * kv(1.0 / 3.0, xi)**2
            ) / angular_normalization
            joint_density = spectral_density * angular_density
            joint_density[~np.isfinite(joint_density)] = 0.0
            conditional_cdf[index, 1:] = np.cumsum(
                0.5 * (joint_density[:-1] + joint_density[1:])
                * np.diff(x_grid))
            endpoint = conditional_cdf[index, -1]
            if endpoint > 0.0:
                conditional_cdf[index] *= (
                    self.spectral_cdf[-1] / endpoint)
        self.joint_band_x = x_grid.copy()
        self.joint_band_log_x = np.log(self.joint_band_x)
        self.joint_band_y = y_grid
        self.joint_band_y_coordinate = np.arcsinh(y_grid)
        self.joint_band_cdf = conditional_cdf

    def _initialize_storage(self):
        if "angular_spectral" in self.enabled_channels:
            key = "angular_spectral"
            shape = (
                self.theta_x_edges.size - 1,
                self.theta_y_edges.size - 1,
                self.energy_edges.size - 1,
            )
            self.data[key] = np.zeros(shape, dtype=np.float64)
            self.data_axes[key] = (
                ("theta_x", self.theta_x_edges),
                ("theta_y", self.theta_y_edges),
                ("energy", self.energy_edges),
            )
            self.data_kinds[key] = "angular_spectral"

        if "observer_time" in self.enabled_channels:
            for detector in self.detectors:
                name = detector["name"]
                time_edges = detector["time_edges"]
                direction_key = "detector/%s/direction" % name
                self.data[direction_key] = np.zeros(
                    time_edges.size - 1, dtype=np.float64)
                self.data_axes[direction_key] = (("time", time_edges),)
                self.data_kinds[direction_key] = "direction_time"
                if detector["half_angle"] > 0.0:
                    aperture_key = "detector/%s/aperture" % name
                    self.data[aperture_key] = np.zeros(
                        time_edges.size - 1, dtype=np.float64)
                    self.data_axes[aperture_key] = (("time", time_edges),)
                    self.data_kinds[aperture_key] = "aperture_time"
                for band in detector["energy_bands"]:
                    band_key = "detector/%s/band/%s/direction" % (
                        name, band["name"])
                    self.data[band_key] = np.zeros(
                        time_edges.size - 1, dtype=np.float64)
                    self.data_axes[band_key] = (("time", time_edges),)
                    self.data_kinds[band_key] = "direction_time_band"
                    if detector["half_angle"] > 0.0:
                        aperture_band_key = \
                            "detector/%s/band/%s/aperture" % (
                                name, band["name"])
                        self.data[aperture_band_key] = np.zeros(
                            time_edges.size - 1, dtype=np.float64)
                        self.data_axes[aperture_band_key] = (
                            ("time", time_edges),)
                        self.data_kinds[aperture_band_key] = \
                            "aperture_time_band"

        if "source" in self.enabled_channels:
            for projection in self.source_projections:
                key = "source/%s" % projection["name"]
                shape = tuple(edges.size - 1 for edges in projection["edges"])
                self.data[key] = np.zeros(shape, dtype=np.float64)
                self.data_axes[key] = tuple(zip(
                    projection["axes"], projection["edges"]))
                self.data_kinds[key] = "source"

        if "source_moments" in self.enabled_channels:
            for selection in self.source_moment_selections:
                name = selection["name"]
                self.moment_stats[name] = np.zeros(
                    _SOURCE_STAT_SIZE, dtype=np.float64)
                self.interval_moment_stats[name] = np.zeros(
                    _SOURCE_STAT_SIZE, dtype=np.float64)
                if self.source_z_intervals:
                    # Underflow, interior bins, and overflow make the
                    # interval coverage explicit instead of silently clipping.
                    self.source_z_histograms[name] = np.zeros(
                        self.source_z_interval_edges.size + 1,
                        dtype=np.float64)

        for name in (
                "max_mass_shell_relative_error",
                "max_orthogonality_relative_error",
                "max_power_identity_relative_error"):
            self.quality[name] = np.zeros(1, dtype=np.float64)
            self.interval_quality[name] = np.zeros(1, dtype=np.float64)

        # Per quantity: maximum, transverse-energy denominator, weighted
        # first moment, weighted second moment, and energy above threshold.
        for name in _RESOLUTION_QUANTITIES:
            self.resolution_stats[name] = np.zeros(5, dtype=np.float64)
            self.interval_resolution_stats[name] = np.zeros(
                5, dtype=np.float64)

        for name in (
                "invalid_pusher_events",
                "eligible_macroparticle_events",
                "sampled_macroparticle_events",
                "physical_macroparticle_weight",
                "effective_sampled_weight",
                "effective_sampled_weight_squared",
                "transverse_energy_sampling_variance",
                "longitudinal_energy_sampling_variance"):
            self.sampling[name] = np.zeros(1, dtype=np.float64)
        for detector in self.detectors:
            if detector["half_angle"] <= 0.0:
                continue
            name = detector["name"]
            for estimator in ("aperture", "outside_aperture"):
                key = (
                    "stochastic_spectral_angular_%s_energy_"
                    "sampling_variance/%s" % (estimator, name))
                self.sampling[key] = np.zeros(1, dtype=np.float64)

        base_accounting = [
            "transverse_energy", "longitudinal_energy",
            "energy_truncated_by_spectral_closure",
        ]
        if self.energy_edges is not None:
            base_accounting.extend((
                "energy_below_range", "energy_within_range",
                "energy_above_range",
            ))
        if "angular_spectral" in self.enabled_channels:
            base_accounting.append("energy_outside_angular_grid")
        for detector in self.detectors:
            if detector["half_angle"] <= 0.0:
                continue
            name = detector["name"]
            base_accounting.extend((
                "deterministic_broadband_aperture_energy/%s" % name,
                "stochastic_spectral_angular_aperture_energy/%s" % name,
                "stochastic_spectral_angular_outside_aperture_energy/%s"
                % name,
                "stochastic_spectral_angular_partition_energy/%s" % name,
            ))
        for name in base_accounting:
            self.accounting[name] = np.zeros(1, dtype=np.float64)
        for key, kind in self.data_kinds.items():
            if kind.startswith("direction_time"):
                suffix = "_energy_per_solid_angle"
            else:
                suffix = "_energy"
            self.accounting["represented/%s%s" % (key, suffix)] = \
                np.zeros(1, dtype=np.float64)

    @property
    def needs_positions(self):
        return bool(
            "observer_time" in self.enabled_channels
            or "source" in self.enabled_channels
            or "source_moments" in self.enabled_channels
            or any(axis in self.particle_selection for axis in ("x", "y", "z"))
        )

    @property
    def needs_joint_band_model(self):
        return any(
            detector["energy_bands"]
            and detector["energy_band_mode"] == "joint"
            for detector in self.detectors)

    @property
    def needs_angular_kernel(self):
        return self.needs_spectral_samples or self.needs_joint_band_model

    @property
    def needs_spectral_samples(self):
        return bool(
            "angular_spectral" in self.enabled_channels
            or ("source" in self.enabled_channels and any(
                not projection["deterministic"]
                for projection in self.source_projections))
            or ("source_moments" in self.enabled_channels and any(
                not selection["deterministic"]
                for selection in self.source_moment_selections))
            or any(
                detector["half_angle"] > 0.0
                for detector in self.detectors)
        )

    def _all_runtime_arrays(self):
        arrays = {
            "spectral_x": self.spectral_x,
            "spectral_cdf": self.spectral_cdf,
        }
        if self.angular_kernel_x is not None:
            arrays["angular_kernel/x"] = self.angular_kernel_x
            arrays["angular_kernel/log_x"] = self.angular_kernel_log_x
            arrays["angular_kernel/probability"] = \
                self.angular_kernel_probability
            arrays["angular_kernel/inverse"] = self.angular_kernel_inverse
        if self.joint_band_x is not None:
            arrays["joint_band/x"] = self.joint_band_x
            arrays["joint_band/log_x"] = self.joint_band_log_x
            arrays["joint_band/y"] = self.joint_band_y
            arrays["joint_band/y_coordinate"] = (
                self.joint_band_y_coordinate)
            arrays["joint_band/cdf"] = self.joint_band_cdf
        if self.energy_edges is not None:
            arrays["energy"] = self.energy_edges
        if self.theta_x_edges is not None:
            arrays["theta_x"] = self.theta_x_edges
        if self.theta_y_edges is not None:
            arrays["theta_y"] = self.theta_y_edges
        if self.common_time_edges is not None:
            arrays["time"] = self.common_time_edges
        for axis, edges in self.source_axis_edges.items():
            arrays["source_axis/%s" % axis] = edges
        if self.source_z_interval_edges is not None:
            arrays["source_z_interval_edges"] = self.source_z_interval_edges
        for detector in self.detectors:
            prefix = "detector/%s" % detector["name"]
            arrays[prefix + "/direction"] = detector["direction"]
            arrays[prefix + "/time"] = detector["time_edges"]
            arrays[prefix + "/rays"] = detector["rays"]
            arrays[prefix + "/ray_weights"] = detector["ray_weights"]
        return arrays

    def send_to_gpu(self):
        if not self.use_cuda or self._on_gpu:
            return
        self.data = {key: cupy.asarray(value)
                     for key, value in self.data.items()}
        self.accounting = {key: cupy.asarray(value)
                           for key, value in self.accounting.items()}
        self.moment_stats = {key: cupy.asarray(value)
                             for key, value in self.moment_stats.items()}
        self.interval_moment_stats = {
            key: cupy.asarray(value)
            for key, value in self.interval_moment_stats.items()}
        self.source_z_histograms = {
            key: cupy.asarray(value)
            for key, value in self.source_z_histograms.items()}
        self.sampling = {key: cupy.asarray(value)
                         for key, value in self.sampling.items()}
        self.quality = {key: cupy.asarray(value)
                        for key, value in self.quality.items()}
        self.interval_quality = {
            key: cupy.asarray(value)
            for key, value in self.interval_quality.items()}
        self.resolution_stats = {
            key: cupy.asarray(value)
            for key, value in self.resolution_stats.items()}
        self.interval_resolution_stats = {
            key: cupy.asarray(value)
            for key, value in self.interval_resolution_stats.items()}
        self.pusher_endpoint_buffer = tuple(
            cupy.asarray(value) for value in self.pusher_endpoint_buffer)
        self._runtime = {key: cupy.asarray(value)
                         for key, value in self._all_runtime_arrays().items()}
        self._on_gpu = True

    def receive_from_gpu(self):
        if not self.use_cuda or not self._on_gpu:
            return
        self.data = {key: value.get() for key, value in self.data.items()}
        self.accounting = {
            key: value.get() for key, value in self.accounting.items()}
        self.moment_stats = {
            key: value.get() for key, value in self.moment_stats.items()}
        self.interval_moment_stats = {
            key: value.get()
            for key, value in self.interval_moment_stats.items()}
        self.source_z_histograms = {
            key: value.get()
            for key, value in self.source_z_histograms.items()}
        self.sampling = {
            key: value.get() for key, value in self.sampling.items()}
        self.quality = {
            key: value.get() for key, value in self.quality.items()}
        self.interval_quality = {
            key: value.get()
            for key, value in self.interval_quality.items()}
        self.resolution_stats = {
            key: value.get()
            for key, value in self.resolution_stats.items()}
        self.interval_resolution_stats = {
            key: value.get()
            for key, value in self.interval_resolution_stats.items()}
        self._runtime = self._all_runtime_arrays()
        self._on_gpu = False

    def _xp(self):
        if self._on_gpu:
            return cupy
        return np

    def _cdf_at(self, scaled_energy, xp):
        """Interpolate the normalized synchrotron-power CDF."""
        x_grid = self._runtime["spectral_x"]
        cdf = self._runtime["spectral_cdf"]
        index = xp.searchsorted(x_grid, scaled_energy, side="right") - 1
        index_clip = xp.clip(index, 0, x_grid.size - 2)
        x0 = x_grid[index_clip]
        x1 = x_grid[index_clip + 1]
        fraction = (scaled_energy - x0) / (x1 - x0)
        value = cdf[index_clip] + fraction * (
            cdf[index_clip + 1] - cdf[index_clip])
        value = xp.where(scaled_energy <= x_grid[0], 0.0, value)
        value = xp.where(scaled_energy >= x_grid[-1], cdf[-1], value)
        return xp.clip(value, 0.0, cdf[-1])

    def _inverse_cdf(self, probability, xp):
        x_grid = self._runtime["spectral_x"]
        cdf = self._runtime["spectral_cdf"]
        probability = xp.clip(probability, 0.0, cdf[-1])
        upper = xp.searchsorted(cdf, probability, side="right")
        upper = xp.clip(upper, 1, cdf.size - 1)
        lower = upper - 1
        denominator = cdf[upper] - cdf[lower]
        denominator = xp.where(denominator > 0.0, denominator, 1.0)
        fraction = (probability - cdf[lower]) / denominator
        return x_grid[lower] + fraction * (x_grid[upper] - x_grid[lower])

    @staticmethod
    def _histogram_add(target, values, edges, weights, xp):
        """Add an arbitrary-dimensional weighted histogram to ``target``."""
        valid = xp.ones(weights.shape, dtype=xp.bool_)
        indices = []
        for coordinate, axis_edges in zip(values, edges):
            index = xp.searchsorted(axis_edges, coordinate, side="right") - 1
            index = xp.where(coordinate == axis_edges[-1],
                             axis_edges.size - 2, index)
            valid &= (index >= 0) & (index < axis_edges.size - 1)
            indices.append(index)
        flat = indices[0]
        for index, axis_edges in zip(indices[1:], edges[1:]):
            flat = flat * (axis_edges.size - 1) + index
        selected_flat = flat[valid]
        selected_weights = weights[valid]
        # Atomic indexed addition avoids a temporary array as large as a
        # potentially high-dimensional source product.
        xp.add.at(target.ravel(), selected_flat, selected_weights)
        return xp.sum(selected_weights)

    def _source_z_add(self, target, z, weights, xp):
        """Accumulate an interval histogram with explicit tail energy."""
        edges = self._runtime["source_z_interval_edges"]
        target[0] += xp.sum(weights[z < edges[0]])
        target[-1] += xp.sum(weights[z > edges[-1]])
        self._histogram_add(
            target[1:-1], (z,), (edges,), weights, xp)


    def _selection_mask(self, selection, values, xp):
        reference = values.get("energy", next(iter(values.values())))
        mask = xp.ones(reference.shape, dtype=xp.bool_)
        for key, value_name in _RADIATION_SELECTION_RANGES.items():
            if key not in selection:
                continue
            lower, upper = selection[key]
            value = values[value_name]
            mask &= (value >= lower) & (value < upper)
        angular_range = selection.get(
            "angular_range", selection.get("angular_region"))
        if angular_range is not None:
            x_range, y_range = angular_range
            mask &= ((values["theta_x"] >= x_range[0])
                     & (values["theta_x"] < x_range[1])
                     & (values["theta_y"] >= y_range[0])
                     & (values["theta_y"] < y_range[1]))
        if "direction" in selection:
            direction = selection["direction"]
            half_angle = float(selection.get("half_angle", 0.0))
            dot = (values["nx"] * direction[0]
                   + values["ny"] * direction[1]
                   + values["nz"] * direction[2])
            mask &= dot >= math.cos(half_angle)
        return mask

    def _particle_mask(self, event, xp):
        mask = event["gamma"] >= self.gamma_cutoff
        for requested, bounds in self.particle_selection.items():
            lower, upper = bounds
            value = event[_PARTICLE_SELECTION_ALIASES[requested]]
            mask &= (value >= lower) & (value < upper)
        return mask

    def _observer_event(
            self, eon, lower_momentum, simulation_time, xp,
            particle_slice=None, batch_count=None, event_index=None):
        """Construct and covariantly transform one centered pusher impulse."""
        if particle_slice is None:
            particle_slice = slice(None)
        if event_index is None:
            event_index = int(round(float(simulation_time) / self.dt_sim))

        # Production coupling buffers are batch-local, whereas explicitly
        # supplied compatibility endpoints may span the complete species.
        if batch_count is None:
            if int(lower_momentum[0].size) == int(eon.Ntot):
                lower_slice = particle_slice
            else:
                lower_slice = slice(None)
        else:
            lower_slice = slice(0, int(batch_count))
        ux_minus = lower_momentum[0][lower_slice]
        uy_minus = lower_momentum[1][lower_slice]
        uz_minus = lower_momentum[2][lower_slice]
        ux_plus = eon.ux[particle_slice]
        uy_plus = eon.uy[particle_slice]
        uz_plus = eon.uz[particle_slice]
        gamma_minus = xp.sqrt(
            1.0 + ux_minus**2 + uy_minus**2 + uz_minus**2)
        gamma_plus = xp.sqrt(
            1.0 + ux_plus**2 + uy_plus**2 + uz_plus**2)

        delta_ux = ux_plus - ux_minus
        delta_uy = uy_plus - uy_minus
        delta_uz = uz_plus - uz_minus
        # Evaluate gamma_+ - gamma_- through the factored mass-shell
        # identity. Direct subtraction loses most significant digits for the
        # small impulse of an ultrarelativistic particle.
        gamma_sum = gamma_plus + gamma_minus
        delta_gamma = (
            delta_ux * (ux_plus + ux_minus)
            + delta_uy * (uy_plus + uy_minus)
            + delta_uz * (uz_plus + uz_minus)
        ) / gamma_sum
        # Since both endpoints are exactly reconstructed on their mass shell,
        # (U_+ + U_-)^2 = 4 + |Delta u|^2 - Delta gamma^2. This form avoids
        # subtracting two O(gamma^2) quantities for a small pusher impulse.
        center_norm2 = (
            4.0 + delta_ux**2 + delta_uy**2 + delta_uz**2
            - delta_gamma**2)
        center_norm = xp.sqrt(xp.maximum(
            center_norm2, xp.finfo(xp.float64).tiny))
        gamma_sim = gamma_sum / center_norm
        ux_sim = (ux_plus + ux_minus) / center_norm
        uy_sim = (uy_plus + uy_minus) / center_norm
        uz_sim = (uz_plus + uz_minus) / center_norm

        # Observer-independent endpoint resolution indicators.  The rapidity
        # form avoids arcosh(1 + epsilon) cancellation for small impulses.
        relative_spacelike2 = xp.maximum(
            delta_ux**2 + delta_uy**2 + delta_uz**2 - delta_gamma**2,
            0.0)
        delta_eta = 2.0 * xp.arcsinh(
            0.5 * xp.sqrt(relative_spacelike2))
        momentum_minus = xp.sqrt(
            ux_minus**2 + uy_minus**2 + uz_minus**2)
        momentum_plus = xp.sqrt(
            ux_plus**2 + uy_plus**2 + uz_plus**2)
        direction_denominator = momentum_minus * momentum_plus
        direction_cosine = (
            ux_minus * ux_plus + uy_minus * uy_plus + uz_minus * uz_plus)
        direction_cosine = direction_cosine / xp.where(
            direction_denominator > 0.0, direction_denominator, 1.0)
        delta_theta_u = xp.where(
            direction_denominator > 0.0,
            xp.arccos(xp.clip(direction_cosine, -1.0, 1.0)),
            0.0)
        chi_turn = gamma_sim * delta_theta_u

        delta_tau = self.dt_sim / gamma_sim
        acceleration_scale = c / delta_tau
        a0_sim = acceleration_scale * delta_gamma
        ax_sim = acceleration_scale * delta_ux
        ay_sim = acceleration_scale * delta_uy
        az_sim = acceleration_scale * delta_uz

        # Apply one Lorentz matrix to both four-vectors. Light-front
        # components keep the longitudinal boost accurate at large gamma.
        lightfront_boost = self.gamma_boost * (1.0 + self.beta_boost)
        transverse_mass2 = 1.0 + ux_sim**2 + uy_sim**2
        u_plus_sim = xp.where(
            uz_sim >= 0.0, gamma_sim + uz_sim,
            transverse_mass2 / (gamma_sim - uz_sim))
        u_minus_sim = transverse_mass2 / u_plus_sim
        u_plus = lightfront_boost * u_plus_sim
        u_minus = u_minus_sim / lightfront_boost
        gamma = 0.5 * (u_plus + u_minus)
        uz = 0.5 * (u_plus - u_minus)
        ux = ux_sim
        uy = uy_sim

        a_plus_sim = a0_sim + az_sim
        a_minus_sim = a0_sim - az_sim
        a_plus = lightfront_boost * a_plus_sim
        a_minus = a_minus_sim / lightfront_boost
        a0 = 0.5 * (a_plus + a_minus)
        az = 0.5 * (a_plus - a_minus)
        ax = ax_sim
        ay = ay_sim

        inv_gamma = 1.0 / gamma
        beta_x = ux * inv_gamma
        beta_y = uy * inv_gamma
        beta_z = uz * inv_gamma
        beta2 = xp.maximum(1.0 - inv_gamma**2, 0.0)
        beta_abs = xp.sqrt(beta2)
        safe_beta2 = xp.where(beta2 > 0.0, beta2, 1.0)

        cross_x = beta_y * az - beta_z * ay
        cross_y = beta_z * ax - beta_x * az
        cross_z = beta_x * ay - beta_y * ax
        cross2 = cross_x**2 + cross_y**2 + cross_z**2
        p_perp = (
            _FOUR_ACCELERATION_POWER_FACTOR * cross2 / safe_beta2)
        p_parallel = (
            _FOUR_ACCELERATION_POWER_FACTOR * a0**2
            * inv_gamma**2 / safe_beta2)
        p_perp = xp.where(beta2 > 0.0, p_perp, 0.0)
        p_parallel = xp.where(beta2 > 0.0, p_parallel, 0.0)
        omega_c = (
            1.5 * gamma * xp.sqrt(cross2)
            / (c * xp.where(beta_abs > 0.0, beta_abs**3, 1.0)))
        omega_c = xp.where((beta2 > 0.0) & (cross2 > 0.0), omega_c, 0.0)

        dot_gamma = a0 / (c * gamma)
        dot_u_x = ax / (c * gamma)
        dot_u_y = ay / (c * gamma)
        dot_u_z = az / (c * gamma)
        dot_beta_x = (ax - beta_x * a0) / (c * gamma**2)
        dot_beta_y = (ay - beta_y * a0) / (c * gamma**2)
        dot_beta_z = (az - beta_z * a0) / (c * gamma**2)

        parallel_coefficient = a0 / safe_beta2
        a_perp_x = ax - beta_x * parallel_coefficient
        a_perp_y = ay - beta_y * parallel_coefficient
        a_perp_z = az - beta_z * parallel_coefficient
        a_perp_x = xp.where(beta2 > 0.0, a_perp_x, 0.0)
        a_perp_y = xp.where(beta2 > 0.0, a_perp_y, 0.0)
        a_perp_z = xp.where(beta2 > 0.0, a_perp_z, 0.0)
        dot_beta_perp_x = a_perp_x / (c * gamma**2)
        dot_beta_perp_y = a_perp_y / (c * gamma**2)
        dot_beta_perp_z = a_perp_z / (c * gamma**2)

        x_sim = eon.x[particle_slice]
        y_sim = eon.y[particle_slice]
        z_sim = eon.z[particle_slice]
        particle_id = xp.asarray(
            eon.tracker.id[particle_slice], dtype=xp.uint64)
        event_index_array = xp.full(
            particle_id.shape, xp.uint64(event_index & _UINT64_MASK),
            dtype=xp.uint64)

        ct_sim = c * float(simulation_time)
        # Transform both observer light-front coordinates and apply the
        # translation before reconstructing ct and z. Retaining these two
        # coordinates also makes near-axis retarded times cancellation-safe.
        ct_plus_z_observer = (
            lightfront_boost * (ct_sim + z_sim)
            + self.observer_translation[0]
            + self.observer_translation[3])
        ct_minus_z_observer = (
            (ct_sim - z_sim) / lightfront_boost
            + self.observer_translation[0]
            - self.observer_translation[3])
        ct_observer = 0.5 * (
            ct_plus_z_observer + ct_minus_z_observer)
        z_observer = 0.5 * (
            ct_plus_z_observer - ct_minus_z_observer)
        x_observer = x_sim + self.observer_translation[1]
        y_observer = y_sim + self.observer_translation[2]
        time_observer = ct_observer / c

        dt_observer = gamma * delta_tau
        mass_shell_error = (
            gamma_sim**2 - ux_sim**2 - uy_sim**2 - uz_sim**2 - 1.0)
        u_dot_a_sim = c * (
            gamma_sim * a0_sim
            - ux_sim * ax_sim - uy_sim * ay_sim - uz_sim * az_sim)
        euclidean_u_norm = xp.sqrt(
            gamma_sim**2 + ux_sim**2 + uy_sim**2 + uz_sim**2)
        euclidean_a_norm = xp.sqrt(
            a0_sim**2 + ax_sim**2 + ay_sim**2 + az_sim**2)
        orthogonality_relative_error = xp.abs(u_dot_a_sim) / xp.where(
            c * euclidean_u_norm * euclidean_a_norm > 0.0,
            c * euclidean_u_norm * euclidean_a_norm, 1.0)
        mass_shell_relative_error = xp.abs(mass_shell_error) / (
            gamma_sim**2 + ux_sim**2 + uy_sim**2 + uz_sim**2)
        invariant_power = -_FOUR_ACCELERATION_POWER_FACTOR * (
            a0**2 - ax**2 - ay**2 - az**2)

        return {
            "ux": ux, "uy": uy, "uz": uz, "gamma": gamma,
            "inv_gamma": inv_gamma,
            "beta_x": beta_x, "beta_y": beta_y, "beta_z": beta_z,
            "beta2": beta2, "beta_abs": beta_abs,
            "a0": a0, "a_x": ax, "a_y": ay, "a_z": az,
            "a_perp_x": a_perp_x, "a_perp_y": a_perp_y,
            "a_perp_z": a_perp_z,
            "dot_gamma": dot_gamma,
            "dot_u_x": dot_u_x, "dot_u_y": dot_u_y, "dot_u_z": dot_u_z,
            "dot_beta_x": dot_beta_x, "dot_beta_y": dot_beta_y,
            "dot_beta_z": dot_beta_z,
            "dot_beta_perp_x": dot_beta_perp_x,
            "dot_beta_perp_y": dot_beta_perp_y,
            "dot_beta_perp_z": dot_beta_perp_z,
            "p_perp": p_perp, "p_parallel": p_parallel,
            "invariant_power": invariant_power,
            "omega_c": omega_c, "delta_tau": delta_tau,
            "dt_observer": dt_observer,
            "dt_ratio": dt_observer / self.dt_sim,
            "weight": eon.w[particle_slice],
            "particle_theta_x": xp.arctan2(ux, uz),
            "particle_theta_y": xp.arctan2(uy, uz),
            "x": x_observer, "y": y_observer, "z": z_observer,
            "time": time_observer,
            "ct_plus_z": ct_plus_z_observer,
            "ct_minus_z": ct_minus_z_observer,
            "particle_id": particle_id,
            "event_index": event_index_array,
            "delta_eta": delta_eta,
            "delta_theta_u": delta_theta_u,
            "chi_turn": chi_turn,
            "mass_shell_error": mass_shell_error,
            "mass_shell_relative_error": mass_shell_relative_error,
            "u_dot_a_sim": u_dot_a_sim,
            "orthogonality_relative_error": orthogonality_relative_error,
        }

    @staticmethod
    def _filtered(event, mask):
        return {key: value[mask] for key, value in event.items()}

    @staticmethod
    def _angular_power(direction, event, xp, transverse_only=False):
        nx, ny, nz = direction
        suffix = "_perp" if transverse_only else ""
        dot_beta_x = event["dot_beta%s_x" % suffix]
        dot_beta_y = event["dot_beta%s_y" % suffix]
        dot_beta_z = event["dot_beta%s_z" % suffix]
        beta_abs = event["beta_abs"]
        safe_beta_abs = xp.where(beta_abs > 0.0, beta_abs, 1.0)
        beta_hat_x = event["beta_x"] / safe_beta_abs
        beta_hat_y = event["beta_y"] / safe_beta_abs
        beta_hat_z = event["beta_z"] / safe_beta_abs
        one_minus_beta = event["inv_gamma"]**2 / (1.0 + beta_abs)
        # The same cancellation affects n - beta in the numerator. Express it
        # as (n - beta_hat) + (1 - |beta|) beta_hat.
        qx = nx - beta_hat_x + one_minus_beta * beta_hat_x
        qy = ny - beta_hat_y + one_minus_beta * beta_hat_y
        qz = nz - beta_hat_z + one_minus_beta * beta_hat_z
        inner_x = qy * dot_beta_z - qz * dot_beta_y
        inner_y = qz * dot_beta_x - qx * dot_beta_z
        inner_z = qx * dot_beta_y - qy * dot_beta_x
        outer_x = ny * inner_z - nz * inner_y
        outer_y = nz * inner_x - nx * inner_z
        outer_z = nx * inner_y - ny * inner_x
        numerator = outer_x**2 + outer_y**2 + outer_z**2
        # Evaluate 1 - n.beta without subtracting two nearly equal values:
        #
        #   1 - n.beta = (1 - |beta|)
        #                + |beta| |n - beta_hat|^2 / 2,
        #   1 - |beta| = gamma^-2 / (1 + |beta|).
        #
        # This remains accurate for an on-axis ultrarelativistic particle.
        direction_difference2 = (
            (nx - beta_hat_x)**2
            + (ny - beta_hat_y)**2
            + (nz - beta_hat_z)**2
        )
        denominator = (
            one_minus_beta
            + 0.5 * beta_abs * direction_difference2
        )
        denominator = xp.where(beta_abs > 0.0, denominator, 1.0)
        return _ANGULAR_POWER_FACTOR * numerator / denominator**5

    def _band_fraction(self, event, bounds, xp):
        scale = hbar * event["omega_c"]
        safe_scale = xp.where(scale > 0.0, scale, 1.0)
        lower = self._cdf_at(bounds[0] / safe_scale, xp)
        upper = self._cdf_at(bounds[1] / safe_scale, xp)
        return xp.where(scale > 0.0, upper - lower, 0.0)

    def _joint_band_cdf_at(self, scaled_energy, scaled_angle, xp):
        x_grid = self._runtime["joint_band/x"]
        log_grid = self._runtime["joint_band/log_x"]
        y_coordinate_grid = self._runtime["joint_band/y_coordinate"]
        table = self._runtime["joint_band/cdf"]

        safe_x = xp.clip(scaled_energy, x_grid[0], x_grid[-1])
        log_x = xp.log(safe_x)
        x_upper = xp.searchsorted(log_grid, log_x, side="right")
        x_upper = xp.clip(x_upper, 1, x_grid.size - 1)
        x_lower = x_upper - 1
        x_fraction = (log_x - log_grid[x_lower]) / (
            log_grid[x_upper] - log_grid[x_lower])

        y_coordinate = xp.arcsinh(xp.maximum(scaled_angle, 0.0))
        y_coordinate = xp.clip(
            y_coordinate, y_coordinate_grid[0], y_coordinate_grid[-1])
        y_upper = xp.searchsorted(
            y_coordinate_grid, y_coordinate, side="right")
        y_upper = xp.clip(y_upper, 1, y_coordinate_grid.size - 1)
        y_lower = y_upper - 1
        y_fraction = (
            (y_coordinate - y_coordinate_grid[y_lower])
            / (y_coordinate_grid[y_upper] - y_coordinate_grid[y_lower]))

        lower_y = table[y_lower, x_lower] + x_fraction * (
            table[y_lower, x_upper] - table[y_lower, x_lower])
        upper_y = table[y_upper, x_lower] + x_fraction * (
            table[y_upper, x_upper] - table[y_upper, x_lower])
        value = lower_y + y_fraction * (upper_y - lower_y)
        value = xp.where(scaled_energy <= x_grid[0], 0.0, value)
        value = xp.where(
            scaled_energy >= x_grid[-1], self.spectral_cdf[-1], value)
        return xp.clip(value, 0.0, self.spectral_cdf[-1])

    def _joint_band_fraction(self, direction, event, bounds, xp):
        """Energy-band fraction conditioned on the detector emission angle."""
        safe_plane_norm = xp.sqrt(
            event["a_perp_x"]**2 + event["a_perp_y"]**2
            + event["a_perp_z"]**2)
        safe_plane_norm = xp.where(safe_plane_norm > 0.0, safe_plane_norm, 1.0)
        plane_x = event["a_perp_x"] / safe_plane_norm
        plane_y = event["a_perp_y"] / safe_plane_norm
        plane_z = event["a_perp_z"] / safe_plane_norm
        safe_beta = xp.where(event["beta_abs"] > 0.0, event["beta_abs"], 1.0)
        tangent_x = event["beta_x"] / safe_beta
        tangent_y = event["beta_y"] / safe_beta
        tangent_z = event["beta_z"] / safe_beta
        normal_x = tangent_y * plane_z - tangent_z * plane_y
        normal_y = tangent_z * plane_x - tangent_x * plane_z
        normal_z = tangent_x * plane_y - tangent_y * plane_x
        vertical_sine = xp.abs(
            direction[0] * normal_x + direction[1] * normal_y
            + direction[2] * normal_z)
        vertical_angle = xp.arcsin(xp.clip(vertical_sine, 0.0, 1.0))
        scaled_angle = event["gamma"] * vertical_angle

        scale = hbar * event["omega_c"]
        safe_scale = xp.where(scale > 0.0, scale, 1.0)
        lower = self._joint_band_cdf_at(
            bounds[0] / safe_scale, scaled_angle, xp)
        upper = self._joint_band_cdf_at(
            bounds[1] / safe_scale, scaled_angle, xp)
        return xp.where(scale > 0.0, upper - lower, 0.0)

    def _accumulate_detectors(self, event, xp):
        for detector in self.detectors:
            name = detector["name"]
            prefix = "detector/%s" % name
            direction = self._runtime[prefix + "/direction"]
            time_edges = self._runtime[prefix + "/time"]
            d_power = self._angular_power(direction, event, xp)
            emitted = (
                d_power * event["weight"] * event["dt_observer"])
            tau = self._retarded_time(event, direction)
            key = prefix + "/direction"
            represented = self._histogram_add(
                self.data[key], (tau,), (time_edges,), emitted, xp)
            account_key = "represented/%s_energy_per_solid_angle" % key
            self.accounting[account_key][0] += represented

            if detector["energy_bands"]:
                curvature_angular_power = self._angular_power(
                    direction, event, xp, transverse_only=True)
                curvature_emitted = (
                    curvature_angular_power * event["weight"]
                    * event["dt_observer"])
            for band in detector["energy_bands"]:
                if detector["energy_band_mode"] == "joint":
                    fraction = self._joint_band_fraction(
                        direction, event, band["energy_range"], xp)
                else:
                    fraction = self._band_fraction(
                        event, band["energy_range"], xp)
                band_key = prefix + "/band/%s/direction" % band["name"]
                represented = self._histogram_add(
                    self.data[band_key], (tau,), (time_edges,),
                    curvature_emitted * fraction, xp)
                account_key = (
                    "represented/%s_energy_per_solid_angle" % band_key)
                self.accounting[account_key][0] += represented

            if detector["half_angle"] <= 0.0:
                continue
            rays = self._runtime[prefix + "/rays"]
            ray_weights = self._runtime[prefix + "/ray_weights"]
            aperture_key = prefix + "/aperture"
            for ray_index in range(rays.shape[0]):
                ray = rays[ray_index]
                ray_power = self._angular_power(ray, event, xp)
                ray_energy = (
                    ray_power * ray_weights[ray_index] * event["weight"]
                    * event["dt_observer"])
                self.accounting[
                    "deterministic_broadband_aperture_energy/%s"
                    % name][0] += xp.sum(ray_energy)
                ray_tau = self._retarded_time(event, ray)
                represented = self._histogram_add(
                    self.data[aperture_key], (ray_tau,), (time_edges,),
                    ray_energy, xp)
                self.accounting[
                    "represented/%s_energy" % aperture_key][0] += represented
                if detector["energy_bands"]:
                    ray_curvature_power = self._angular_power(
                        ray, event, xp, transverse_only=True)
                    ray_curvature_energy = (
                        ray_curvature_power * ray_weights[ray_index]
                        * event["weight"] * event["dt_observer"])
                for band in detector["energy_bands"]:
                    if detector["energy_band_mode"] == "joint":
                        fraction = self._joint_band_fraction(
                            ray, event, band["energy_range"], xp)
                    else:
                        fraction = self._band_fraction(
                            event, band["energy_range"], xp)
                    band_key = prefix + "/band/%s/aperture" % band["name"]
                    represented = self._histogram_add(
                        self.data[band_key], (ray_tau,), (time_edges,),
                        ray_curvature_energy * fraction, xp)
                    self.accounting[
                        "represented/%s_energy" % band_key][0] += represented

    def _sample_direction(
            self, event, scaled_energy, xp, sample_index=0):
        """Sample the energy-conditioned angle in the local orbit frame."""
        plane_x = event["a_perp_x"]
        plane_y = event["a_perp_y"]
        plane_z = event["a_perp_z"]
        plane_norm = xp.sqrt(plane_x**2 + plane_y**2 + plane_z**2)
        plane_x = plane_x / plane_norm
        plane_y = plane_y / plane_norm
        plane_z = plane_z / plane_norm
        velocity_x = event["beta_x"] / event["beta_abs"]
        velocity_y = event["beta_y"] / event["beta_abs"]
        velocity_z = event["beta_z"] / event["beta_abs"]
        normal_x = velocity_y * plane_z - velocity_z * plane_y
        normal_y = velocity_z * plane_x - velocity_x * plane_z
        normal_z = velocity_x * plane_y - velocity_y * plane_x

        # Invert the tabulated conditional CDF with bilinear interpolation in
        # log(x) and probability. The stateless streams are keyed by the
        # physical event, so batching, sorting, and execution order do not
        # change a particle event's packets.
        x_grid = self._runtime["angular_kernel/x"]
        inverse = self._runtime["angular_kernel/inverse"]
        x_safe = xp.clip(scaled_energy, x_grid[0], x_grid[-1])
        log_x = xp.log(x_safe)
        log_grid = self._runtime["angular_kernel/log_x"]
        x_upper = xp.searchsorted(log_grid, log_x, side="right")
        x_upper = xp.clip(x_upper, 1, x_grid.size - 1)
        x_lower = x_upper - 1
        x_fraction = (log_x - log_grid[x_lower]) / (
            log_grid[x_upper] - log_grid[x_lower])

        random_probability = _event_uniform(
            self.random_seed, event["particle_id"], event["event_index"],
            _RANDOM_STREAM_ANGULAR_MAGNITUDE, sample_index, xp,
            self.random_namespace)
        probability_position = random_probability * (inverse.shape[1] - 1)
        p_lower = xp.floor(probability_position).astype(xp.int64)
        p_lower = xp.clip(p_lower, 0, inverse.shape[1] - 2)
        p_fraction = probability_position - p_lower
        lower_q = inverse[x_lower, p_lower] + p_fraction * (
            inverse[x_lower, p_lower + 1] - inverse[x_lower, p_lower])
        upper_q = inverse[x_upper, p_lower] + p_fraction * (
            inverse[x_upper, p_lower + 1] - inverse[x_upper, p_lower])
        q = lower_q + x_fraction * (upper_q - lower_q)
        y = q / x_safe**(1.0 / 3.0)
        psi = xp.minimum(y * event["inv_gamma"], 0.5 * math.pi)
        sign = xp.where(
            _event_uniform(
                self.random_seed, event["particle_id"],
                event["event_index"], _RANDOM_STREAM_ANGULAR_SIGN,
                sample_index, xp, self.random_namespace) < 0.5,
            -1.0, 1.0)
        sin_psi = sign * xp.sin(psi)
        cos_psi = xp.cos(psi)
        return (
            cos_psi * velocity_x + sin_psi * normal_x,
            cos_psi * velocity_y + sin_psi * normal_y,
            cos_psi * velocity_z + sin_psi * normal_z,
        )

    @staticmethod
    def _update_moments(stats, values, weights, quantities, xp):
        """Merge a referenced, weighted centered batch into stable state."""
        if weights.size == 0:
            return
        active = []
        if "position" in quantities:
            active.extend((0, 1, 2))
        if "angle" in quantities:
            active.extend((3, 4))
        if "time" in quantities:
            active.append(5)
        names = ("x", "y", "z", "theta_x", "theta_y", "time")
        batch_weight = xp.sum(weights)
        safe_batch_weight = xp.where(
            batch_weight > 0.0, batch_weight, 1.0)
        batch_references = {}
        batch_offsets = {}
        for index in active:
            coordinate = values[names[index]]
            reference = coordinate[0]
            batch_references[index] = reference
            batch_offsets[index] = (
                xp.sum(weights * (coordinate - reference))
                / safe_batch_weight)

        old_weight = stats[0].copy()
        total_weight = old_weight + batch_weight
        safe_total_weight = xp.where(
            total_weight > 0.0, total_weight, 1.0)
        merge_scale = old_weight * batch_weight / safe_total_weight
        has_old = old_weight > 0.0
        reference_offset = 1
        mean_offset = 1 + _SOURCE_VARIABLE_COUNT
        m2_offset = 1 + 2 * _SOURCE_VARIABLE_COUNT
        old_references = {
            index: stats[reference_offset + index].copy()
            for index in active}
        old_offsets = {
            index: stats[mean_offset + index].copy()
            for index in active}
        deltas = {
            index: (
                (batch_references[index] - old_references[index])
                + (batch_offsets[index] - old_offsets[index]))
            for index in active}

        for first_position, first in enumerate(active):
            first_centered = (
                values[names[first]] - batch_references[first]
                - batch_offsets[first])
            for second in active[first_position:]:
                second_centered = (
                    values[names[second]] - batch_references[second]
                    - batch_offsets[second])
                batch_m2 = xp.sum(
                    weights * first_centered * second_centered)
                packed = _packed_upper_index(first, second)
                stats[m2_offset + packed] += (
                    batch_m2 + deltas[first] * deltas[second] * merge_scale)
        for index in active:
            stats[reference_offset + index] = xp.where(
                has_old, old_references[index], batch_references[index])
            merged_offset = (
                old_offsets[index]
                + deltas[index] * batch_weight / safe_total_weight)
            stats[mean_offset + index] = xp.where(
                has_old, merged_offset, batch_offsets[index])
        stats[0] = total_weight

    @staticmethod
    def _retarded_time(event, direction):
        """Return t-n.r/c using stable observer light-front coordinates."""
        nx, ny, nz = direction
        ct_minus_n_dot_r = (
            0.5 * (1.0 - nz) * event["ct_plus_z"]
            + 0.5 * (1.0 + nz) * event["ct_minus_z"]
            - nx * event["x"] - ny * event["y"])
        return ct_minus_n_dot_r / c

    def _conditioned_source_values(
            self, values, event, reference, direction, photon_direction=None):
        conditioned = dict(values)
        if reference == "photon_direction":
            conditioned["time"] = self._retarded_time(
                event, photon_direction)
        else:
            conditioned["time"] = self._retarded_time(event, direction)
        return conditioned

    def _accumulate_sample(
            self, event, scaled_energy, packet_weight, xp, sample_index,
            aperture_hits=None):
        nx, ny, nz = self._sample_direction(
            event, scaled_energy, xp, sample_index)
        photon_direction = (nx, ny, nz)
        theta_x = xp.arctan2(nx, nz)
        theta_y = xp.arctan2(ny, nz)
        energy = hbar * event["omega_c"] * scaled_energy
        values = {
            "theta_x": theta_x, "theta_y": theta_y,
            "energy": energy,
            "nx": nx, "ny": ny, "nz": nz,
            "x": event["x"], "y": event["y"], "z": event["z"],
        }

        if "angular_spectral" in self.enabled_channels:
            edges = (
                self._runtime["theta_x"], self._runtime["theta_y"],
                self._runtime["energy"])
            represented = self._histogram_add(
                self.data["angular_spectral"],
                (theta_x, theta_y, energy), edges, packet_weight, xp)
            self.accounting[
                "represented/angular_spectral_energy"][0] += represented
            inside_angle = (
                (theta_x >= edges[0][0]) & (theta_x <= edges[0][-1])
                & (theta_y >= edges[1][0]) & (theta_y <= edges[1][-1]))
            self.accounting["energy_outside_angular_grid"][0] += xp.sum(
                packet_weight[~inside_angle])

        for detector in self.detectors:
            if detector["half_angle"] <= 0.0:
                continue
            name = detector["name"]
            direction = self._runtime[
                "detector/%s/direction" % name]
            inside = (
                nx * direction[0] + ny * direction[1]
                + nz * direction[2]
            ) >= math.cos(detector["half_angle"])
            inside_energy = xp.sum(packet_weight[inside])
            outside_energy = xp.sum(packet_weight[~inside])
            self.accounting[
                "stochastic_spectral_angular_aperture_energy/%s"
                % name][0] += inside_energy
            self.accounting[
                "stochastic_spectral_angular_outside_aperture_energy/%s"
                % name][0] += outside_energy
            self.accounting[
                "stochastic_spectral_angular_partition_energy/%s"
                % name][0] += inside_energy + outside_energy
            if aperture_hits is not None:
                aperture_hits[name] += inside.astype(xp.float64)

        if "source" in self.enabled_channels:
            for projection in self.source_projections:
                if projection["deterministic"]:
                    continue
                conditioned = self._conditioned_source_values(
                    values, event, projection["time_reference"],
                    projection["time_direction"], photon_direction)
                selected = self._selection_mask(
                    projection["selection"], conditioned, xp)
                key = "source/%s" % projection["name"]
                runtime_edges = tuple(
                    self._runtime["source_axis/%s" % axis]
                    for axis in projection["axes"])
                represented = self._histogram_add(
                    self.data[key],
                    tuple(conditioned[axis][selected]
                          for axis in projection["axes"]),
                    runtime_edges, packet_weight[selected], xp)
                self.accounting[
                    "represented/%s_energy" % key][0] += represented

        if "source_moments" in self.enabled_channels:
            for selection in self.source_moment_selections:
                if selection["deterministic"]:
                    continue
                conditioned = self._conditioned_source_values(
                    values, event, selection["time_reference"],
                    selection["time_direction"], photon_direction)
                selected = self._selection_mask(selection, conditioned, xp)
                selected_values = {
                    key: value[selected]
                    for key, value in conditioned.items()}
                for storage in (
                        self.moment_stats, self.interval_moment_stats):
                    self._update_moments(
                        storage[selection["name"]], selected_values,
                        packet_weight[selected], selection["quantities"], xp)
                if selection["name"] in self.source_z_histograms:
                    self._source_z_add(
                        self.source_z_histograms[selection["name"]],
                        conditioned["z"][selected],
                        packet_weight[selected], xp)


    def _accumulate_coordinate_products(self, event, w_perp, xp):
        """Accumulate source products that need no stochastic photon packet."""
        values = {axis: event[axis] for axis in ("x", "y", "z")}
        if "source" in self.enabled_channels:
            for projection in self.source_projections:
                if not projection["deterministic"]:
                    continue
                conditioned = dict(values)
                if ("time" in projection["axes"]
                        or any(
                            key in projection["selection"] for key in (
                                "observer_time_range", "time_range"))):
                    conditioned = self._conditioned_source_values(
                        values, event, projection["time_reference"],
                        projection["time_direction"])
                selected = self._selection_mask(
                    projection["selection"], conditioned, xp)
                key = "source/%s" % projection["name"]
                runtime_edges = tuple(
                    self._runtime["source_axis/%s" % axis]
                    for axis in projection["axes"])
                represented = self._histogram_add(
                    self.data[key],
                    tuple(conditioned[axis][selected]
                          for axis in projection["axes"]),
                    runtime_edges, w_perp[selected], xp)
                self.accounting[
                    "represented/%s_energy" % key][0] += represented
        if "source_moments" in self.enabled_channels:
            for selection in self.source_moment_selections:
                if not selection["deterministic"]:
                    continue
                conditioned = dict(values)
                if ("time" in selection["quantities"]
                        or any(
                            key in selection for key in (
                                "observer_time_range", "time_range"))):
                    conditioned = self._conditioned_source_values(
                        values, event, selection["time_reference"],
                        selection["time_direction"])
                selected = self._selection_mask(selection, conditioned, xp)
                selected_values = {
                    key: value[selected]
                    for key, value in conditioned.items()}
                for storage in (
                        self.moment_stats, self.interval_moment_stats):
                    self._update_moments(
                        storage[selection["name"]], selected_values,
                        w_perp[selected], selection["quantities"], xp)
                if selection["name"] in self.source_z_histograms:
                    self._source_z_add(
                        self.source_z_histograms[selection["name"]],
                        conditioned["z"][selected], w_perp[selected], xp)


    def _accumulate_batch(
            self, lower_momentum, particle_slice, simulation_time, xp,
            batch_count=None, event_index=None):
        event = self._observer_event(
            self.eon, lower_momentum, float(simulation_time), xp,
            particle_slice, batch_count=batch_count, event_index=event_index)
        finite_event = xp.ones(event["gamma"].shape, dtype=xp.bool_)
        for name in (
                "gamma", "ux", "uy", "uz",
                "beta_x", "beta_y", "beta_z",
                "a0", "a_x", "a_y", "a_z",
                "a_perp_x", "a_perp_y", "a_perp_z",
                "dot_beta_x", "dot_beta_y", "dot_beta_z",
                "p_perp", "p_parallel", "invariant_power", "omega_c",
                "delta_tau", "dt_observer", "weight",
                "delta_eta", "delta_theta_u", "chi_turn",
                "x", "y", "z", "time", "ct_plus_z", "ct_minus_z"):
            finite_event &= xp.isfinite(event[name])
        finite_event &= event["delta_tau"] > 0.0
        self.sampling["invalid_pusher_events"][0] += xp.sum(~finite_event)
        if event["gamma"].size:
            acceleration_norm2 = (
                event["a_x"]**2 + event["a_y"]**2 + event["a_z"]**2
                + event["a0"]**2)
            power_denominator = xp.maximum(
                xp.abs(event["invariant_power"]),
                _FOUR_ACCELERATION_POWER_FACTOR * acceleration_norm2
                * xp.finfo(xp.float64).eps)
            power_identity_error = xp.abs(
                event["p_perp"] + event["p_parallel"]
                - event["invariant_power"]) / xp.where(
                    power_denominator > 0.0, power_denominator, 1.0)
            def finite_max(value):
                return xp.max(xp.where(xp.isfinite(value), value, 0.0))

            batch_quality = {
                "max_mass_shell_relative_error": finite_max(
                    event["mass_shell_relative_error"]),
                "max_orthogonality_relative_error": finite_max(
                    event["orthogonality_relative_error"]),
                "max_power_identity_relative_error": finite_max(
                    power_identity_error),
            }
            for storage in (self.quality, self.interval_quality):
                for key, value in batch_quality.items():
                    storage[key][0] = xp.maximum(storage[key][0], value)

        physical_mask = self._particle_mask(event, xp)
        physical_mask &= finite_event
        eligible = self._filtered(event, physical_mask)

        physical_weight = eligible["weight"]
        self.sampling["eligible_macroparticle_events"][0] += xp.sum(
            physical_mask)
        self.sampling["physical_macroparticle_weight"][0] += xp.sum(
            physical_weight)

        resolution_energy = (
            physical_weight * eligible["p_perp"]
            * eligible["dt_observer"])
        for name in _RESOLUTION_QUANTITIES:
            values = eligible[name]
            threshold = self.resolution_warning_thresholds[name]
            for storage in (
                    self.resolution_stats,
                    self.interval_resolution_stats):
                state = storage[name]
                if values.size:
                    state[0] = xp.maximum(state[0], xp.max(values))
                state[1] += xp.sum(resolution_energy)
                state[2] += xp.sum(resolution_energy * values)
                state[3] += xp.sum(resolution_energy * values**2)
                state[4] += xp.sum(
                    resolution_energy[values > threshold])

        probability = self.particle_sampling_fraction
        if probability < 1.0:
            keep = _event_uniform(
                self.random_seed, eligible["particle_id"],
                eligible["event_index"], _RANDOM_STREAM_PARTICLE_THINNING,
                0, xp, self.random_namespace) < probability
        else:
            keep = xp.ones(physical_weight.shape, dtype=xp.bool_)
        event = self._filtered(eligible, keep)
        sampled_physical_weight = event["weight"]
        effective_weight = sampled_physical_weight / probability
        event["weight"] = effective_weight
        self.sampling["sampled_macroparticle_events"][0] += xp.sum(keep)
        self.sampling["effective_sampled_weight"][0] += xp.sum(
            effective_weight)
        self.sampling["effective_sampled_weight_squared"][0] += xp.sum(
            effective_weight**2)

        if probability < 1.0:
            transverse_item = (
                sampled_physical_weight * event["p_perp"]
                * event["dt_observer"])
            longitudinal_item = (
                sampled_physical_weight * event["p_parallel"]
                * event["dt_observer"])
            variance_factor = (1.0 - probability) / probability**2
            self.sampling[
                "transverse_energy_sampling_variance"][0] += (
                    variance_factor * xp.sum(transverse_item**2))
            self.sampling[
                "longitudinal_energy_sampling_variance"][0] += (
                    variance_factor * xp.sum(longitudinal_item**2))

        w_perp = (
            event["weight"] * event["p_perp"] * event["dt_observer"])
        w_parallel = (
            event["weight"] * event["p_parallel"] * event["dt_observer"])
        transverse_energy = xp.sum(w_perp)
        self.accounting["transverse_energy"][0] += transverse_energy
        self.accounting["longitudinal_energy"][0] += xp.sum(w_parallel)
        self.accounting[
            "energy_truncated_by_spectral_closure"][0] += (
                transverse_energy * self.spectral_truncated_fraction)
        self._accumulate_coordinate_products(event, w_perp, xp)

        if "observer_time" in self.enabled_channels:
            self._accumulate_detectors(event, xp)

        if self.energy_edges is None and not self.needs_spectral_samples:
            return

        spectral_mask = (w_perp > 0.0) & (event["omega_c"] > 0.0)
        spectral_event = self._filtered(event, spectral_mask)
        spectral_weight = w_perp[spectral_mask]
        retained_fraction = self.spectral_cdf[-1]
        if self.energy_edges is not None:
            scale = hbar * spectral_event["omega_c"]
            below = self._cdf_at(self.energy_edges[0] / scale, xp)
            above = 1.0 - self._cdf_at(self.energy_edges[-1] / scale, xp)
            self.accounting["energy_below_range"][0] += xp.sum(
                spectral_weight * below)
            self.accounting["energy_within_range"][0] += xp.sum(
                spectral_weight * (1.0 - below - above))
            self.accounting["energy_above_range"][0] += xp.sum(
                spectral_weight * above)

        if not self.needs_spectral_samples:
            return
        aperture_hits = {
            detector["name"]: xp.zeros(
                spectral_weight.shape, dtype=xp.float64)
            for detector in self.detectors
            if detector["half_angle"] > 0.0
        }
        for sample_index in range(self.samples_per_particle):
            jitter = _event_uniform(
                self.random_seed, spectral_event["particle_id"],
                spectral_event["event_index"], _RANDOM_STREAM_PHOTON_ENERGY,
                sample_index, xp, self.random_namespace)
            spectral_probability = (
                (sample_index + jitter) / self.samples_per_particle
                * retained_fraction)
            scaled_energy = self._inverse_cdf(spectral_probability, xp)
            packet_weight = (
                spectral_weight * retained_fraction
                / self.samples_per_particle)
            self._accumulate_sample(
                spectral_event, scaled_energy, packet_weight, xp,
                sample_index, aperture_hits=aperture_hits)

        total_packet_energy = spectral_weight * retained_fraction
        for name, hits in aperture_hits.items():
            probability_estimate = hits / self.samples_per_particle
            if self.samples_per_particle > 1:
                packet_variance = (
                    total_packet_energy**2
                    * probability_estimate * (1.0 - probability_estimate)
                    / (self.samples_per_particle - 1))
            else:
                # With one packet there is no within-event replicate. Report
                # the conservative Bernoulli upper bound rather than a false
                # zero uncertainty.
                packet_variance = 0.25 * total_packet_energy**2
            estimates = {
                "aperture": total_packet_energy * probability_estimate,
                "outside_aperture": total_packet_energy
                * (1.0 - probability_estimate),
            }
            thinning_probability = self.particle_sampling_fraction
            for estimator, estimate in estimates.items():
                # Combine within-event packet variance with the Bernoulli
                # Horvitz-Thompson contribution from particle thinning. This
                # reduces to packet variance when all particles are retained.
                variance = (
                    thinning_probability * packet_variance
                    + (1.0 - thinning_probability) * estimate**2)
                key = (
                    "stochastic_spectral_angular_%s_energy_"
                    "sampling_variance/%s" % (estimator, name))
                self.sampling[key][0] += xp.sum(variance)

    def accumulate_impulse_batch(
            self, lower_momentum, particle_slice, batch_count,
            simulation_time, event_index):
        """Consume one pusher-owned endpoint batch immediately."""
        if lower_momentum is None or len(lower_momentum) != 3:
            raise ValueError(
                "A pusher impulse requires three lower momentum arrays.")
        batch_count = int(batch_count)
        if batch_count < 0 or any(
                int(array.size) < batch_count for array in lower_momentum):
            raise ValueError("The pusher endpoint batch is undersized.")
        self._accumulate_batch(
            lower_momentum, particle_slice, simulation_time, self._xp(),
            batch_count=batch_count, event_index=int(event_index))

    def complete_impulse(self, simulation_time, event_index):
        """Complete one streamed event, recording timing exactly once."""
        event_index = int(event_index)
        if (self.last_completed_event_index is not None
                and event_index <= self.last_completed_event_index):
            raise RuntimeError(
                "Radiation pusher events must complete once in increasing "
                "event-index order.")
        self._record_event_timing(float(simulation_time), event_index)
        self.completed_event_count += 1
        self.last_completed_event_index = event_index

    def accumulate_impulse(
            self, lower_momentum, simulation_time=0.0, event_index=None):
        """Compatibility path for caller-owned complete endpoint arrays."""
        if lower_momentum is None or len(lower_momentum) != 3:
            raise ValueError(
                "A pusher impulse requires three lower momentum arrays.")
        particle_count = int(self.eon.Ntot)
        if any(int(array.size) != particle_count for array in lower_momentum):
            raise ValueError(
                "Pusher endpoint arrays changed before radiation accumulation.")
        if event_index is None:
            event_index = int(round(float(simulation_time) / self.dt_sim))
        xp = self._xp()
        for start in range(0, particle_count, self.particle_batch_size):
            stop = min(start + self.particle_batch_size, particle_count)
            self._accumulate_batch(
                lower_momentum, slice(start, stop), simulation_time, xp,
                event_index=event_index)
        self.complete_impulse(simulation_time, event_index)

    def accumulate(self, simulation_time=0.0):
        """Reject accumulation without explicit pusher endpoints."""
        raise RuntimeError(
            "Use accumulate_impulse with a completed momentum-push interval.")

    def snapshot(self, interval_moments=False):
        """Return a host-side copy of cumulative or interval-only state."""
        if self._on_gpu:
            raise RuntimeError("Receive observer radiation from the GPU first.")
        if interval_moments:
            # Dense products and additive counters are differenced by the
            # writer. Do not duplicate them merely to obtain the mergeable
            # interval moment, quality, and timing state.
            return {
                "quality": {
                    key: value.copy()
                    for key, value in self.interval_quality.items()},
                "resolution": {
                    key: value.copy()
                    for key, value in self.interval_resolution_stats.items()},
                "moments": {
                    key: value.copy()
                    for key, value in self.interval_moment_stats.items()},
                "timing": dict(self.interval_timing),
            }
        return {
            "data": {key: value.copy() for key, value in self.data.items()},
            "accounting": {
                key: value.copy() for key, value in self.accounting.items()},
            "sampling": {
                key: value.copy() for key, value in self.sampling.items()},
            "source_z": {
                key: value.copy()
                for key, value in self.source_z_histograms.items()},
            "quality": {
                key: value.copy() for key, value in self.quality.items()},
            "resolution": {
                key: value.copy()
                for key, value in self.resolution_stats.items()},
            "moments": {
                key: value.copy()
                for key, value in self.moment_stats.items()},
            "timing": dict(self.cumulative_timing),
        }

    def reset_interval_state(self):
        """Reset non-subtractable centered moments after a successful write."""
        for stats in self.interval_moment_stats.values():
            stats.fill(0.0)
        for value in self.interval_quality.values():
            value.fill(0.0)
        for value in self.interval_resolution_stats.values():
            value.fill(0.0)
        self.interval_timing = self._empty_timing()



def _as_edges_like_grid(values, name):
    """Validate a strictly increasing interpolation grid (not bin edges)."""
    grid = np.asarray(values, dtype=np.float64)
    if (grid.ndim != 1 or grid.size < 2
            or not np.all(np.isfinite(grid))
            or not np.all(np.diff(grid) > 0.0)):
        raise ValueError("`%s` must be a strictly increasing 1-D grid." % name)
    return grid.copy()

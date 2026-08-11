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
import re
from functools import lru_cache

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, m_e
from scipy.special import kv

from fbpic.utils.cuda import cuda_installed

if cuda_installed:
    import cupy


_POWER_FACTOR = e**2 / (6.0 * math.pi * epsilon_0 * c)
_ANGULAR_POWER_FACTOR = e**2 / (16.0 * math.pi**2 * epsilon_0 * c)
_E_MC = e / (m_e * c)
_PROJECTED_ANGLE_LIMIT = 0.5 * math.pi

_SOURCE_STAT_SIZE = 30

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
def _cached_angular_kernel(maximum_scaled_energy):
    """Tabulate and cache the Schwinger vertical-angle inverse CDF."""
    x_grid = np.geomspace(1.0e-8, maximum_scaled_energy, 96)
    probability = np.linspace(0.0, 1.0, 257)
    q_grid = np.linspace(0.0, 8.0, 1025)
    inverse = np.empty((x_grid.size, probability.size), dtype=np.float64)
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
        cdf /= cdf[-1]
        inverse[index] = np.interp(probability, cdf, q_grid)
    log_x = np.log(x_grid)
    for array in (x_grid, log_x, probability, inverse):
        array.setflags(write=False)
    return x_grid, log_x, probability, inverse


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


def _normalize_detector(detector, index, common_time_edges):
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
    rays, ray_weights = _cone_quadrature(
        direction, half_angle, quadrature)
    band_values = item.get("energy_bands", [])
    if isinstance(band_values, dict):
        band_values = [band_values]
    bands = [
        _normalize_band(band, band_index)
        for band_index, band in enumerate(band_values)
    ]
    _require_unique(bands, "Detector energy-band")
    interval = item.get("pulse_interval", (0.05, 0.95))
    if len(interval) != 2 or not (0.0 <= interval[0] < interval[1] <= 1.0):
        raise ValueError("`pulse_interval` must contain two ordered fractions.")
    return {
        "name": name,
        "direction": direction,
        "time_edges": time_edges,
        "half_angle": half_angle,
        "rays": rays,
        "ray_weights": ray_weights,
        "energy_bands": bands,
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
            samples_per_particle=1, gamma_cutoff=10.0, particle_selection=None,
            particle_batch_size=262144):
        self.eon = radiating_species
        self.use_cuda = radiating_species.use_cuda
        self.dt_sim = float(dt_sim)
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
            if source_moments:
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
            _normalize_detector(detector, index, self.common_time_edges)
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
        for projection in self.source_projections:
            projection["deterministic"] = bool(
                set(projection["axes"]) <= {"x", "y", "z"}
                and set(projection["selection"]) <= coordinate_selection_keys)
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
            selection["deterministic"] = bool(
                set(quantities) == {"position"}
                and set(selection) <= (
                    {"name", "quantities"} | coordinate_selection_keys))
            normalized_selections.append(selection)
        self.source_moment_selections = normalized_selections
        _require_unique(self.source_moment_selections, "Source moment selection")

        self.data = {}
        self.data_axes = {}
        self.data_kinds = {}
        self.accounting = {}
        self.moment_stats = {}
        self._initialize_storage()
        self.angular_kernel_x = None
        self.angular_kernel_log_x = None
        self.angular_kernel_probability = None
        self.angular_kernel_inverse = None
        if self.needs_spectral_samples:
            self._initialize_angular_kernel()
        self._runtime = self._all_runtime_arrays()
        self._on_gpu = False

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
                self.moment_stats[selection["name"]] = np.zeros(
                    _SOURCE_STAT_SIZE, dtype=np.float64)

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
        if self.needs_spectral_samples and any(
                detector["half_angle"] > 0.0
                for detector in self.detectors):
            base_accounting.append("energy_outside_apertures")
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
    def needs_spectral_samples(self):
        return bool(
            "angular_spectral" in self.enabled_channels
            or ("source" in self.enabled_channels and any(
                not projection["deterministic"]
                for projection in self.source_projections))
            or ("source_moments" in self.enabled_channels and any(
                not selection["deterministic"]
                for selection in self.source_moment_selections))
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

    def _observer_event(self, eon, simulation_time, xp, particle_slice=None):
        """Transform the complete PIC event tuple available to the process.

        Position and momentum are at the particle half step; fields are those
        gathered at the preceding integer step.
        """
        if particle_slice is None:
            particle_slice = slice(None)
        gamma_sim = 1.0 / eon.inv_gamma[particle_slice]
        ux_sim = eon.ux[particle_slice]
        uy_sim = eon.uy[particle_slice]
        uz_sim = eon.uz[particle_slice]
        transverse_mass2 = 1.0 + ux_sim**2 + uy_sim**2
        p_plus_sim = xp.where(
            uz_sim >= 0.0, gamma_sim + uz_sim,
            transverse_mass2 / (gamma_sim - uz_sim))
        p_minus_sim = transverse_mass2 / p_plus_sim
        lightfront_boost = self.gamma_boost * (1.0 + self.beta_boost)
        p_plus = lightfront_boost * p_plus_sim
        p_minus = p_minus_sim / lightfront_boost
        gamma = 0.5 * (p_plus + p_minus)
        uz = 0.5 * (p_plus - p_minus)
        ux, uy = ux_sim, uy_sim
        inv_gamma = 1.0 / gamma
        dt_ratio = gamma / gamma_sim

        Ex_sim = eon.Ex[particle_slice]
        Ey_sim = eon.Ey[particle_slice]
        Ez_sim = eon.Ez[particle_slice]
        cBx_sim = c * eon.Bx[particle_slice]
        cBy_sim = c * eon.By[particle_slice]
        cBz = c * eon.Bz[particle_slice]
        ex_plus_cby = lightfront_boost * (Ex_sim + cBy_sim)
        ex_minus_cby = (Ex_sim - cBy_sim) / lightfront_boost
        ey_minus_cbx = lightfront_boost * (Ey_sim - cBx_sim)
        ey_plus_cbx = (Ey_sim + cBx_sim) / lightfront_boost
        Ex = 0.5 * (ex_plus_cby + ex_minus_cby)
        cBy = 0.5 * (ex_plus_cby - ex_minus_cby)
        Ey = 0.5 * (ey_plus_cbx + ey_minus_cbx)
        cBx = 0.5 * (ey_plus_cbx - ey_minus_cbx)
        Ez = Ez_sim

        beta_x = ux * inv_gamma
        beta_y = uy * inv_gamma
        beta_z = uz * inv_gamma
        beta2 = beta_x**2 + beta_y**2 + beta_z**2
        beta_abs = xp.sqrt(beta2)

        du_x = -_E_MC * (Ex + beta_y * cBz - beta_z * cBy)
        du_y = -_E_MC * (Ey + beta_z * cBx - beta_x * cBz)
        du_z = -_E_MC * (Ez + beta_x * cBy - beta_y * cBx)
        dgamma = -_E_MC * (
            beta_x * Ex + beta_y * Ey + beta_z * Ez)
        cross_x = beta_y * du_z - beta_z * du_y
        cross_y = beta_z * du_x - beta_x * du_z
        cross_z = beta_x * du_y - beta_y * du_x
        cross2 = cross_x**2 + cross_y**2 + cross_z**2
        safe_beta2 = xp.where(beta2 > 0.0, beta2, 1.0)
        p_perp = _POWER_FACTOR * gamma**2 * cross2 / safe_beta2
        p_parallel = _POWER_FACTOR * dgamma**2 / safe_beta2
        p_perp = xp.where(beta2 > 0.0, p_perp, 0.0)
        p_parallel = xp.where(beta2 > 0.0, p_parallel, 0.0)
        omega_c = 1.5 * gamma**2 * xp.sqrt(cross2) / (
            xp.where(beta_abs > 0.0, beta_abs**3, 1.0))
        omega_c = xp.where(cross2 > 0.0, omega_c, 0.0)

        dot_beta_x = (du_x - beta_x * dgamma) * inv_gamma
        dot_beta_y = (du_y - beta_y * dgamma) * inv_gamma
        dot_beta_z = (du_z - beta_z * dgamma) * inv_gamma
        parallel_coefficient = dgamma / safe_beta2
        dot_beta_perp_x = (du_x - beta_x * parallel_coefficient) * inv_gamma
        dot_beta_perp_y = (du_y - beta_y * parallel_coefficient) * inv_gamma
        dot_beta_perp_z = (du_z - beta_z * parallel_coefficient) * inv_gamma
        dot_beta_perp_x = xp.where(beta2 > 0.0, dot_beta_perp_x, 0.0)
        dot_beta_perp_y = xp.where(beta2 > 0.0, dot_beta_perp_y, 0.0)
        dot_beta_perp_z = xp.where(beta2 > 0.0, dot_beta_perp_z, 0.0)

        event = {
            "ux": ux, "uy": uy, "uz": uz, "gamma": gamma,
            "inv_gamma": inv_gamma,
            "beta_x": beta_x, "beta_y": beta_y, "beta_z": beta_z,
            "beta2": beta2, "beta_abs": beta_abs,
            "du_x": du_x, "du_y": du_y, "du_z": du_z,
            "dgamma": dgamma,
            "dot_beta_x": dot_beta_x, "dot_beta_y": dot_beta_y,
            "dot_beta_z": dot_beta_z,
            "dot_beta_perp_x": dot_beta_perp_x,
            "dot_beta_perp_y": dot_beta_perp_y,
            "dot_beta_perp_z": dot_beta_perp_z,
            "p_perp": p_perp, "p_parallel": p_parallel,
            "omega_c": omega_c, "dt_ratio": dt_ratio,
            "weight": eon.w[particle_slice],
            "particle_theta_x": xp.arctan2(ux, uz),
            "particle_theta_y": xp.arctan2(uy, uz),
        }

        if self.needs_positions:
            x_sim = eon.x[particle_slice]
            y_sim = eon.y[particle_slice]
            z_sim = eon.z[particle_slice]
            ct_sim = c * simulation_time
            event_plus = lightfront_boost * (ct_sim + z_sim)
            event_minus = (ct_sim - z_sim) / lightfront_boost
            ct_observer = 0.5 * (event_plus + event_minus)
            z_observer = 0.5 * (event_plus - event_minus)
            event["x"] = x_sim + self.observer_translation[1]
            event["y"] = y_sim + self.observer_translation[2]
            event["z"] = z_observer + self.observer_translation[3]
            event["time"] = (
                ct_observer + self.observer_translation[0]) / c
        return event

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

    def _accumulate_detectors(self, event, xp):
        for detector in self.detectors:
            name = detector["name"]
            prefix = "detector/%s" % name
            direction = self._runtime[prefix + "/direction"]
            time_edges = self._runtime[prefix + "/time"]
            d_power = self._angular_power(direction, event, xp)
            emitted = d_power * event["weight"] * self.dt_sim \
                * event["dt_ratio"]
            tau = event["time"] - (
                direction[0] * event["x"]
                + direction[1] * event["y"]
                + direction[2] * event["z"]) / c
            key = prefix + "/direction"
            represented = self._histogram_add(
                self.data[key], (tau,), (time_edges,), emitted, xp)
            account_key = "represented/%s_energy_per_solid_angle" % key
            self.accounting[account_key][0] += represented

            if detector["energy_bands"]:
                curvature_angular_power = self._angular_power(
                    direction, event, xp, transverse_only=True)
                curvature_emitted = (
                    curvature_angular_power * event["weight"] * self.dt_sim
                    * event["dt_ratio"])
            for band in detector["energy_bands"]:
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
                ray_energy = (ray_power * ray_weights[ray_index]
                              * event["weight"] * self.dt_sim
                              * event["dt_ratio"])
                ray_tau = event["time"] - (
                    ray[0] * event["x"] + ray[1] * event["y"]
                    + ray[2] * event["z"]) / c
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
                        * event["weight"] * self.dt_sim * event["dt_ratio"])
                for band in detector["energy_bands"]:
                    fraction = self._band_fraction(
                        event, band["energy_range"], xp)
                    band_key = prefix + "/band/%s/aperture" % band["name"]
                    represented = self._histogram_add(
                        self.data[band_key], (ray_tau,), (time_edges,),
                        ray_curvature_energy * fraction, xp)
                    self.accounting[
                        "represented/%s_energy" % band_key][0] += represented

    def _sample_direction(self, event, scaled_energy, xp):
        projection = (
            event["beta_x"] * event["du_x"]
            + event["beta_y"] * event["du_y"]
            + event["beta_z"] * event["du_z"]) / event["beta2"]
        plane_x = event["du_x"] - projection * event["beta_x"]
        plane_y = event["du_y"] - projection * event["beta_y"]
        plane_z = event["du_z"] - projection * event["beta_z"]
        plane_norm = xp.sqrt(plane_x**2 + plane_y**2 + plane_z**2)
        plane_x /= plane_norm
        plane_y /= plane_norm
        plane_z /= plane_norm
        velocity_x = event["beta_x"] / event["beta_abs"]
        velocity_y = event["beta_y"] / event["beta_abs"]
        velocity_z = event["beta_z"] / event["beta_abs"]
        normal_x = velocity_y * plane_z - velocity_z * plane_y
        normal_y = velocity_z * plane_x - velocity_x * plane_z
        normal_z = velocity_x * plane_y - velocity_y * plane_x

        # Invert the tabulated conditional CDF with bilinear interpolation in
        # log(x) and probability.  The distribution is symmetric about the
        # orbit plane, hence a separate random sign.  The local tangent fixes
        # the in-plane direction; this is the standard local synchrotron
        # (strong-wiggler) closure rather than a trajectory-phase solver.
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

        random_probability = xp.random.random(scaled_energy.size)
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
        sign = xp.where(xp.random.random(psi.size) < 0.5, -1.0, 1.0)
        sin_psi = sign * xp.sin(psi)
        cos_psi = xp.cos(psi)
        return (
            cos_psi * velocity_x + sin_psi * normal_x,
            cos_psi * velocity_y + sin_psi * normal_y,
            cos_psi * velocity_z + sin_psi * normal_z,
        )

    @staticmethod
    def _update_moments(stats, values, weights, quantities, xp):
        stats[0] += xp.sum(weights)
        coordinates = None
        if "position" in quantities:
            coordinates = tuple(values[axis] for axis in ("x", "y", "z"))
            for i, coordinate in enumerate(coordinates):
                stats[1 + i] += xp.sum(weights * coordinate)
            for i, first in enumerate(coordinates):
                for j, second in enumerate(coordinates):
                    stats[4 + 3 * i + j] += xp.sum(
                        weights * first * second)
        if "angle" in quantities:
            angles = (values["theta_x"], values["theta_y"])
            for i, angle in enumerate(angles):
                stats[13 + i] += xp.sum(weights * angle)
            for i, first in enumerate(angles):
                for j, second in enumerate(angles):
                    stats[15 + 2 * i + j] += xp.sum(
                        weights * first * second)
            if coordinates is not None:
                for i, coordinate in enumerate(coordinates):
                    for j, angle in enumerate(angles):
                        stats[19 + 2 * i + j] += xp.sum(
                            weights * coordinate * angle)
        if "time" in quantities:
            tau = values["time"]
            stats[25] += xp.sum(weights * tau)
            stats[26] += xp.sum(weights * tau**2)
            if coordinates is not None:
                for i, coordinate in enumerate(coordinates):
                    stats[27 + i] += xp.sum(weights * coordinate * tau)

    def _accumulate_sample(self, event, scaled_energy, packet_weight, xp):
        nx, ny, nz = self._sample_direction(event, scaled_energy, xp)
        theta_x = xp.arctan2(nx, nz)
        theta_y = xp.arctan2(ny, nz)
        energy = hbar * event["omega_c"] * scaled_energy
        values = {
            "theta_x": theta_x, "theta_y": theta_y,
            "energy": energy,
            "nx": nx, "ny": ny, "nz": nz,
        }
        if "source" in self.enabled_channels \
                or "source_moments" in self.enabled_channels:
            tau = event["time"] - (
                nx * event["x"] + ny * event["y"] + nz * event["z"]) / c
            values.update({
                "x": event["x"], "y": event["y"], "z": event["z"],
                "time": tau,
            })

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

        apertures = [
            detector for detector in self.detectors
            if "observer_time" in self.enabled_channels
            and detector["half_angle"] > 0.0
        ]
        if apertures and "energy_outside_apertures" in self.accounting:
            inside_any = xp.zeros(packet_weight.shape, dtype=xp.bool_)
            for detector in apertures:
                direction = detector["direction"]
                inside_any |= (
                    nx * direction[0] + ny * direction[1]
                    + nz * direction[2]
                ) >= math.cos(detector["half_angle"])
            self.accounting["energy_outside_apertures"][0] += xp.sum(
                packet_weight[~inside_any])

        if "source" in self.enabled_channels:
            for projection in self.source_projections:
                if projection["deterministic"]:
                    continue
                selected = self._selection_mask(
                    projection["selection"], values, xp)
                key = "source/%s" % projection["name"]
                runtime_edges = tuple(
                    self._runtime["source_axis/%s" % axis]
                    for axis in projection["axes"])
                represented = self._histogram_add(
                    self.data[key],
                    tuple(values[axis][selected]
                          for axis in projection["axes"]),
                    runtime_edges, packet_weight[selected], xp)
                self.accounting[
                    "represented/%s_energy" % key][0] += represented

        if "source_moments" in self.enabled_channels:
            for selection in self.source_moment_selections:
                if selection["deterministic"]:
                    continue
                selected = self._selection_mask(selection, values, xp)
                selected_values = {
                    key: value[selected] for key, value in values.items()}
                self._update_moments(
                    self.moment_stats[selection["name"]], selected_values,
                    packet_weight[selected], selection["quantities"], xp)

    def _accumulate_coordinate_products(self, event, w_perp, xp):
        """Accumulate source products that need no photon packet."""
        values = {axis: event[axis] for axis in ("x", "y", "z")}
        if "source" in self.enabled_channels:
            for projection in self.source_projections:
                if not projection["deterministic"]:
                    continue
                selected = self._selection_mask(
                    projection["selection"], values, xp)
                key = "source/%s" % projection["name"]
                runtime_edges = tuple(
                    self._runtime["source_axis/%s" % axis]
                    for axis in projection["axes"])
                represented = self._histogram_add(
                    self.data[key],
                    tuple(values[axis][selected]
                          for axis in projection["axes"]),
                    runtime_edges, w_perp[selected], xp)
                self.accounting[
                    "represented/%s_energy" % key][0] += represented
        if "source_moments" in self.enabled_channels:
            for selection in self.source_moment_selections:
                if not selection["deterministic"]:
                    continue
                selected = self._selection_mask(selection, values, xp)
                selected_values = {
                    key: value[selected] for key, value in values.items()}
                self._update_moments(
                    self.moment_stats[selection["name"]], selected_values,
                    w_perp[selected], selection["quantities"], xp)

    def _accumulate_batch(self, particle_slice, simulation_time, xp):
        eon = self.eon
        event = self._observer_event(
            eon, float(simulation_time), xp, particle_slice)
        mask = self._particle_mask(event, xp)
        event = self._filtered(event, mask)

        dt_observer = self.dt_sim * event["dt_ratio"]
        w_perp = event["weight"] * event["p_perp"] * dt_observer
        w_parallel = event["weight"] * event["p_parallel"] * dt_observer
        self.accounting["transverse_energy"][0] += xp.sum(w_perp)
        self.accounting["longitudinal_energy"][0] += xp.sum(w_parallel)
        self.accounting[
            "energy_truncated_by_spectral_closure"][0] += (
                xp.sum(w_perp) * self.spectral_truncated_fraction)
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
        for sample_index in range(self.samples_per_particle):
            probability = (
                sample_index + xp.random.random(spectral_weight.size)
            ) / self.samples_per_particle * retained_fraction
            scaled_energy = self._inverse_cdf(probability, xp)
            packet_weight = (
                spectral_weight * retained_fraction
                / self.samples_per_particle
            )
            self._accumulate_sample(
                spectral_event, scaled_energy, packet_weight, xp)

    def accumulate(self, simulation_time=0.0):
        """Accumulate one available PIC event tuple in bounded batches."""
        if self.eon.Ntot == 0:
            return
        xp = self._xp()
        for start in range(0, self.eon.Ntot, self.particle_batch_size):
            stop = min(start + self.particle_batch_size, self.eon.Ntot)
            self._accumulate_batch(
                slice(start, stop), simulation_time, xp)

    def snapshot(self):
        """Return a host-side copy of all additive accumulator state."""
        if self._on_gpu:
            raise RuntimeError("Receive observer radiation from the GPU first.")
        return {
            "data": {key: value.copy() for key, value in self.data.items()},
            "accounting": {
                key: value.copy() for key, value in self.accounting.items()},
            "moments": {
                key: value.copy() for key, value in self.moment_stats.items()},
        }


def _as_edges_like_grid(values, name):
    """Validate a strictly increasing interpolation grid (not bin edges)."""
    grid = np.asarray(values, dtype=np.float64)
    if (grid.ndim != 1 or grid.size < 2
            or not np.all(np.isfinite(grid))
            or not np.all(np.diff(grid) > 0.0)):
        raise ValueError("`%s` must be a strictly increasing 1-D grid." % name)
    return grid.copy()

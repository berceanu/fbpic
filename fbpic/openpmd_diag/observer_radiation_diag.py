# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""openPMD-compatible output for observer-frame synchrotron products."""

import json
import math
import os

import numpy as np
from scipy.constants import c

from fbpic.utils.mpi import comm as comm_simple


_DIMENSIONLESS = np.zeros(7, dtype=np.float64)
_ENERGY_DIMENSION = np.array([2., 1., -2., 0., 0., 0., 0.])
_POWER_DIMENSION = np.array([2., 1., -3., 0., 0., 0., 0.])
_LENGTH_DIMENSION = np.array([1., 0., 0., 0., 0., 0., 0.])
_TIME_DIMENSION = np.array([0., 0., 1., 0., 0., 0., 0.])


def _bytes(value):
    return np.bytes_(str(value))


def _clean(value):
    return str(value).replace("/", "_").replace("-", "_")


def _subtract_snapshot(current, previous):
    return {
        category: {
            key: current[category][key] - previous[category][key]
            for key in current[category]
        }
        for category in ("data", "accounting", "moments")
    }


def _zero_snapshot_like(snapshot):
    return {
        category: {
            key: np.zeros_like(value)
            for key, value in snapshot[category].items()
        }
        for category in ("data", "accounting", "moments")
    }


def _direction_from_projected_angles(theta_x, theta_y):
    tangent_x = math.tan(theta_x)
    tangent_y = math.tan(theta_y)
    vector = np.array([tangent_x, tangent_y, 1.0])
    return vector / np.linalg.norm(vector)


def _triangle_solid_angle(a, b, d):
    numerator = abs(np.dot(a, np.cross(b, d)))
    denominator = 1.0 + np.dot(a, b) + np.dot(b, d) + np.dot(d, a)
    return 2.0 * math.atan2(numerator, denominator)


def angular_cell_measure(theta_x_edges, theta_y_edges, measure):
    """Return dtheta_x dtheta_y or exact spherical quadrilateral areas."""
    theta_x_edges = np.asarray(theta_x_edges, dtype=np.float64)
    theta_y_edges = np.asarray(theta_y_edges, dtype=np.float64)
    limit = 0.5 * math.pi
    for name, edges in (
            ("theta_x_edges", theta_x_edges),
            ("theta_y_edges", theta_y_edges)):
        if (edges.ndim != 1 or edges.size < 2
                or not np.all(np.isfinite(edges))
                or not np.all(np.diff(edges) > 0.0)
                or edges[0] <= -limit or edges[-1] >= limit):
            raise ValueError(
                "`%s` must be strictly increasing and lie inside "
                "(-pi/2, pi/2)." % name)
    if measure not in ("solid_angle", "projected_angles"):
        raise ValueError("Unknown angular cell measure `%s`." % measure)
    if measure == "projected_angles":
        return np.diff(theta_x_edges)[:, None] * np.diff(theta_y_edges)[None, :]
    output = np.empty(
        (theta_x_edges.size - 1, theta_y_edges.size - 1),
        dtype=np.float64)
    for ix in range(output.shape[0]):
        for iy in range(output.shape[1]):
            lower_left = _direction_from_projected_angles(
                theta_x_edges[ix], theta_y_edges[iy])
            lower_right = _direction_from_projected_angles(
                theta_x_edges[ix + 1], theta_y_edges[iy])
            upper_right = _direction_from_projected_angles(
                theta_x_edges[ix + 1], theta_y_edges[iy + 1])
            upper_left = _direction_from_projected_angles(
                theta_x_edges[ix], theta_y_edges[iy + 1])
            output[ix, iy] = (
                _triangle_solid_angle(lower_left, lower_right, upper_right)
                + _triangle_solid_angle(lower_left, upper_right, upper_left))
    return output


def _moment_component_selected(name, quantities):
    """Return whether a derived component belongs to requested quantities."""
    quantities = set(quantities)
    if name == "energy":
        return True
    if "observer_time" in name:
        needs_position = name.startswith(("covariance_", "correlation_"))
        return "time" in quantities and (
            not needs_position or "position" in quantities)
    if "theta_" in name:
        needs_position = name.startswith((
            "covariance_x_", "covariance_y_", "covariance_z_",
            "correlation_x_", "correlation_y_", "correlation_z_",
        ))
        return "angle" in quantities and (
            not needs_position or "position" in quantities)
    return "position" in quantities


def _source_moment_dimensions(quantities=("position", "angle", "time")):
    dimensions = {"energy": _ENERGY_DIMENSION}
    for axis in "xyz":
        dimensions["centroid_%s" % axis] = _LENGTH_DIMENSION
        dimensions["rms_%s" % axis] = _LENGTH_DIMENSION
    for first_index, first in enumerate("xyz"):
        for second_index, second in enumerate("xyz"):
            if second_index >= first_index:
                dimensions["covariance_%s%s" % (first, second)] = \
                    2.0 * _LENGTH_DIMENSION
    dimensions.update({
        "longitudinal_emission_extent": _LENGTH_DIMENSION,
        "transverse_major_rms": _LENGTH_DIMENSION,
        "transverse_minor_rms": _LENGTH_DIMENSION,
        "transverse_ellipticity": _DIMENSIONLESS,
        "transverse_orientation": _DIMENSIONLESS,
        "principal_axis_x": _DIMENSIONLESS,
        "principal_axis_y": _DIMENSIONLESS,
        "centroid_theta_x": _DIMENSIONLESS,
        "centroid_theta_y": _DIMENSIONLESS,
        "rms_theta_x": _DIMENSIONLESS,
        "rms_theta_y": _DIMENSIONLESS,
        "covariance_theta_x_theta_x": _DIMENSIONLESS,
        "covariance_theta_x_theta_y": _DIMENSIONLESS,
        "covariance_theta_y_theta_y": _DIMENSIONLESS,
        "centroid_observer_time": _TIME_DIMENSION,
        "rms_observer_time": _TIME_DIMENSION,
    })
    for position_axis in "xyz":
        for angle_axis in ("theta_x", "theta_y"):
            dimensions["covariance_%s_%s" % (
                position_axis, angle_axis)] = _LENGTH_DIMENSION
            dimensions["correlation_%s_%s" % (
                position_axis, angle_axis)] = _DIMENSIONLESS
        dimensions["covariance_%s_observer_time" % position_axis] = \
            _LENGTH_DIMENSION + _TIME_DIMENSION
        dimensions["correlation_%s_observer_time" % position_axis] = \
            _DIMENSIONLESS
    return {
        name: dimension for name, dimension in dimensions.items()
        if _moment_component_selected(name, quantities)
    }


def source_moment_components(
        stats, quantities=("position", "angle", "time")):
    """Derive physical source observables from additive sufficient statistics."""
    stats = np.asarray(stats, dtype=np.float64)
    weight = stats[0]
    energy_item = (float(weight), _ENERGY_DIMENSION)
    if not (weight > 0.0):
        empty = {
            name: (np.nan, dimension)
            for name, dimension in _source_moment_dimensions(
                quantities).items()}
        empty["energy"] = energy_item
        return empty

    mean_x = stats[1:4] / weight
    second_x = stats[4:13].reshape(3, 3) / weight
    covariance_x = 0.5 * (
        second_x + second_x.T) - np.outer(mean_x, mean_x)
    covariance_x[np.diag_indices(3)] = np.maximum(
        covariance_x.diagonal(), 0.0)
    rms_x = np.sqrt(covariance_x.diagonal())

    mean_angle = stats[13:15] / weight
    second_angle = stats[15:19].reshape(2, 2) / weight
    covariance_angle = 0.5 * (
        second_angle + second_angle.T) - np.outer(mean_angle, mean_angle)
    covariance_angle[np.diag_indices(2)] = np.maximum(
        covariance_angle.diagonal(), 0.0)
    rms_angle = np.sqrt(covariance_angle.diagonal())

    covariance_x_angle = (
        stats[19:25].reshape(3, 2) / weight
        - np.outer(mean_x, mean_angle))
    mean_time = stats[25] / weight
    variance_time = max(stats[26] / weight - mean_time**2, 0.0)
    rms_time = math.sqrt(variance_time)
    covariance_x_time = stats[27:30] / weight - mean_x * mean_time

    transverse_values, transverse_vectors = np.linalg.eigh(
        covariance_x[:2, :2])
    minor_variance = max(transverse_values[0], 0.0)
    major_variance = max(transverse_values[1], 0.0)
    minor_rms = math.sqrt(minor_variance)
    major_rms = math.sqrt(major_variance)
    major_vector = transverse_vectors[:, 1]
    if major_vector[0] < 0.0 or (
            major_vector[0] == 0.0 and major_vector[1] < 0.0):
        major_vector = -major_vector
    orientation = math.atan2(major_vector[1], major_vector[0])
    ellipticity = 0.0
    if major_rms + minor_rms > 0.0:
        ellipticity = (major_rms - minor_rms) / (major_rms + minor_rms)

    output = {"energy": energy_item}
    for index, axis in enumerate("xyz"):
        output["centroid_%s" % axis] = (
            float(mean_x[index]), _LENGTH_DIMENSION)
        output["rms_%s" % axis] = (
            float(rms_x[index]), _LENGTH_DIMENSION)
    for i, first in enumerate("xyz"):
        for j, second in enumerate("xyz"):
            if j < i:
                continue
            output["covariance_%s%s" % (first, second)] = (
                float(covariance_x[i, j]), 2.0 * _LENGTH_DIMENSION)
    output["longitudinal_emission_extent"] = (
        float(rms_x[2]), _LENGTH_DIMENSION)
    output["transverse_major_rms"] = (major_rms, _LENGTH_DIMENSION)
    output["transverse_minor_rms"] = (minor_rms, _LENGTH_DIMENSION)
    output["transverse_ellipticity"] = (ellipticity, _DIMENSIONLESS)
    output["transverse_orientation"] = (orientation, _DIMENSIONLESS)
    output["principal_axis_x"] = (float(major_vector[0]), _DIMENSIONLESS)
    output["principal_axis_y"] = (float(major_vector[1]), _DIMENSIONLESS)

    for index, axis in enumerate(("theta_x", "theta_y")):
        output["centroid_%s" % axis] = (
            float(mean_angle[index]), _DIMENSIONLESS)
        output["rms_%s" % axis] = (
            float(rms_angle[index]), _DIMENSIONLESS)
    output["covariance_theta_x_theta_x"] = (
        float(covariance_angle[0, 0]), _DIMENSIONLESS)
    output["covariance_theta_x_theta_y"] = (
        float(covariance_angle[0, 1]), _DIMENSIONLESS)
    output["covariance_theta_y_theta_y"] = (
        float(covariance_angle[1, 1]), _DIMENSIONLESS)

    for i, position_axis in enumerate("xyz"):
        for j, angle_axis in enumerate(("theta_x", "theta_y")):
            covariance = covariance_x_angle[i, j]
            denominator = rms_x[i] * rms_angle[j]
            correlation = covariance / denominator if denominator > 0.0 else 0.0
            suffix = "%s_%s" % (position_axis, angle_axis)
            output["covariance_%s" % suffix] = (
                float(covariance), _LENGTH_DIMENSION)
            output["correlation_%s" % suffix] = (
                float(correlation), _DIMENSIONLESS)

    output["centroid_observer_time"] = (mean_time, _TIME_DIMENSION)
    output["rms_observer_time"] = (rms_time, _TIME_DIMENSION)
    for i, axis in enumerate("xyz"):
        denominator = rms_x[i] * rms_time
        correlation = (
            covariance_x_time[i] / denominator if denominator > 0.0 else 0.0)
        output["covariance_%s_observer_time" % axis] = (
            float(covariance_x_time[i]), _LENGTH_DIMENSION + _TIME_DIMENSION)
        output["correlation_%s_observer_time" % axis] = (
            float(correlation), _DIMENSIONLESS)
    return {
        name: value for name, value in output.items()
        if _moment_component_selected(name, quantities)
    }


def pulse_components(raw_energy, time_edges, per_solid_angle, interval):
    widths = np.diff(time_edges)
    power = raw_energy / widths
    total = float(raw_energy.sum())
    centers = 0.5 * (time_edges[:-1] + time_edges[1:])
    peak_index = int(np.argmax(power)) if power.size else 0
    peak = float(power[peak_index]) if power.size else 0.0
    peak_time = float(centers[peak_index]) if centers.size else np.nan

    def quantile(fraction):
        if total <= 0.0:
            return np.nan
        cumulative = np.cumsum(raw_energy)
        index = min(int(np.searchsorted(cumulative, fraction * total)),
                    raw_energy.size - 1)
        previous = cumulative[index - 1] if index > 0 else 0.0
        within = 0.0
        if raw_energy[index] > 0.0:
            within = (fraction * total - previous) / raw_energy[index]
        return float(time_edges[index] + within * widths[index])

    lower = quantile(interval[0])
    upper = quantile(interval[1])
    energy_name = "energy_per_solid_angle" if per_solid_angle else "energy"
    power_name = "peak_power_per_solid_angle" if per_solid_angle else "peak_power"
    return {
        energy_name: (total, _ENERGY_DIMENSION),
        power_name: (peak, _POWER_DIMENSION),
        "peak_observer_time": (peak_time, _TIME_DIMENSION),
        "interval_start": (lower, _TIME_DIMENSION),
        "interval_end": (upper, _TIME_DIMENSION),
        "interval_duration": (upper - lower, _TIME_DIMENSION),
    }


class ObserverRadiationWriter(object):
    """Write reduced observer-frame accumulator snapshots."""

    def __init__(self, diagnostic, output_mode):
        self.diagnostic = diagnostic
        self.output_mode = output_mode
        self.previous = {}
        self.angular_measures = {}

    def _reduce(self, value):
        diagnostic = self.diagnostic
        size = 1 if diagnostic.comm is None else diagnostic.comm.size
        if size == 1:
            return value.copy()
        receive = np.empty_like(value) if diagnostic.rank == 0 else None
        comm_simple.Reduce(value, receive, root=0)
        return receive

    def _reduce_snapshot(self, snapshot):
        return {
            category: {
                key: self._reduce(snapshot[category][key])
                for key in sorted(snapshot[category])
            }
            for category in ("data", "accounting", "moments")
        }

    def _snapshots(self):
        modes = {}
        for species_name, accumulator in self.diagnostic.accumulators.items():
            current = accumulator.snapshot()
            previous = self.previous.get(
                species_name, _zero_snapshot_like(current))
            interval = _subtract_snapshot(current, previous)
            self.previous[species_name] = current
            selected = {}
            if self.output_mode in ("cumulative", "both"):
                selected["cumulative"] = self._reduce_snapshot(current)
            if self.output_mode in ("interval", "both"):
                selected["interval"] = self._reduce_snapshot(interval)
            modes[species_name] = selected
        return modes

    @staticmethod
    def _lorentz_matrix(accumulator):
        gamma = accumulator.gamma_boost
        gamma_beta = gamma * accumulator.beta_boost
        return np.array([
            [gamma, 0.0, 0.0, gamma_beta],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [gamma_beta, 0.0, 0.0, gamma],
        ])

    def _common_attributes(self, record, accumulator, species_name, mode):
        record.attrs["observerFrame"] = _bytes(accumulator.observer_frame)
        record.attrs["observerFrameGamma"] = accumulator.gamma_boost
        record.attrs["observerFrameBeta"] = accumulator.beta_boost
        record.attrs["observerLorentzTransform"] = self._lorentz_matrix(
            accumulator).ravel()
        record.attrs["observerLorentzTransformShape"] = np.array(
            [4, 4], dtype=np.uint64)
        record.attrs["observerTranslation"] = accumulator.observer_translation
        record.attrs["observerTranslationCoordinates"] = np.array(
            [b"ct", b"x", b"y", b"z"])
        record.attrs["observerTranslationUnitSI"] = 1.0
        record.attrs["observerTranslationUnitDimension"] = _LENGTH_DIMENSION
        record.attrs["observerTransformConvention"] = _bytes(
            "x_observer=Lambda_simulation_to_observer*x_simulation+b")
        record.attrs["worldlineIntervalTransform"] = _bytes(
            "dt_observer=(gamma_observer/gamma_simulation)*dt_simulation")
        record.attrs["species"] = _bytes(species_name)
        record.attrs["accumulationMode"] = _bytes(mode)
        record.attrs["cumulative"] = np.uint32(mode == "cumulative")
        record.attrs["spectralModel"] = _bytes(
            "normalized_classical_synchrotron_curvature")
        record.attrs["localAngularModel"] = _bytes("synchrotron")
        record.attrs["spectralAngularKernel"] = _bytes(
            "polarization_summed_Schwinger_vertical_conditional")
        record.attrs["angularKernelQMaximum"] = 8.0
        record.attrs["sampledAngleCap"] = 0.5 * math.pi
        record.attrs["samplesPerParticle"] = accumulator.samples_per_particle
        record.attrs["particleBatchSize"] = accumulator.particle_batch_size
        record.attrs["gammaThreshold"] = accumulator.gamma_cutoff
        record.attrs["particleSelection"] = _bytes(json.dumps(
            accumulator.particle_selection,
            default=lambda value: value.tolist()))
        record.attrs["spectralClosureMaximumX"] = accumulator.spectral_x[-1]
        record.attrs["spectralClosureTruncatedEnergyFraction"] = (
            accumulator.spectral_truncated_fraction)
        record.attrs["spectralClosureTailTreatment"] = _bytes(
            "reported_as_unrepresented_energy_not_renormalized")
        record.attrs["angularMeasure"] = _bytes(accumulator.angular_measure)
        record.attrs["angularCoordinateConvention"] = _bytes(
            "theta_x=atan2(n_x,n_z);theta_y=atan2(n_y,n_z)")
        record.attrs["photonEnergyDefinition"] = _bytes(
            "E_photon=hbar*omega_observer")
        record.attrs["macroparticleClosure"] = _bytes(
            "incoherent_linear_in_weight")
        record.attrs["radiationReaction"] = np.uint32(0)
        record.attrs["passiveDiagnostic"] = np.uint32(1)
        record.attrs["picEventTimeStaggering"] = _bytes(
            "position_and_momentum_at_particle_half_step;"
            "fields_from_preceding_integer_step")
        record.attrs["unitSI"] = 1.0
        record.attrs["timeOffset"] = 0.0

    @staticmethod
    def _axis_dimension(axis):
        if axis in ("x", "y", "z"):
            return _LENGTH_DIMENSION
        if axis == "time":
            return _TIME_DIMENSION
        if axis == "energy":
            return _ENERGY_DIMENSION
        return _DIMENSIONLESS

    @staticmethod
    def _source_dimension(axes):
        dimension = _ENERGY_DIMENSION.copy()
        for axis in axes:
            if axis in ("x", "y", "z"):
                dimension -= _LENGTH_DIMENSION
            elif axis == "time":
                dimension -= _TIME_DIMENSION
            elif axis == "energy":
                dimension -= _ENERGY_DIMENSION
        return dimension

    def _density(self, accumulator, key, raw):
        kind = accumulator.data_kinds[key]
        axes = accumulator.data_axes[key]
        if kind.startswith("direction_time") or kind.startswith("aperture_time"):
            return raw / np.diff(axes[0][1])

        density = raw.copy()
        axis_names = [axis for axis, _ in axes]
        handled_angular = set()
        if "theta_x" in axis_names and "theta_y" in axis_names:
            ix = axis_names.index("theta_x")
            iy = axis_names.index("theta_y")
            cache_key = id(accumulator)
            if cache_key not in self.angular_measures:
                self.angular_measures[cache_key] = angular_cell_measure(
                    accumulator.theta_x_edges, accumulator.theta_y_edges,
                    accumulator.angular_measure)
            measure = self.angular_measures[cache_key]
            if ix > iy:
                measure = measure.T
            reshape = [1] * raw.ndim
            reshape[ix] = raw.shape[ix]
            reshape[iy] = raw.shape[iy]
            density /= measure.reshape(reshape)
            handled_angular = {"theta_x", "theta_y"}
        for dimension, (axis, edges) in enumerate(axes):
            if axis in handled_angular:
                continue
            reshape = [1] * raw.ndim
            reshape[dimension] = raw.shape[dimension]
            density /= np.diff(edges).reshape(reshape)
        return density

    def _product_dimension(self, accumulator, key):
        kind = accumulator.data_kinds[key]
        if kind == "angular_spectral":
            return _DIMENSIONLESS
        if kind.startswith("direction_time") or kind.startswith("aperture_time"):
            return _POWER_DIMENSION
        if kind == "source":
            return self._source_dimension(
                [axis for axis, _ in accumulator.data_axes[key]])
        return _DIMENSIONLESS

    @staticmethod
    def _record_name(key, species_name, mode):
        if key == "angular_spectral":
            product = "AngularSpectrum"
        elif key.startswith("source/"):
            product = "Source_" + key.split("/", 1)[1]
        elif key.startswith("detector/"):
            product = "ObserverTime_" + "_".join(key.split("/")[1:])
        else:
            product = _clean(key)
        return "radiation%s_%s_%s" % (
            product, _clean(species_name), mode)

    def _write_axes(self, iteration_group, record, record_name, axes,
                    accumulator, species_name, mode):
        root = iteration_group.require_group("radiationAxes")
        record_axes = root.require_group(record_name)
        paths = []
        centers = []
        spacings = []
        offsets = []
        for axis, edges in axes:
            edge_name = _clean(axis) + "Edges"
            dataset = record_axes.create_dataset(edge_name, data=edges)
            dataset.attrs["unitSI"] = 1.0
            dataset.attrs["unitDimension"] = self._axis_dimension(axis)
            dataset.attrs["observerFrame"] = _bytes(accumulator.observer_frame)
            dataset.attrs["axis"] = _bytes(axis)
            paths.append(dataset.name)
            axis_centers = 0.5 * (edges[:-1] + edges[1:])
            centers.append(axis_centers)
            offsets.append(axis_centers[0])
            differences = np.diff(axis_centers)
            if differences.size == 0:
                spacings.append(1.0)
            elif np.allclose(differences, differences[0], rtol=1.e-12, atol=0.0):
                spacings.append(differences[0])
            else:
                spacings.append(1.0)
        record.attrs["axisEdgePaths"] = np.array(
            [_bytes(path) for path in paths])
        record.attrs["axisLabels"] = np.array(
            [_bytes(axis) for axis, _ in axes])
        record.attrs["gridGlobalOffset"] = np.asarray(offsets)
        record.attrs["gridSpacing"] = np.asarray(spacings)
        record.attrs["nonuniformGrid"] = np.uint32(any(
            not np.allclose(np.diff(center), np.diff(center)[0],
                            rtol=1.e-12, atol=0.0)
            if center.size > 2 else False
            for center in centers))
        # Small edge arrays are duplicated as attributes for convenient
        # inspection, while the path above remains safe for large axes.
        for axis, edges in axes:
            if edges.size <= 256:
                record.attrs["binEdges_%s" % axis] = edges

    def _product_metadata(self, record, accumulator, key):
        kind = accumulator.data_kinds[key]
        record.attrs["radiationProduct"] = _bytes(key)
        if kind == "angular_spectral":
            record.attrs["longName"] = _bytes(
                "d^3 W_perp / (d angular_measure d photon_energy)")
            record.attrs["longitudinalAccelerationIncluded"] = np.uint32(0)
        elif kind.startswith("direction_time"):
            record.attrs["longName"] = _bytes(
                "d^2 W / (d observer_time d solid_angle)")
            record.attrs["longitudinalAccelerationIncluded"] = np.uint32(
                "band/" not in key)
        elif kind.startswith("aperture_time"):
            record.attrs["longName"] = _bytes(
                "d W_aperture / d observer_time")
            record.attrs["longitudinalAccelerationIncluded"] = np.uint32(
                "band/" not in key)
        elif kind == "source":
            record.attrs["longName"] = _bytes(
                "radiation-weighted observer-frame source distribution")
            record.attrs["longitudinalAccelerationIncluded"] = np.uint32(0)
            projection_name = key.split("/", 1)[1]
            projection = next(
                item for item in accumulator.source_projections
                if item["name"] == projection_name)
            record.attrs["sourceSelection"] = _bytes(json.dumps(
                projection["selection"], default=lambda value: value.tolist()))
            record.attrs["sourceAccumulation"] = _bytes(
                "deterministic_integrated_curvature_energy"
                if projection["deterministic"] else
                "sampled_local_spectral_angular_closure")
            angular_axes = [
                axis for axis in projection["axes"]
                if axis in ("theta_x", "theta_y")]
            if len(angular_axes) == 2:
                effective_measure = accumulator.angular_measure
            elif len(angular_axes) == 1:
                effective_measure = "d%s_marginal" % angular_axes[0]
            else:
                effective_measure = "integrated_over_sampled_angles"
            record.attrs["sourceEffectiveAngularMeasure"] = _bytes(
                effective_measure)
            if "time" in projection["axes"]:
                record.attrs["observerTimeDefinition"] = _bytes(
                    "t_observer_minus_sampled_photon_direction_dot_r_over_c")
                record.attrs["observerTimeConditioning"] = _bytes(
                    "direction_conditioned_radiation_phase_coordinate;"
                    "angle_marginals_mix_distinct_null_coordinates")

        if key.startswith("detector/"):
            detector_name = key.split("/")[1]
            detector = next(
                item for item in accumulator.detectors
                if item["name"] == detector_name)
            record.attrs["detectorDirection"] = detector["direction"]
            record.attrs["apertureHalfAngle"] = detector["half_angle"]
            record.attrs["apertureSolidAngle"] = (
                2.0 * math.pi * (1.0 - math.cos(detector["half_angle"])))
            record.attrs["apertureQuadrature"] = detector["rays"].shape[0]
            if kind.startswith("aperture_time"):
                time_definition = (
                    "t_observer_minus_each_aperture_quadrature_"
                    "direction_dot_r_over_c")
            else:
                time_definition = \
                    "t_observer_minus_detector_direction_dot_r_over_c"
            record.attrs["observerTimeDefinition"] = _bytes(time_definition)
            record.attrs["broadbandAngularModel"] = _bytes(
                "exact_local_Lienard_power")
            if "/band/" in key:
                band_name = key.split("/")[3]
                band = next(
                    item for item in detector["energy_bands"]
                    if item["name"] == band_name)
                record.attrs["photonEnergySelection"] = band["energy_range"]
                record.attrs["bandSpectralClosure"] = _bytes(
                    "curvature_only_finite_synchrotron_CDF_with_reported_tail")
                record.attrs["bandAngularModel"] = _bytes(
                    "exact_transverse_Lienard_pattern")
                record.attrs["bandSpectralAngularClosure"] = _bytes(
                    "separable_transverse_Lienard_angular_pattern_times_"
                    "angle_integrated_synchrotron_band_fraction")
                record.attrs["bandEnergyAngleCouplingRetained"] = np.uint32(0)
                record.attrs["bandClosureScope"] = _bytes(
                    "approximation;not_the_joint_spectral_angular_kernel")

    def _write_product(self, field_group, iteration_group, accumulator,
                       species_name, mode, key, raw):
        name = self._record_name(key, species_name, mode)
        density = self._density(accumulator, key, raw)
        record = field_group.create_dataset(name, data=density)
        self._common_attributes(record, accumulator, species_name, mode)
        record.attrs["unitDimension"] = self._product_dimension(accumulator, key)
        record.attrs["geometry"] = _bytes("cartesian")
        record.attrs["dataOrder"] = _bytes("C")
        record.attrs["gridUnitSI"] = 1.0
        record.attrs["fieldSmoothing"] = _bytes("none")
        record.attrs["position"] = np.zeros(raw.ndim)
        record.attrs["binning"] = _bytes("cell_integrated_then_density")
        self._product_metadata(record, accumulator, key)
        self._write_axes(
            iteration_group, record, name, accumulator.data_axes[key],
            accumulator, species_name, mode)

    def _write_component_group(self, field_group, name, components,
                               accumulator, species_name, mode, attributes=None):
        records = {}
        for component_name, (value, dimension) in sorted(components.items()):
            record_name = "%s_%s" % (name, _clean(component_name))
            component = field_group.create_dataset(
                record_name, data=np.array([value], dtype=np.float64))
            self._common_attributes(
                component, accumulator, species_name, mode)
            component.attrs["geometry"] = _bytes("cartesian")
            component.attrs["dataOrder"] = _bytes("C")
            component.attrs["axisLabels"] = np.array([b"scalar"])
            component.attrs["gridSpacing"] = np.array([1.0])
            component.attrs["gridGlobalOffset"] = np.array([0.0])
            component.attrs["gridUnitSI"] = 1.0
            component.attrs["fieldSmoothing"] = _bytes("none")
            component.attrs["unitSI"] = 1.0
            component.attrs["unitDimension"] = dimension
            component.attrs["position"] = np.array([0.0])
            component.attrs["recordGroup"] = _bytes(name)
            component.attrs["componentName"] = _bytes(component_name)
            if attributes:
                for key, attribute_value in attributes.items():
                    component.attrs[key] = attribute_value
            records[component_name] = component
        return records

    def _write_accounting(self, field_group, accumulator, species_name,
                          mode, accounting):
        components = {
            key: (float(value[0]), _ENERGY_DIMENSION)
            for key, value in accounting.items()
        }
        transverse = float(accounting["transverse_energy"][0])
        longitudinal = float(accounting["longitudinal_energy"][0])
        total = transverse + longitudinal
        components["total_radiated_energy"] = (total, _ENERGY_DIMENSION)
        components["longitudinal_energy_fraction"] = (
            longitudinal / total if total > 0.0 else 0.0,
            _DIMENSIONLESS)
        name = "radiationAccounting_%s_%s" % (
            _clean(species_name), mode)
        self._write_component_group(
            field_group, name, components, accumulator, species_name, mode,
            {
                "longName": _bytes(
                    "observer-frame radiation energy accounting"),
                "energyRangeAccounting": _bytes(
                    "deterministic_integral_of_finite_synchrotron_CDF;"
                    "above_range_includes_any_unrepresented_high_x_tail"),
                "spectralTailAccounting": _bytes(
                    "energy_truncated_by_spectral_closure_is_informational;"
                    "it_overlaps_energy_above_range_when_that_range_is_finite"),
                "energyRangeIncludesLongitudinalAcceleration": np.uint32(0),
                "angularLossAccounting": _bytes(
                    "sampled_local_curvature_closure"),
                "angularLossIncludesLongitudinalAcceleration": np.uint32(0),
            })

    def _write_moments(self, field_group, accumulator, species_name,
                       mode, moments):
        selections = {
            item["name"]: item for item in accumulator.source_moment_selections}
        for selection_name, stats in moments.items():
            name = "radiationSourceMoments_%s_%s_%s" % (
                _clean(selection_name), _clean(species_name), mode)
            selection = selections[selection_name]
            selection_json = json.dumps(
                selection,
                default=lambda value: value.tolist())
            attributes = {
                "longName": _bytes(
                    "radiation-weighted observer-frame source moments"),
                "sourceMomentSelection": _bytes(selection_json),
                "sourceMomentQuantities": np.array([
                    _bytes(value) for value in selection["quantities"]]),
                "sourceMomentAccumulation": _bytes(
                    "deterministic_integrated_curvature_energy"
                    if selection["deterministic"] else
                    "sampled_local_spectral_angular_closure"),
                "longitudinalAccelerationIncluded": np.uint32(0),
                "transverseEllipticityDefinition": _bytes(
                    "(sigma_major-sigma_minor)/(sigma_major+sigma_minor)"),
                "transverseOrientationConvention": _bytes(
                    "canonical_major_eigenvector_with_nonnegative_x;"
                    "orientation_modulo_pi"),
            }
            if "time" in selection["quantities"]:
                attributes.update({
                    "observerTimeDefinition": _bytes(
                        "t_observer_minus_sampled_photon_direction_dot_r_over_c"),
                    "observerTimeConditioning": _bytes(
                        "direction_conditioned_radiation_phase_coordinate;"
                        "angle_marginals_mix_distinct_null_coordinates"),
                })
            self._write_component_group(
                field_group, name, source_moment_components(
                    stats, selection["quantities"]),
                accumulator, species_name, mode,
                attributes)

    def _write_pulse_metrics(self, field_group, accumulator, species_name,
                             mode, data):
        for detector in accumulator.detectors:
            prefix = "detector/%s" % detector["name"]
            for suffix, per_solid_angle in (
                    ("direction", True), ("aperture", False)):
                key = prefix + "/" + suffix
                if key not in data:
                    continue
                components = pulse_components(
                    data[key], detector["time_edges"], per_solid_angle,
                    detector["pulse_interval"])
                name = "radiationPulseMetrics_%s_%s_%s_%s" % (
                    _clean(detector["name"]), suffix,
                    _clean(species_name), mode)
                self._write_component_group(
                    field_group, name, components, accumulator, species_name,
                    mode,
                    {
                        "detectorDirection": detector["direction"],
                        "apertureHalfAngle": detector["half_angle"],
                        "cumulativeEnergyInterval": detector["pulse_interval"],
                        "observerTimeDefinition": _bytes(
                            "t_observer_minus_each_aperture_quadrature_"
                            "direction_dot_r_over_c"
                            if suffix == "aperture" else
                            "t_observer_minus_detector_direction_dot_r_over_c"),
                        "broadbandAngularModel": _bytes(
                            "exact_local_Lienard_power"),
                        "longName": _bytes(
                            "broadband observer-time pulse metrics"),
                        "peakDefinition": _bytes("maximum_bin_average_power"),
                    })

    def write(self, iteration):
        diagnostic = self.diagnostic
        file_handle = None
        if diagnostic.use_cuda:
            for species_name in diagnostic.species_names:
                diagnostic.species[species_name].synchrotron_radiator \
                    .receive_from_gpu()
        try:
            snapshots = self._snapshots()
            first = next(iter(diagnostic.accumulators.values()))
            observer_time = (
                first.gamma_boost * iteration * diagnostic.dt_sim
                + first.observer_translation[0] / c)
            observer_dt = first.gamma_boost * diagnostic.dt_sim
            filename = "data%08d.h5" % iteration
            fullpath = os.path.join(diagnostic.write_dir, "hdf5", filename)
            file_handle = diagnostic.open_file(fullpath)
            if file_handle is None:
                return
            diagnostic.setup_openpmd_file(
                file_handle, iteration, observer_time, observer_dt)
            iteration_group = file_handle["/data/%d" % iteration]
            iteration_group.attrs["timeReferenceFrame"] = _bytes(
                first.observer_frame)
            iteration_group.attrs["timeReferenceEvent"] = _bytes(
                "simulation_origin_z_equals_zero")
            iteration_group.attrs["observerLorentzTransform"] = \
                self._lorentz_matrix(first).ravel()
            iteration_group.attrs["observerLorentzTransformShape"] = \
                np.array([4, 4], dtype=np.uint64)
            iteration_group.attrs["observerTranslation"] = \
                first.observer_translation
            field_group = iteration_group.require_group("fields")
            for species_name, mode_snapshots in snapshots.items():
                accumulator = diagnostic.accumulators[species_name]
                for mode, snapshot in mode_snapshots.items():
                    for key, raw in snapshot["data"].items():
                        self._write_product(
                            field_group, iteration_group, accumulator,
                            species_name, mode, key, raw)
                    self._write_accounting(
                        field_group, accumulator, species_name, mode,
                        snapshot["accounting"])
                    self._write_moments(
                        field_group, accumulator, species_name, mode,
                        snapshot["moments"])
                    self._write_pulse_metrics(
                        field_group, accumulator, species_name, mode,
                        snapshot["data"])
        finally:
            if file_handle is not None:
                file_handle.close()
            if diagnostic.use_cuda:
                for species_name in diagnostic.species_names:
                    diagnostic.species[species_name].synchrotron_radiator \
                        .send_to_gpu()

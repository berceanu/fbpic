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


_ADDITIVE_SNAPSHOT_CATEGORIES = (
    "data", "accounting", "sampling", "source_z")


def _subtract_snapshot(current, previous):
    return {
        category: {
            key: current[category][key] - previous[category][key]
            for key in current[category]
        }
        for category in _ADDITIVE_SNAPSHOT_CATEGORIES
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
        if name.startswith(("covariance_", "correlation_")):
            paired_quantity = (
                "angle" if "theta_" in name else "position")
            return (
                "time" in quantities and paired_quantity in quantities)
        return "time" in quantities
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
    for angle_axis in ("theta_x", "theta_y"):
        dimensions["covariance_%s_observer_time" % angle_axis] = \
            _TIME_DIMENSION
        dimensions["correlation_%s_observer_time" % angle_axis] = \
            _DIMENSIONLESS
    return {
        name: dimension for name, dimension in dimensions.items()
        if _moment_component_selected(name, quantities)
    }


_SOURCE_VARIABLE_COUNT = 6
_SOURCE_REFERENCED_STAT_SIZE = 34
_SOURCE_CENTRAL_STAT_SIZE = 28


def _packed_upper_index(first, second, count=_SOURCE_VARIABLE_COUNT):
    if second < first:
        first, second = second, first
    return first * count - first * (first - 1) // 2 + second - first


def merge_source_moment_stats(first, second):
    """Merge two referenced centered states without raw cancellation."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if (first.size != _SOURCE_REFERENCED_STAT_SIZE
            or second.size != _SOURCE_REFERENCED_STAT_SIZE):
        raise ValueError(
            "Referenced source-moment states must have 34 values.")
    if first[0] <= 0.0:
        return second.copy()
    if second[0] <= 0.0:
        return first.copy()
    output = first.copy()
    first_weight, second_weight = first[0], second[0]
    total_weight = first_weight + second_weight
    reference_offset = 1
    mean_offset = 1 + _SOURCE_VARIABLE_COUNT
    m2_offset = 1 + 2 * _SOURCE_VARIABLE_COUNT
    first_reference = first[reference_offset:mean_offset]
    second_reference = second[reference_offset:mean_offset]
    first_offset = first[mean_offset:m2_offset]
    second_offset = second[mean_offset:m2_offset]
    delta = (
        (second_reference - first_reference)
        + (second_offset - first_offset))
    output[0] = total_weight
    output[reference_offset:mean_offset] = first_reference
    output[mean_offset:m2_offset] = (
        first_offset + delta * second_weight / total_weight)
    merge_scale = first_weight * second_weight / total_weight
    for first_index in range(_SOURCE_VARIABLE_COUNT):
        for second_index in range(first_index, _SOURCE_VARIABLE_COUNT):
            packed = _packed_upper_index(first_index, second_index)
            output[m2_offset + packed] = (
                first[m2_offset + packed] + second[m2_offset + packed]
                + delta[first_index] * delta[second_index] * merge_scale)
    return output


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

    if stats.size == _SOURCE_REFERENCED_STAT_SIZE:
        reference_offset = 1
        mean_offset = 1 + _SOURCE_VARIABLE_COUNT
        m2_offset = 1 + 2 * _SOURCE_VARIABLE_COUNT
        means = (
            stats[reference_offset:mean_offset]
            + stats[mean_offset:m2_offset])
        covariance = np.zeros((6, 6), dtype=np.float64)
        offset = m2_offset
        for first_index in range(_SOURCE_VARIABLE_COUNT):
            for second_index in range(
                    first_index, _SOURCE_VARIABLE_COUNT):
                value = stats[
                    offset + _packed_upper_index(
                        first_index, second_index)] / weight
                covariance[first_index, second_index] = value
                covariance[second_index, first_index] = value
        covariance[np.diag_indices(6)] = np.maximum(
            covariance.diagonal(), 0.0)
        mean_x = means[:3]
        covariance_x = covariance[:3, :3]
        mean_angle = means[3:5]
        covariance_angle = covariance[3:5, 3:5]
        covariance_x_angle = covariance[:3, 3:5]
        mean_time = means[5]
        variance_time = covariance[5, 5]
        covariance_x_time = covariance[:3, 5]
        covariance_angle_time = covariance[3:5, 5]
    elif stats.size == _SOURCE_CENTRAL_STAT_SIZE:
        means = stats[1:7]
        covariance = np.zeros((6, 6), dtype=np.float64)
        offset = 1 + _SOURCE_VARIABLE_COUNT
        for first_index in range(_SOURCE_VARIABLE_COUNT):
            for second_index in range(
                    first_index, _SOURCE_VARIABLE_COUNT):
                value = stats[
                    offset + _packed_upper_index(
                        first_index, second_index)] / weight
                covariance[first_index, second_index] = value
                covariance[second_index, first_index] = value
        covariance[np.diag_indices(6)] = np.maximum(
            covariance.diagonal(), 0.0)
        mean_x = means[:3]
        covariance_x = covariance[:3, :3]
        mean_angle = means[3:5]
        covariance_angle = covariance[3:5, 3:5]
        covariance_x_angle = covariance[:3, 3:5]
        mean_time = means[5]
        variance_time = covariance[5, 5]
        covariance_x_time = covariance[:3, 5]
        covariance_angle_time = covariance[3:5, 5]
    elif stats.size == 30:
        # Read legacy additive raw sums for compatibility with old files and
        # callers, but all new accumulation uses the centered state above.
        mean_x = stats[1:4] / weight
        second_x = stats[4:13].reshape(3, 3) / weight
        covariance_x = 0.5 * (
            second_x + second_x.T) - np.outer(mean_x, mean_x)
        mean_angle = stats[13:15] / weight
        second_angle = stats[15:19].reshape(2, 2) / weight
        covariance_angle = 0.5 * (
            second_angle + second_angle.T) - np.outer(
                mean_angle, mean_angle)
        covariance_x_angle = (
            stats[19:25].reshape(3, 2) / weight
            - np.outer(mean_x, mean_angle))
        mean_time = stats[25] / weight
        variance_time = max(
            stats[26] / weight - mean_time**2, 0.0)
        covariance_x_time = (
            stats[27:30] / weight - mean_x * mean_time)
        # The legacy raw layout did not retain angle-time cross terms.
        covariance_angle_time = np.full(2, np.nan)
    else:
        raise ValueError("Unknown source-moment state layout.")

    covariance_x[np.diag_indices(3)] = np.maximum(
        covariance_x.diagonal(), 0.0)
    covariance_angle[np.diag_indices(2)] = np.maximum(
        covariance_angle.diagonal(), 0.0)
    rms_x = np.sqrt(covariance_x.diagonal())
    rms_angle = np.sqrt(covariance_angle.diagonal())
    rms_time = math.sqrt(max(variance_time, 0.0))

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
    for index, axis in enumerate(("theta_x", "theta_y")):
        denominator = rms_angle[index] * rms_time
        correlation = (
            covariance_angle_time[index] / denominator
            if denominator > 0.0 else 0.0)
        output["covariance_%s_observer_time" % axis] = (
            float(covariance_angle_time[index]), _TIME_DIMENSION)
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


def source_z_interval_components(histogram, edges, fractions):
    """Derive equal-tail central energy intervals from a mergeable histogram."""
    histogram = np.asarray(histogram, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if histogram.size != edges.size + 1:
        raise ValueError("Source-z histogram must include two tail bins.")
    underflow = float(histogram[0])
    interior = histogram[1:-1]
    overflow = float(histogram[-1])
    contained = float(interior.sum())
    total = underflow + contained + overflow

    def quantile(fraction):
        if not total > 0.0:
            return np.nan
        target = fraction * total
        if target <= underflow:
            return -np.inf
        target -= underflow
        if target > contained:
            return np.inf
        cumulative = np.cumsum(interior)
        index = min(
            int(np.searchsorted(cumulative, target, side="left")),
            interior.size - 1)
        previous = cumulative[index - 1] if index > 0 else 0.0
        within = 0.0
        if interior[index] > 0.0:
            within = (target - previous) / interior[index]
        return float(
            edges[index] + within * (edges[index + 1] - edges[index]))

    components = {
        "source_z_interval_total_energy": (total, _ENERGY_DIMENSION),
        "source_z_interval_contained_energy": (
            contained, _ENERGY_DIMENSION),
        "source_z_interval_underflow_energy": (
            underflow, _ENERGY_DIMENSION),
        "source_z_interval_overflow_energy": (
            overflow, _ENERGY_DIMENSION),
        "source_z_interval_contained_fraction": (
            contained / total if total > 0.0 else 0.0,
            _DIMENSIONLESS),
    }
    for fraction in fractions:
        lower = quantile(0.5 * (1.0 - fraction))
        upper = quantile(0.5 * (1.0 + fraction))
        label = ("%g" % (100.0 * fraction)).replace(".", "p")
        prefix = "central_%s_percent_z" % label
        components[prefix + "_start"] = (lower, _LENGTH_DIMENSION)
        components[prefix + "_end"] = (upper, _LENGTH_DIMENSION)
        components[prefix + "_width"] = (
            upper - lower, _LENGTH_DIMENSION)
    return components


class ObserverRadiationWriter(object):
    """Write reduced observer-frame accumulator snapshots."""

    def __init__(self, diagnostic, output_mode):
        self.diagnostic = diagnostic
        self.output_mode = output_mode
        self.previous = {}
        self.pending_previous = {}
        self.angular_measures = {}
        self.last_written_events = {
            name: 0 for name in diagnostic.species_names}
        self.active_timing = {}
        self.final_flush = False

    def has_unwritten_events(self):
        local_pending = any(
            accumulator.completed_event_count
            > self.last_written_events.get(species_name, 0)
            for species_name, accumulator
            in self.diagnostic.accumulators.items())
        diagnostic = self.diagnostic
        size = 1 if diagnostic.comm is None else diagnostic.comm.size
        if size == 1:
            return local_pending
        return bool(comm_simple.allreduce(int(local_pending)))

    def latest_unwritten_event_index(self):
        """Return the latest completed event not represented by a file."""
        event = self.latest_unwritten_event()
        return None if event is None else event[0]

    def latest_unwritten_event(self):
        """Return ``(index, center time)`` for the newest dirty event.

        Every rank, including an empty rank, completes the same pusher event,
        so this local metadata is decomposition independent without a particle
        reduction.
        """
        events = [
            (accumulator.last_completed_event_index,
             accumulator.cumulative_timing["last_event_center"])
            for species_name, accumulator
            in self.diagnostic.accumulators.items()
            if (accumulator.completed_event_count
                > self.last_written_events.get(species_name, 0)
                and accumulator.last_completed_event_index is not None)
        ]
        return max(events, key=lambda item: item[0]) if events else None

    def _reduce(self, value):
        diagnostic = self.diagnostic
        size = 1 if diagnostic.comm is None else diagnostic.comm.size
        if size == 1:
            # Snapshot arrays are already private, read-only writer inputs.
            return value
        receive = np.empty_like(value) if diagnostic.rank == 0 else None
        comm_simple.Reduce(value, receive, root=0)
        return receive

    def _reduce_max(self, value):
        diagnostic = self.diagnostic
        size = 1 if diagnostic.comm is None else diagnostic.comm.size
        if size == 1:
            return value
        gathered = comm_simple.gather(value, root=0)
        if diagnostic.rank != 0:
            return None
        return np.maximum.reduce(gathered)

    def _reduce_resolution(self, value):
        """Reduce a [maximum, additive statistics...] resolution state."""
        reduced = self._reduce(value)
        maximum = self._reduce_max(value[:1])
        if self.diagnostic.rank == 0:
            reduced[0] = maximum[0]
        return reduced

    def _reduce_moments(self, value):
        diagnostic = self.diagnostic
        size = 1 if diagnostic.comm is None else diagnostic.comm.size
        if size == 1:
            return value
        gathered = comm_simple.gather(value, root=0)
        if diagnostic.rank != 0:
            return None
        merged = np.zeros_like(value)
        for item in gathered:
            merged = merge_source_moment_stats(merged, item)
        return merged

    def _reduce_timing(self, timing):
        diagnostic = self.diagnostic
        size = 1 if diagnostic.comm is None else diagnostic.comm.size
        if size == 1:
            return dict(timing)
        gathered = comm_simple.gather(dict(timing), root=0)
        if diagnostic.rank != 0:
            return None
        nonempty = [
            item for item in gathered if item["event_count"] > 0]
        if not nonempty:
            return {
                "event_count": 0,
                "first_event_index": np.iinfo(np.int64).max,
                "last_event_index": np.iinfo(np.int64).min,
                "first_event_center": math.inf,
                "last_event_center": -math.inf,
                "represented_interval_start": math.inf,
                "represented_interval_end": -math.inf,
            }
        return {
            "event_count": max(
                item["event_count"] for item in nonempty),
            "first_event_index": min(
                item["first_event_index"] for item in nonempty),
            "last_event_index": max(
                item["last_event_index"] for item in nonempty),
            "first_event_center": min(
                item["first_event_center"] for item in nonempty),
            "last_event_center": max(
                item["last_event_center"] for item in nonempty),
            "represented_interval_start": min(
                item["represented_interval_start"] for item in nonempty),
            "represented_interval_end": max(
                item["represented_interval_end"] for item in nonempty),
        }

    def _reduce_snapshot(self, snapshot):
        reduced = {
            category: {
                key: self._reduce(snapshot[category][key])
                for key in sorted(snapshot[category])
            }
            for category in _ADDITIVE_SNAPSHOT_CATEGORIES
        }
        reduced["quality"] = {
            key: self._reduce_max(snapshot["quality"][key])
            for key in sorted(snapshot["quality"])}
        reduced["resolution"] = {
            key: self._reduce_resolution(snapshot["resolution"][key])
            for key in sorted(snapshot["resolution"])}
        reduced["moments"] = {
            key: self._reduce_moments(snapshot["moments"][key])
            for key in sorted(snapshot["moments"])}
        reduced["timing"] = self._reduce_timing(snapshot["timing"])
        return reduced

    def _snapshots(self):
        modes = {}
        self.active_timing = {}
        self.pending_previous = {}
        for species_name, accumulator in self.diagnostic.accumulators.items():
            current = accumulator.snapshot()
            selected = {}
            if self.output_mode in ("cumulative", "both"):
                selected["cumulative"] = self._reduce_snapshot(current)
            if self.output_mode in ("interval", "both"):
                interval_state = accumulator.snapshot(
                    interval_moments=True)
                previous = self.previous.get(species_name)
                if previous is None:
                    # The first interval is the cumulative state itself. A
                    # shallow mapping is sufficient because writer inputs are
                    # never mutated.
                    interval = {
                        category: dict(current[category])
                        for category in _ADDITIVE_SNAPSHOT_CATEGORIES}
                else:
                    interval = _subtract_snapshot(current, previous)
                interval["quality"] = interval_state["quality"]
                interval["resolution"] = interval_state["resolution"]
                interval["moments"] = interval_state["moments"]
                interval["timing"] = interval_state["timing"]
                # The cumulative host snapshot is already detached from the
                # accumulator. Reuse it as the next frozen interval baseline
                # instead of copying every dense product a second time.
                self.pending_previous[species_name] = {
                    category: dict(current[category])
                    for category in _ADDITIVE_SNAPSHOT_CATEGORIES}
                selected["interval"] = self._reduce_snapshot(interval)
            for mode, snapshot in selected.items():
                self.active_timing[(species_name, mode)] = snapshot["timing"]
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
        record.attrs["randomStreamIdentifiers"] = _bytes(json.dumps(
            accumulator.random_stream_ids, sort_keys=True))
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
        record.attrs["randomSeed"] = np.uint64(accumulator.random_seed)
        record.attrs["randomSamplingMethod"] = _bytes(
            "stateless_splitmix64_persistent_particle_identity")
        record.attrs["randomEventKey"] = _bytes(
            "diagnostic_seed;persistent_particle_id;event_index;"
            "independent_stream_id;sample_index;species_namespace")
        record.attrs["randomSpeciesNamespace"] = np.uint64(
            accumulator.random_namespace)
        record.attrs["persistentParticleIdentitySource"] = _bytes(
            "fbpic_particle_tracker_uint64")
        record.attrs["randomOrderIndependence"] = _bytes(
            "particle_batching;particle_sorting;particle_migration;"
            "MPI_ownership;CPU_GPU_execution_order")
        record.attrs["particleSamplingFraction"] = (
            accumulator.particle_sampling_fraction)
        record.attrs["particleThinning"] = np.uint32(
            accumulator.particle_sampling_fraction < 1.0)
        record.attrs["particleThinningEstimator"] = _bytes(
            "Bernoulli_Horvitz_Thompson_linear_in_macroparticle_weight")
        record.attrs["estimatedDenseProductBytes"] = np.uint64(
            accumulator.estimated_dense_product_bytes)
        record.attrs["estimatedTotalDiagnosticBytes"] = np.uint64(
            accumulator.estimated_total_allocation_bytes)
        record.attrs["maxAllocationBytes"] = (
            np.uint64(accumulator.max_allocation_bytes)
            if accumulator.max_allocation_bytes is not None
            else _bytes("unlimited"))
        record.attrs["allocationLimitScope"] = _bytes(
            "complete_peak_diagnostic_footprint_including_persistent_batch_"
            "lookup_reduction_writer_and_pusher_coupling_memory")
        record.attrs["allocationBreakdown"] = _bytes(json.dumps(
            accumulator.allocation_breakdown, sort_keys=True))
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
        record.attrs["angularDensityMeasure"] = _bytes(
            "per_solid_angle" if accumulator.angular_measure == "solid_angle"
            else "per_projected_angle_measure_dtheta_x_dtheta_y")
        record.attrs["angularJacobianTreatment"] = _bytes(
            "exact_spherical_cell_solid_angle"
            if accumulator.angular_measure == "solid_angle"
            else "projected_angle_bin_area")
        record.attrs["angularCoordinateConvention"] = _bytes(
            "theta_x=atan2(n_x,n_z);theta_y=atan2(n_y,n_z)")
        record.attrs["photonEnergyDefinition"] = _bytes(
            "E_photon=hbar*omega_observer")
        record.attrs["macroparticleClosure"] = _bytes(
            "incoherent_linear_in_weight")
        record.attrs["radiationReaction"] = np.uint32(0)
        record.attrs["passiveDiagnostic"] = np.uint32(1)
        record.attrs["eventModel"] = _bytes(
            "centered_covariant_pusher_impulse_v1")
        record.attrs["accelerationSource"] = _bytes(
            "completed_particle_momentum_push_not_gathered_fields")
        record.attrs["picEventTimeStaggering"] = _bytes(
            "position_at_integer_time_center;"
            "momentum_endpoints_at_n_minus_and_plus_one_half")
        record.attrs["diagnosticWritePhase"] = _bytes(
            "post_momentum_impulse_pre_elementary_process")
        record.attrs["eventFourVelocityDefinition"] = _bytes(
            "normalized_sum_of_dimensionless_endpoint_four_velocities")
        record.attrs["eventFourAccelerationDefinition"] = _bytes(
            "c_times_endpoint_four_velocity_difference_over_centered_"
            "proper_time")
        record.attrs["eventInvariantContract"] = _bytes(
            "U_squared_equals_one;U_dot_A_equals_zero")
        record.attrs["observerCoordinateTimeWeight"] = _bytes(
            "delta_t_observer=gamma_observer*delta_tau")
        record.attrs["observerTimeConvention"] = _bytes(
            "tau_D=t_observer-n_D_dot_x_observer/c")
        record.attrs["retardedTimeEvaluation"] = _bytes(
            "observer_light_front_coordinates_cancellation_safe")
        record.attrs["finalFlush"] = np.uint32(self.final_flush)
        timing = self.active_timing.get((species_name, mode))
        if timing is not None:
            record.attrs["representedEventCount"] = np.uint64(
                timing["event_count"])
            if timing["event_count"] > 0:
                record.attrs["firstIncludedEventIteration"] = np.int64(
                    timing["first_event_index"])
                record.attrs["lastIncludedEventIteration"] = np.int64(
                    timing["last_event_index"])
                record.attrs["firstEventCenterTimeSimulation"] = (
                    timing["first_event_center"])
                record.attrs["lastEventCenterTimeSimulation"] = (
                    timing["last_event_center"])
                record.attrs["representedIntervalStartSimulation"] = (
                    timing["represented_interval_start"])
                record.attrs["representedIntervalEndSimulation"] = (
                    timing["represented_interval_end"])
                record.attrs["representedIntervalConvention"] = _bytes(
                    "centered_momentum_push_coordinate_time_interval")
                record.attrs["eventTimingUnitSI"] = 1.0
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
            uses_source_time = (
                "time" in projection["axes"]
                or any(key in projection["selection"] for key in (
                    "observer_time_range", "time_range")))
            if uses_source_time:
                reference = projection["time_reference"]
                record.attrs["sourceTimeReference"] = _bytes(reference)
                if reference == "photon_direction":
                    record.attrs["sourceTimeConvention"] = _bytes(
                        "sampled_photon_direction_source_time")
                    record.attrs["observerTimeDefinition"] = _bytes(
                        "t_observer_minus_sampled_photon_direction_dot_"
                        "r_over_c")
                    record.attrs["observerTimeConditioning"] = _bytes(
                        "direction_conditioned_radiation_phase_coordinate;"
                        "angle_marginals_mix_distinct_null_coordinates")
                else:
                    record.attrs["sourceTimeConvention"] = _bytes(
                        "fixed_detector_referenced_source_time")
                    record.attrs["observerTimeDefinition"] = _bytes(
                        "tau_D=t_observer-n_D_dot_r_observer/c")
                    record.attrs["observerTimeConditioning"] = _bytes(
                        "detector_referenced_source_attribution")
                    record.attrs["sourceTimeDetector"] = _bytes(reference)
                    record.attrs["sourceTimeDetectorDirection"] = (
                        projection["time_direction"])

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
                band_mode = detector["energy_band_mode"]
                record.attrs["bandMode"] = _bytes(band_mode)
                if band_mode == "joint":
                    record.attrs["bandSpectralAngularClosure"] = _bytes(
                        "Schwinger_angle_conditioned_energy_CDF_with_exact_"
                        "transverse_Lienard_angular_marginal")
                    record.attrs[
                        "bandEnergyAngleCouplingRetained"] = np.uint32(1)
                    record.attrs["bandClosureScope"] = _bytes(
                        "local_joint_spectral_angular_closure;"
                        "finite_spectral_tail_reported_separately")
                else:
                    record.attrs["bandSpectralAngularClosure"] = _bytes(
                        "separable_transverse_Lienard_angular_pattern_times_"
                        "angle_integrated_synchrotron_band_fraction")
                    record.attrs[
                        "bandEnergyAngleCouplingRetained"] = np.uint32(0)
                    record.attrs["bandClosureScope"] = _bytes(
                        "explicit_fast_approximation;"
                        "not_the_joint_spectral_angular_kernel")

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
                    "stochastic_spectral_angular_packet_estimator"),
                "deterministicApertureAccounting": _bytes(
                    "independent_Lienard_quadrature_estimator"),
                "apertureComplementSemantics": _bytes(
                    "deterministic_broadband_aperture_energy_is_independent_"
                    "of_stochastic_spectral_angular_inside_and_outside;"
                    "only_stochastic_inside_plus_stochastic_outside_is_an_"
                    "exact_same_packet_partition"),
                "deterministicBroadbandApertureEstimator": _bytes(
                    "Lienard_direction_quadrature_including_transverse_and_"
                    "longitudinal_acceleration"),
                "stochasticApertureEstimator": _bytes(
                    "retained_transverse_synchrotron_spectral_angular_packets"),
                "stochasticOutsideApertureEstimator": _bytes(
                    "same_retained_transverse_packets_not_inside_detector"),
                "angularLossIncludesLongitudinalAcceleration": np.uint32(0),
            })

    def _write_quality(self, field_group, accumulator, species_name,
                       mode, quality, resolution):
        components = {
            key: (float(value[0]), _DIMENSIONLESS)
            for key, value in quality.items()}
        for quantity, state in resolution.items():
            maximum, energy, weighted_sum, weighted_square, above = (
                float(value) for value in state)
            mean = weighted_sum / energy if energy > 0.0 else 0.0
            rms = (
                math.sqrt(max(weighted_square / energy, 0.0))
                if energy > 0.0 else 0.0)
            above_fraction = above / energy if energy > 0.0 else 0.0
            components["max_%s" % quantity] = (
                maximum, _DIMENSIONLESS)
            components[
                "transverse_energy_weighted_mean_%s" % quantity] = (
                    mean, _DIMENSIONLESS)
            components[
                "transverse_energy_weighted_rms_%s" % quantity] = (
                    rms, _DIMENSIONLESS)
            components[
                "transverse_energy_fraction_above_%s_warning_threshold"
                % quantity] = (above_fraction, _DIMENSIONLESS)
        name = "radiationEventQuality_%s_%s" % (
            _clean(species_name), mode)
        self._write_component_group(
            field_group, name, components, accumulator, species_name, mode,
            {
                "longName": _bytes(
                    "centered pusher impulse invariants and physical "
                    "finite-step resolution"),
                "massShellResidualDefinition": _bytes(
                    "abs(U_center_squared_minus_one)_"
                    "over_Euclidean_four_velocity_norm_squared"),
                "orthogonalityResidualDefinition": _bytes(
                    "abs(U_center_dot_A)_over_product_of_Euclidean_norms"),
                "powerIdentityResidualDefinition": _bytes(
                    "abs(P_perp_plus_P_parallel_plus_C_A_A_squared)_"
                    "over_invariant_power"),
                "deltaEtaDefinition": _bytes(
                    "arcosh(U_plus_dot_U_minus)"),
                "deltaThetaUDefinition": _bytes(
                    "arccos(clipped_u_plus_hat_dot_u_minus_hat)"),
                "zeroMomentumTurnConvention": _bytes(
                    "delta_theta_u_equals_zero_if_either_endpoint_has_"
                    "zero_spatial_momentum"),
                "chiTurnDefinition": _bytes(
                    "gamma_center_simulation_times_delta_theta_u"),
                "resolutionWarningThresholds": _bytes(json.dumps(
                    accumulator.resolution_warning_thresholds,
                    sort_keys=True)),
                "resolutionWeighting": _bytes(
                    "exact_unthinned_transverse_radiation_energy"),
                "resolutionIndicatorsModifyEvents": np.uint32(0),
            })

    def _write_sampling(self, field_group, accumulator, species_name,
                        mode, sampling):
        components = {}
        for key, value in sampling.items():
            is_energy_variance = "energy_sampling_variance" in key
            dimension = (
                2.0 * _ENERGY_DIMENSION
                if is_energy_variance else _DIMENSIONLESS)
            variance = float(value[0])
            components[key] = (variance, dimension)
            if is_energy_variance and "/" in key:
                uncertainty_name = key.replace(
                    "energy_sampling_variance", "energy_sampling_uncertainty")
                components[uncertainty_name] = (
                    math.sqrt(max(variance, 0.0)), _ENERGY_DIMENSION)
        transverse_variance = float(
            sampling["transverse_energy_sampling_variance"][0])
        longitudinal_variance = float(
            sampling["longitudinal_energy_sampling_variance"][0])
        components["transverse_energy_sampling_uncertainty"] = (
            math.sqrt(max(transverse_variance, 0.0)), _ENERGY_DIMENSION)
        components["longitudinal_energy_sampling_uncertainty"] = (
            math.sqrt(max(longitudinal_variance, 0.0)), _ENERGY_DIMENSION)
        effective_weight = float(
            sampling["effective_sampled_weight"][0])
        effective_weight_squared = float(
            sampling["effective_sampled_weight_squared"][0])
        effective_sample_size = (
            effective_weight**2 / effective_weight_squared
            if effective_weight_squared > 0.0 else 0.0)
        components["effective_sample_size"] = (
            effective_sample_size, _DIMENSIONLESS)
        name = "radiationSampling_%s_%s" % (
            _clean(species_name), mode)
        self._write_component_group(
            field_group, name, components, accumulator, species_name, mode,
            {
                "longName": _bytes(
                    "diagnostic particle thinning and uncertainty"),
                "samplingFraction": accumulator.particle_sampling_fraction,
                "samplingEstimator": _bytes(
                    "Bernoulli_Horvitz_Thompson"),
                "uncertaintyScope": _bytes(
                    "particle_thinning_for_total_energy;"
                    "spectral_angular_packet_membership_for_stochastic_"
                    "aperture_estimators"),
                "apertureSamplingUncertaintyEstimator": _bytes(
                    "within_event_packet_replication_variance;"
                    "Bernoulli_Horvitz_Thompson_particle_thinning_variance;"
                    "single_packet_uses_conservative_Bernoulli_upper_bound"),
                "stochasticAperturePartition": _bytes(
                    "inside_plus_outside_equals_same_packet_estimator_"
                    "exactly_for_each_detector"),
                "unbiasedLinearObservables": np.uint32(1),
            })

    def _write_moments(self, field_group, accumulator, species_name,
                       mode, moments, source_z):
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
            uses_source_time = (
                "time" in selection["quantities"]
                or any(key in selection for key in (
                    "observer_time_range", "time_range")))
            if uses_source_time:
                reference = selection["time_reference"]
                if reference == "photon_direction":
                    attributes.update({
                        "sourceTimeReference": _bytes(reference),
                        "sourceTimeConvention": _bytes(
                            "sampled_photon_direction_source_time"),
                        "observerTimeDefinition": _bytes(
                            "t_observer_minus_sampled_photon_direction_dot_"
                            "r_over_c"),
                        "observerTimeConditioning": _bytes(
                            "direction_conditioned_radiation_phase_coordinate;"
                            "angle_marginals_mix_distinct_null_coordinates"),
                    })
                else:
                    attributes.update({
                        "sourceTimeReference": _bytes(reference),
                        "sourceTimeConvention": _bytes(
                            "fixed_detector_referenced_source_time"),
                        "observerTimeDefinition": _bytes(
                            "tau_D=t_observer-n_D_dot_r_observer/c"),
                        "observerTimeConditioning": _bytes(
                            "detector_referenced_source_attribution"),
                        "sourceTimeDetector": _bytes(reference),
                        "sourceTimeDetectorDirection": (
                            selection["time_direction"]),
                    })
            components = source_moment_components(
                stats, selection["quantities"])
            if selection_name in source_z:
                components.update(source_z_interval_components(
                    source_z[selection_name],
                    accumulator.source_z_interval_edges,
                    accumulator.source_z_intervals))
                attributes.update({
                    "sourceZCentralIntervalFractions":
                        np.asarray(accumulator.source_z_intervals),
                    "sourceZIntervalEdges":
                        accumulator.source_z_interval_edges,
                    "sourceZIntervalDefinition": _bytes(
                        "equal_tail_central_radiation_energy_interval"),
                    "sourceZIntervalEstimator": _bytes(
                        "mergeable_weighted_histogram_with_explicit_"
                        "underflow_and_overflow"),
                })
            self._write_component_group(
                field_group, name, components,
                accumulator, species_name, mode, attributes)

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

    def write(self, iteration=None, final_flush=False):
        """Write one dirty snapshot using its latest event as file index."""
        if not self.has_unwritten_events():
            return False
        latest_event = self.latest_unwritten_event()
        if latest_event is None:
            raise RuntimeError(
                "Radiation state is dirty but has no completed event index.")
        file_iteration, last_event_center = latest_event
        file_iteration = int(file_iteration)
        requested_iteration = (
            None if iteration is None else int(iteration))
        diagnostic = self.diagnostic
        file_handle = None
        successful = False
        self.final_flush = bool(final_flush)
        initially_on_gpu = {
            species_name: bool(
                diagnostic.accumulators[species_name]._on_gpu)
            for species_name in diagnostic.species_names}
        if diagnostic.use_cuda:
            for species_name in diagnostic.species_names:
                if not initially_on_gpu[species_name]:
                    continue
                radiator = diagnostic.species[
                    species_name].synchrotron_radiator
                radiator.receive_from_gpu()
        try:
            snapshots = self._snapshots()
            first = next(iter(diagnostic.accumulators.values()))
            observer_time = (
                first.gamma_boost * last_event_center
                + first.observer_translation[0] / c)
            observer_dt = first.gamma_boost * diagnostic.dt_sim
            filename = "data%08d.h5" % file_iteration
            fullpath = os.path.join(diagnostic.write_dir, "hdf5", filename)
            file_handle = diagnostic.open_file(fullpath)
            if file_handle is not None:
                diagnostic.setup_openpmd_file(
                    file_handle, file_iteration, observer_time, observer_dt)
                iteration_group = file_handle[
                    "/data/%d" % file_iteration]
                iteration_group.attrs["timeReferenceFrame"] = _bytes(
                    first.observer_frame)
                iteration_group.attrs["timeReferenceEvent"] = _bytes(
                    "simulation_origin_z_equals_zero_at_latest_included_"
                    "event_center")
                iteration_group.attrs["radiationOutputPhase"] = _bytes(
                    "post_centered_pusher_impulse")
                iteration_group.attrs["radiationFinalFlush"] = np.uint32(
                    final_flush)
                iteration_group.attrs["radiationWriteTrigger"] = _bytes(
                    "explicit_finalization" if final_flush
                    else "scheduled_cadence")
                iteration_group.attrs[
                    "radiationLastIncludedEventIteration"] = np.int64(
                    file_iteration)
                iteration_group.attrs[
                    "radiationLastIncludedEventCenterSimulation"] = (
                    last_event_center)
                if requested_iteration is not None:
                    iteration_group.attrs[
                        "radiationRequestedWriteIteration"] = np.int64(
                        requested_iteration)
                iteration_group.attrs["observerLorentzTransform"] = (
                    self._lorentz_matrix(first).ravel())
                iteration_group.attrs["observerLorentzTransformShape"] = (
                    np.array([4, 4], dtype=np.uint64))
                iteration_group.attrs["observerTranslation"] = (
                    first.observer_translation)
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
                        self._write_quality(
                            field_group, accumulator, species_name, mode,
                            snapshot["quality"], snapshot["resolution"])
                        self._write_sampling(
                            field_group, accumulator, species_name, mode,
                            snapshot["sampling"])
                        self._write_moments(
                            field_group, accumulator, species_name, mode,
                            snapshot["moments"], snapshot["source_z"])
                        self._write_pulse_metrics(
                            field_group, accumulator, species_name, mode,
                            snapshot["data"])
            successful = True
        finally:
            if file_handle is not None:
                file_handle.close()
            if successful:
                self.previous = self.pending_previous
                for species_name, accumulator in (
                        diagnostic.accumulators.items()):
                    self.last_written_events[species_name] = (
                        accumulator.completed_event_count)
                    accumulator.reset_interval_state()
            self.final_flush = False
            if diagnostic.use_cuda:
                for species_name in diagnostic.species_names:
                    if not initially_on_gpu[species_name]:
                        continue
                    radiator = diagnostic.species[
                        species_name].synchrotron_radiator
                    radiator.send_to_gpu()
        return True

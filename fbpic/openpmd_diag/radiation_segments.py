# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Canonical checkpoint-aligned observer-radiation segment files."""

import hashlib
import json
import math
import os
import subprocess
import uuid

import h5py
import numpy as np

from fbpic import __version__ as fbpic_version


RADIATION_SEGMENT_SCHEMA_VERSION = 1
RADIATION_CONFIGURATION_SCHEMA_VERSION = 1

_SUM_CATEGORIES = ("data", "accounting", "sampling", "source_z")
_STATE_CATEGORIES = (
    "data", "accounting", "sampling", "source_z",
    "quality", "resolution", "moments",
)
_TIMING_KEYS = (
    "event_count", "first_event_index", "last_event_index",
    "first_event_center", "last_event_center",
    "represented_interval_start", "represented_interval_end",
)


def _host_array(value):
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {
            str(key): _jsonable(value[key])
            for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if math.isinf(value):
            return {"float": "inf" if value > 0.0 else "-inf"}
        if value == 0.0:
            return 0.0
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _restore_jsonable(value):
    """Decode tagged non-finite floats used by canonical JSON."""
    if isinstance(value, dict):
        if set(value) == {"float"}:
            tag = value["float"]
            if tag == "inf":
                return math.inf
            if tag == "-inf":
                return -math.inf
            if tag == "nan":
                return math.nan
        return {key: _restore_jsonable(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_jsonable(item) for item in value]
    return value

def canonical_json(value):
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True)


def _array_digest(value):
    array = np.ascontiguousarray(_host_array(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _normalized_detector(detector):
    return {
        "name": detector["name"],
        "direction": detector["direction"],
        "time_edges": detector["time_edges"],
        "half_angle": detector["half_angle"],
        "aperture_quadrature": detector["aperture_quadrature"],
        "energy_bands": detector["energy_bands"],
        "energy_band_mode": detector["energy_band_mode"],
        "pulse_interval": detector["pulse_interval"],
        "quadrature_rays_digest": (
            None if detector.get("rays") is None
            else _array_digest(detector["rays"])),
        "quadrature_weights_digest": (
            None if detector.get("ray_weights") is None
            else _array_digest(detector["ray_weights"])),
    }


def diagnostic_configuration(diagnostic):
    """Return normalized configuration and its deterministic fingerprint."""
    species_configurations = {}
    for species_name in sorted(diagnostic.species_names):
        accumulator = diagnostic.accumulators[species_name]
        particle_species = diagnostic.species[species_name]
        lookup_digests = {}
        for name in (
                "spectral_x", "spectral_cdf",
                "angular_kernel_x", "angular_kernel_log_x",
                "angular_kernel_probability", "angular_kernel_inverse",
                "joint_band_x", "joint_band_log_x", "joint_band_y",
                "joint_band_y_coordinate", "joint_band_cdf"):
            value = getattr(accumulator, name, None)
            if value is not None:
                lookup_digests[name] = _array_digest(value)

        species_configurations[species_name] = {
            "observer_frame": accumulator.observer_frame,
            "observer_frame_gamma": accumulator.gamma_boost,
            "observer_frame_beta": accumulator.beta_boost,
            "observer_translation": accumulator.observer_translation,
            "simulation_timestep": accumulator.dt_sim,
            "enabled_channels": sorted(accumulator.enabled_channels),
            "photon_energy_edges": accumulator.energy_edges,
            "theta_x_edges": accumulator.theta_x_edges,
            "theta_y_edges": accumulator.theta_y_edges,
            "angular_measure": accumulator.angular_measure,
            "angular_coordinate_convention":
                "theta_x=atan2(n_x,n_z);theta_y=atan2(n_y,n_z)",
            "detectors": [
                _normalized_detector(item) for item in accumulator.detectors],
            "observer_time_edges": accumulator.common_time_edges,
            "source_axis_edges": accumulator.source_axis_edges,
            "source_projections": accumulator.source_projections,
            "source_moment_selections":
                accumulator.source_moment_selections,
            "source_z_intervals": accumulator.source_z_intervals,
            "source_z_interval_edges":
                accumulator.source_z_interval_edges,
            "source_time_convention":
                "direction_conditioned_or_named_fixed_detector",
            "gamma_cutoff": accumulator.gamma_cutoff,
            "particle_selection": accumulator.particle_selection,
            "energy_band_mode": accumulator.energy_band_mode,
            "samples_per_particle": accumulator.samples_per_particle,
            "particle_sampling_fraction":
                accumulator.particle_sampling_fraction,
            "random_seed": int(accumulator.random_seed),
            "random_namespace": int(accumulator.random_namespace),
            "random_stream_ids": accumulator.random_stream_ids,
            "resolution_warning_thresholds":
                accumulator.resolution_warning_thresholds,
            "spectral_closure_maximum_x":
                float(_host_array(accumulator.spectral_x)[-1]),
            "spectral_closure_truncated_fraction":
                accumulator.spectral_truncated_fraction,
            "spectral_angular_closure":
                "polarization_summed_Schwinger_vertical_conditional",
            "macroparticle_closure": "incoherent_linear_in_weight",
            "event_model": "centered_covariant_pusher_impulse_v1",
            "species_charge": float(particle_species.q),
            "species_mass": float(particle_species.m),
            "lookup_array_digests": lookup_digests,
        }

    descriptor = {
        "configurationSchemaVersion":
            RADIATION_CONFIGURATION_SCHEMA_VERSION,
        "radiationStateSchemaVersion": RADIATION_SEGMENT_SCHEMA_VERSION,
        "diagnosticId": diagnostic.diagnostic_id,
        "species": species_configurations,
    }
    serialized = canonical_json(descriptor)
    fingerprint = hashlib.sha256(serialized.encode("ascii")).hexdigest()
    return _jsonable(descriptor), fingerprint


def code_revision():
    override = os.environ.get("FBPIC_GIT_REVISION")
    if override:
        return override
    package_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=package_root,
            stderr=subprocess.DEVNULL).decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_text_dataset(group, name, value):
    group.create_dataset(name, data=np.bytes_(str(value)))


def _read_text_dataset(group, name):
    value = group[name][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _write_named_arrays(group, values):
    for key in sorted(values):
        parts = str(key).strip("/").split("/")
        target = group
        for part in parts[:-1]:
            target = target.require_group(part)
        target.create_dataset(parts[-1], data=_host_array(values[key]))


def _read_named_arrays(group):
    output = {}

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            output[name] = np.asarray(item[()])

    group.visititems(visitor)
    return output


def _write_state(root, state):
    for category in _STATE_CATEGORIES:
        _write_named_arrays(
            root.require_group(category), state.get(category, {}))
    timing = root.require_group("timing")
    for key in _TIMING_KEYS:
        timing.attrs[key] = state["timing"][key]


def _read_state(root):
    state = {
        category: _read_named_arrays(root[category])
        for category in _STATE_CATEGORIES}
    state["timing"] = {
        key: root["timing"].attrs[key].item()
        if isinstance(root["timing"].attrs[key], np.generic)
        else root["timing"].attrs[key]
        for key in _TIMING_KEYS}
    return state


def write_segment(path, metadata, descriptor, fingerprint, states):
    """Atomically write one closed, canonical radiation segment."""
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = "%s.%s.partial" % (path, uuid.uuid4().hex)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["artifactType"] = "radiation_segment"
            output.attrs["radiationStateSchemaVersion"] = (
                RADIATION_SEGMENT_SCHEMA_VERSION)
            output.attrs["segmentStatus"] = "closed"
            output.attrs["segmentId"] = metadata["segmentId"]
            output.attrs["runId"] = metadata["runId"]
            output.attrs["diagnosticId"] = metadata["diagnosticId"]
            output.attrs["configurationFingerprint"] = fingerprint
            output.attrs["eventBegin"] = np.int64(metadata["eventBegin"])
            output.attrs["eventEndExclusive"] = np.int64(
                metadata["eventEndExclusive"])
            output.attrs["accumulationScope"] = "segment"
            output.attrs["closeReason"] = metadata["closeReason"]
            output.attrs["commitManifest"] = metadata["commitManifest"]
            output.attrs["fbpicVersion"] = fbpic_version
            output.attrs["codeRevision"] = metadata["codeRevision"]
            _write_text_dataset(
                output.require_group("metadata"), "json",
                canonical_json(metadata))
            _write_text_dataset(
                output.require_group("configuration"), "json",
                canonical_json(descriptor))
            state_root = output.require_group("radiationState")
            for species_name in sorted(states):
                _write_state(
                    state_root.require_group(species_name),
                    states[species_name])
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _manifest_references_segment(manifest, identity):
    """Verify the complete segment/checkpoint commit relationship."""
    expected_path = os.path.abspath(identity["path"])
    identity_keys = (
        "segmentId", "runId", "diagnosticId", "parentCheckpointId",
        "parentCheckpointIteration", "closingCheckpointId",
        "closingCheckpointIteration", "eventBegin", "eventEndExclusive",
        "configurationFingerprint", "closeReason",
    )
    for reference in manifest.get("segments", []):
        if os.path.abspath(reference.get("path", "")) != expected_path:
            continue
        if any(reference.get(key) != identity[key]
               for key in identity_keys):
            continue
        if manifest.get("checkpointStatus") == "committed":
            if (manifest.get("checkpointManifestSchemaVersion") != 1
                    or identity["closeReason"] != "checkpoint"
                    or manifest.get("checkpointId")
                    != identity["closingCheckpointId"]
                    or manifest.get("iteration")
                    != identity["closingCheckpointIteration"]
                    or manifest.get("parentCheckpointId")
                    != identity["parentCheckpointId"]
                    or manifest.get("parentCheckpointIteration")
                    != identity["parentCheckpointIteration"]):
                continue
            if int(manifest.get("eventEndExclusive", -1)) != int(
                    identity["eventEndExclusive"]):
                continue
        elif manifest.get("segmentStatus") == "committed":
            if (manifest.get("radiationSegmentManifestSchemaVersion") != 1
                    or manifest.get("manifestType")
                    != "radiation_finalization"):
                continue
            if (identity["closingCheckpointId"] is not None
                    or identity["closingCheckpointIteration"] is not None
                    or identity["closeReason"] != "finalize"):
                continue
            if (manifest.get("runId") != identity["runId"]
                    or manifest.get("diagnosticId")
                    != identity["diagnosticId"]
                    or manifest.get("configurationFingerprint")
                    != identity["configurationFingerprint"]
                    or int(manifest.get("eventEndExclusive", -1)) != int(
                        identity["eventEndExclusive"])):
                continue
        else:
            continue
        return True
    return False


def _attribute_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def radiation_segment_status(path):
    """Return committed or orphaned for a persisted segment artifact."""
    path = os.path.abspath(path)
    if path.endswith(".partial") or not os.path.isfile(path):
        return "orphaned"
    try:
        with h5py.File(path, "r") as source:
            if _attribute_text(source.attrs.get("artifactType", "")) != (
                    "radiation_segment"):
                return "orphaned"
            if _attribute_text(source.attrs.get("segmentStatus", "")) != (
                    "closed"):
                return "orphaned"
            if int(source.attrs.get("radiationStateSchemaVersion", -1)) != (
                    RADIATION_SEGMENT_SCHEMA_VERSION):
                return "orphaned"
            if _attribute_text(source.attrs.get("accumulationScope", "")) != (
                    "segment"):
                return "orphaned"
            attributes = {
                "segmentId": _attribute_text(source.attrs["segmentId"]),
                "runId": _attribute_text(source.attrs["runId"]),
                "diagnosticId": _attribute_text(
                    source.attrs["diagnosticId"]),
                "configurationFingerprint": _attribute_text(
                    source.attrs["configurationFingerprint"]),
                "eventBegin": int(source.attrs["eventBegin"]),
                "eventEndExclusive": int(source.attrs[
                    "eventEndExclusive"]),
            }
            metadata = json.loads(_read_text_dataset(
                source["metadata"], "json"))
            if any(metadata.get(key) != value
                   for key, value in attributes.items()):
                return "orphaned"
            if (metadata.get("artifactType") != "radiation_segment"
                    or metadata.get("segmentStatus") != "closed"
                    or metadata.get("accumulationScope") != "segment"
                    or metadata.get("radiationStateSchemaVersion")
                    != RADIATION_SEGMENT_SCHEMA_VERSION):
                return "orphaned"
            close_reason = _attribute_text(
                source.attrs.get("closeReason", ""))
            if metadata.get("closeReason") != close_reason:
                return "orphaned"
            manifest_path = _attribute_text(
                source.attrs.get("commitManifest", ""))
            if (not manifest_path
                    or os.path.abspath(metadata.get("commitManifest", ""))
                    != os.path.abspath(manifest_path)):
                return "orphaned"
            identity = dict(attributes)
            identity.update({
                "path": path,
                "parentCheckpointId": metadata.get("parentCheckpointId"),
                "parentCheckpointIteration": metadata.get(
                    "parentCheckpointIteration"),
                "closingCheckpointId": metadata.get("closingCheckpointId"),
                "closingCheckpointIteration": metadata.get(
                    "closingCheckpointIteration"),
                "closeReason": close_reason,
            })
    except (OSError, KeyError, TypeError, ValueError, UnicodeError):
        return "orphaned"
    if not os.path.isfile(manifest_path):
        return "orphaned"
    try:
        with open(manifest_path, "r") as source:
            manifest = json.load(source)
    except (OSError, ValueError):
        return "orphaned"
    if _manifest_references_segment(manifest, identity):
        return "committed"
    return "orphaned"


def read_segment(path, require_committed=True):
    """Read and validate a canonical segment."""
    path = os.path.abspath(path)
    status = radiation_segment_status(path)
    if require_committed and status != "committed":
        raise RuntimeError(
            "Radiation segment is not committed: %s" % path)
    with h5py.File(path, "r") as source:
        if _attribute_text(source.attrs.get("artifactType", "")) != (
                "radiation_segment"):
            raise ValueError("Not a canonical radiation segment: %s" % path)
        if int(source.attrs["radiationStateSchemaVersion"]) != (
                RADIATION_SEGMENT_SCHEMA_VERSION):
            raise ValueError("Incompatible radiation state schema.")
        attributes = {
            "segmentId": _attribute_text(source.attrs["segmentId"]),
            "runId": _attribute_text(source.attrs["runId"]),
            "diagnosticId": _attribute_text(source.attrs["diagnosticId"]),
            "configurationFingerprint": _attribute_text(
                source.attrs["configurationFingerprint"]),
            "eventBegin": int(source.attrs["eventBegin"]),
            "eventEndExclusive": int(source.attrs["eventEndExclusive"]),
        }
        metadata = json.loads(_read_text_dataset(source["metadata"], "json"))
        descriptor = json.loads(
            _read_text_dataset(source["configuration"], "json"))
        states = {
            species_name: _read_state(source["radiationState"][species_name])
            for species_name in source["radiationState"]}
        fingerprint = attributes["configurationFingerprint"]
    calculated = hashlib.sha256(
        canonical_json(descriptor).encode("ascii")).hexdigest()
    if calculated != fingerprint:
        raise ValueError("Radiation configuration fingerprint is corrupt.")
    for key in (
            "segmentId", "runId", "diagnosticId",
            "configurationFingerprint", "eventBegin", "eventEndExclusive"):
        if metadata.get(key) != attributes[key]:
            raise ValueError(
                "Radiation segment metadata disagrees with `%s`." % key)
    begin = attributes["eventBegin"]
    end = attributes["eventEndExclusive"]
    if end < begin:
        raise ValueError("Radiation segment has a reversed event range.")
    expected = end - begin
    if set(states) != set(descriptor.get("species", {})):
        raise ValueError(
            "Radiation state species disagree with the configuration.")
    for species_name, state in states.items():
        timing = state["timing"]
        count = int(timing["event_count"])
        if count != expected:
            raise ValueError(
                "Radiation state for `%s` contains %d events, expected %d."
                % (species_name, count, expected))
        if expected and (
                int(timing["first_event_index"]) != begin
                or int(timing["last_event_index"]) != end - 1):
            raise ValueError(
                "Radiation state timing does not cover its declared range.")
    expected_first = begin if expected else None
    expected_last = end - 1 if expected else None
    if (metadata.get("firstRadiationEventIndex") != expected_first
            or metadata.get("lastRadiationEventIndex") != expected_last):
        raise ValueError(
            "Radiation segment first/last event metadata is inconsistent.")
    return {
        "path": path,
        "status": status,
        "metadata": metadata,
        "descriptor": descriptor,
        "fingerprint": fingerprint,
        "states": states,
    }


def _copy_state(state):
    copied = {
        category: {
            key: np.array(value, copy=True)
            for key, value in state[category].items()}
        for category in _STATE_CATEGORIES
    }
    copied["timing"] = dict(state["timing"])
    return copied


def _require_same_keys(first, second, label):
    if set(first) != set(second):
        raise ValueError("Incompatible radiation %s keys." % label)


def _merge_timing(first, second):
    if int(first["event_count"]) == 0:
        return dict(second)
    if int(second["event_count"]) == 0:
        return dict(first)
    return {
        "event_count":
            int(first["event_count"]) + int(second["event_count"]),
        "first_event_index": min(
            int(first["first_event_index"]),
            int(second["first_event_index"])),
        "last_event_index": max(
            int(first["last_event_index"]),
            int(second["last_event_index"])),
        "first_event_center": min(
            float(first["first_event_center"]),
            float(second["first_event_center"])),
        "last_event_center": max(
            float(first["last_event_center"]),
            float(second["last_event_center"])),
        "represented_interval_start": min(
            float(first["represented_interval_start"]),
            float(second["represented_interval_start"])),
        "represented_interval_end": max(
            float(first["represented_interval_end"]),
            float(second["represented_interval_end"])),
    }


def _merge_state(target, incoming):
    from .observer_radiation_diag import merge_source_moment_stats

    for category in _STATE_CATEGORIES:
        _require_same_keys(target[category], incoming[category], category)
        for key in target[category]:
            if (target[category][key].shape
                    != incoming[category][key].shape):
                raise ValueError(
                    "Incompatible radiation %s shape for `%s`."
                    % (category, key))

    for category in _SUM_CATEGORIES:
        for key in target[category]:
            target[category][key] += incoming[category][key]
    for key in target["quality"]:
        target["quality"][key] = np.maximum(
            target["quality"][key], incoming["quality"][key])
    for key in target["resolution"]:
        target["resolution"][key][0] = max(
            target["resolution"][key][0], incoming["resolution"][key][0])
        target["resolution"][key][1:] += incoming["resolution"][key][1:]
    for key in target["moments"]:
        target["moments"][key] = merge_source_moment_stats(
            target["moments"][key], incoming["moments"][key])
    target["timing"] = _merge_timing(
        target["timing"], incoming["timing"])


def _component_value_dimension(item):
    value, dimension = item
    return float(value), np.asarray(dimension, dtype=np.float64)


def _write_derived(output, states, descriptor):
    from .observer_radiation_diag import (
        pulse_components, source_moment_components,
        source_z_interval_components,
    )

    derived = output.require_group("derived")
    configurations = descriptor["species"]
    for species_name in sorted(states):
        state = states[species_name]
        config = _restore_jsonable(configurations[species_name])
        species_group = derived.require_group(species_name)

        selections = {
            item["name"]: item
            for item in config["source_moment_selections"]}
        moment_group = species_group.require_group("sourceMoments")
        for name, statistics in sorted(state["moments"].items()):
            quantities = selections[name]["quantities"]
            components = source_moment_components(statistics, quantities)
            if name in state["source_z"]:
                edges = np.asarray(
                    config["source_z_interval_edges"], dtype=np.float64)
                fractions = tuple(config["source_z_intervals"])
                components.update(source_z_interval_components(
                    state["source_z"][name], edges, fractions))
            selection_group = moment_group.require_group(name)
            for component_name, item in sorted(components.items()):
                value, dimension = _component_value_dimension(item)
                dataset = selection_group.create_dataset(
                    component_name, data=np.asarray([value]))
                dataset.attrs["unitDimension"] = dimension

        pulse_group = species_group.require_group("pulseMetrics")
        detectors = {
            item["name"]: item for item in config["detectors"]}
        for key, raw in sorted(state["data"].items()):
            parts = key.split("/")
            if len(parts) != 3 or parts[0] != "detector":
                continue
            detector_name, channel = parts[1], parts[2]
            if channel not in ("direction", "aperture"):
                continue
            detector = detectors[detector_name]
            components = pulse_components(
                raw, np.asarray(detector["time_edges"], dtype=np.float64),
                channel == "direction", detector["pulse_interval"])
            channel_group = pulse_group.require_group(
                "%s/%s" % (detector_name, channel))
            for component_name, item in sorted(components.items()):
                value, dimension = _component_value_dimension(item)
                dataset = channel_group.create_dataset(
                    component_name, data=np.asarray([value]))
                dataset.attrs["unitDimension"] = dimension


def merge_radiation_segments(segment_paths, output_path):
    """Strictly merge one selected, committed radiation lineage.

    Segments are ordered by their half-open absolute event ranges. Gaps,
    overlaps, broken checkpoint ancestry, duplicate IDs, incompatible
    configurations, and uncommitted files are rejected. Rebinning is never
    implicit.
    """
    paths = [os.path.abspath(path) for path in segment_paths]
    if not paths:
        raise ValueError("At least one radiation segment is required.")
    records = [read_segment(path, require_committed=True) for path in paths]
    records.sort(key=lambda item: (
        int(item["metadata"]["eventBegin"]),
        int(item["metadata"]["eventEndExclusive"])))

    fingerprint = records[0]["fingerprint"]
    run_id = records[0]["metadata"]["runId"]
    descriptor_json = canonical_json(records[0]["descriptor"])
    seen_ids = set()
    previous = None
    for record in records:
        metadata = record["metadata"]
        segment_id = metadata["segmentId"]
        if segment_id in seen_ids:
            raise ValueError("Duplicate radiation segment ID.")
        seen_ids.add(segment_id)
        if record["fingerprint"] != fingerprint:
            raise ValueError("Incompatible radiation configuration fingerprint.")
        if canonical_json(record["descriptor"]) != descriptor_json:
            raise ValueError("Fingerprint collision or configuration mismatch.")
        if metadata["runId"] != run_id:
            raise ValueError("Radiation segments belong to different runs.")
        begin = int(metadata["eventBegin"])
        end = int(metadata["eventEndExclusive"])
        if end < begin:
            raise ValueError("Radiation segment has a reversed event range.")
        if previous is not None:
            previous_end = int(previous["metadata"]["eventEndExclusive"])
            if begin > previous_end:
                raise ValueError("Gap between radiation segments.")
            if begin < previous_end:
                raise ValueError("Overlap between radiation segments.")
            if (previous["metadata"].get("closingCheckpointId")
                    != metadata.get("parentCheckpointId")):
                raise ValueError("Broken checkpoint ancestry between segments.")
        previous = record

    species_names = set(records[0]["states"])
    merged = {
        name: _copy_state(records[0]["states"][name])
        for name in species_names}
    for record in records[1:]:
        if set(record["states"]) != species_names:
            raise ValueError("Incompatible radiation species sets.")
        for species_name in species_names:
            _merge_state(merged[species_name], record["states"][species_name])

    origin = records[0]["metadata"].get("continuity") in (
        "genesis", "genesis_after_finalize")
    transitions_are_seamless = all(
        item["metadata"].get("continuity") == "seamless"
        for item in records[1:])
    terminal = records[-1]["metadata"].get("closeReason") == "finalize"
    complete_lineage = bool(origin and transitions_are_seamless and terminal)
    artifact_type = (
        "radiation_merged_whole_run" if complete_lineage
        else "radiation_merged_lineage_selection")
    metadata = {
        "artifactType": artifact_type,
        "radiationStateSchemaVersion": RADIATION_SEGMENT_SCHEMA_VERSION,
        "runId": run_id,
        "configurationFingerprint": fingerprint,
        "eventBegin": int(records[0]["metadata"]["eventBegin"]),
        "eventEndExclusive":
            int(records[-1]["metadata"]["eventEndExclusive"]),
        "segments": [
            dict(item["metadata"], path=item["path"])
            for item in records],
        "seamless": bool(origin and transitions_are_seamless),
        "completeLineage": complete_lineage,
    }

    output_path = os.path.abspath(output_path)
    directory = os.path.dirname(output_path)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = "%s.%s.partial" % (output_path, uuid.uuid4().hex)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs["artifactType"] = artifact_type
            output.attrs["mergedWholeRun"] = np.uint32(complete_lineage)
            output.attrs["accumulationScope"] = (
                "whole_run" if complete_lineage else "selected_lineage")
            output.attrs["runId"] = run_id
            output.attrs["configurationFingerprint"] = fingerprint
            output.attrs["radiationStateSchemaVersion"] = (
                RADIATION_SEGMENT_SCHEMA_VERSION)
            _write_text_dataset(
                output.require_group("metadata"), "json",
                canonical_json(metadata))
            _write_text_dataset(
                output.require_group("configuration"), "json",
                descriptor_json)
            state_root = output.require_group("radiationState")
            for species_name in sorted(merged):
                _write_state(
                    state_root.require_group(species_name),
                    merged[species_name])
            _write_derived(output, merged, records[0]["descriptor"])
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return output_path

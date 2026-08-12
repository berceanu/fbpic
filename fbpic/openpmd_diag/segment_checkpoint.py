# Copyright 2026, FBPIC contributors
# License: 3-Clause-BSD-LBNL
"""Checkpoint commit manifests and generic diagnostic segment hooks."""

import json
import os
import uuid

import h5py

from fbpic.utils.mpi import comm


_CHECKPOINT_MANIFEST_SCHEMA = 1


def collective_uuid():
    """Return one UUID shared by the active MPI communicator."""
    value = uuid.uuid4().hex if comm.rank == 0 else None
    if comm.size > 1:
        value = comm.bcast(value, root=0)
    return value


def atomic_write_json(path, payload):
    """Atomically publish a small JSON commit record."""
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    temporary = "%s.%s.partial" % (path, uuid.uuid4().hex)
    try:
        with open(temporary, "w") as output:
            json.dump(payload, output, sort_keys=True, indent=2)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def read_json(path):
    with open(path, "r") as source:
        return json.load(source)


def manifest_directory(checkpoint_dir):
    return os.path.join(os.path.abspath(checkpoint_dir), "manifests")


def checkpoint_manifest_path(checkpoint_dir, iteration):
    return os.path.join(
        manifest_directory(checkpoint_dir),
        "checkpoint%08d.json" % int(iteration))


def active_manifest_path(checkpoint_dir):
    return os.path.join(manifest_directory(checkpoint_dir), "active.json")


def selected_checkpoint_manifest(checkpoint_dir, iteration=None):
    """Return committed metadata for an explicit or active checkpoint."""
    if iteration is None:
        path = active_manifest_path(checkpoint_dir)
    else:
        path = checkpoint_manifest_path(checkpoint_dir, iteration)
    if not os.path.isfile(path):
        return None
    manifest = read_json(path)
    if manifest.get("checkpointManifestSchemaVersion") != (
            _CHECKPOINT_MANIFEST_SCHEMA):
        raise RuntimeError("Unsupported checkpoint manifest schema.")
    if manifest.get("checkpointStatus") != "committed":
        raise RuntimeError("The selected checkpoint is not committed.")
    if iteration is not None and int(manifest["iteration"]) != int(iteration):
        raise RuntimeError("Checkpoint manifest iteration mismatch.")
    for relative in manifest.get("checkpointFiles", []):
        checkpoint_path = (
            relative if os.path.isabs(relative)
            else os.path.join(os.path.abspath(checkpoint_dir), relative))
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                "A committed checkpoint payload is missing: %s"
                % checkpoint_path)
    for reference in manifest.get("segments", []):
        segment_path = reference.get("path")
        if not segment_path or not os.path.isfile(segment_path):
            raise RuntimeError(
                "A committed diagnostic segment is missing from the selected "
                "checkpoint.")
    return manifest


def _stamp_checkpoint_file(path, context):
    """Attach identity metadata without serializing diagnostic state."""
    with h5py.File(path, "a") as output:
        output.attrs["checkpointId"] = context["checkpoint_id"]
        output.attrs["checkpointRunId"] = context["run_id"]
        output.attrs["checkpointParentId"] = (
            context["parent_checkpoint_id"] or "")
        output.attrs["checkpointManifestSchemaVersion"] = (
            _CHECKPOINT_MANIFEST_SCHEMA)
        output.attrs["checkpointCommitState"] = "prepared"


class CheckpointSet(object):
    """Write one complete checkpoint and commit diagnostic segments with it."""

    def __init__(self, sim, period, checkpoint_dir, components):
        self.sim = sim
        self.period = int(period)
        if self.period < 1:
            raise ValueError("Checkpoint period must be positive.")
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.components = list(components)
        if comm.rank == 0:
            directory = manifest_directory(self.checkpoint_dir)
            if not os.path.isdir(directory):
                os.makedirs(directory)
        comm.barrier()

        if getattr(sim, "_run_id", None) is None:
            sim._run_id = collective_uuid()
        elif comm.size > 1:
            sim._run_id = comm.bcast(
                sim._run_id if comm.rank == 0 else None, root=0)
        sim._checkpoint_dir = self.checkpoint_dir

    def _context(self, iteration, checkpoint_id, manifest_path):
        return {
            "run_id": self.sim._run_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_iteration": int(iteration),
            "parent_checkpoint_id": getattr(
                self.sim, "_checkpoint_parent_id", None),
            "parent_checkpoint_iteration": getattr(
                self.sim, "_checkpoint_parent_iteration", None),
            "event_end_exclusive": int(iteration),
            "close_reason": "checkpoint",
            "commit_manifest": os.path.abspath(manifest_path),
        }

    def write(self, iteration):
        if int(iteration) % self.period != 0:
            return False

        checkpoint_id = collective_uuid()
        manifest_path = checkpoint_manifest_path(
            self.checkpoint_dir, iteration)
        context = self._context(iteration, checkpoint_id, manifest_path)

        checkpoint_files = [
            os.path.join(
                "proc%d" % rank, "hdf5",
                "data%08d.h5" % int(iteration))
            for rank in range(comm.size)]
        preparing_manifest = {
            "checkpointManifestSchemaVersion": _CHECKPOINT_MANIFEST_SCHEMA,
            "checkpointStatus": "preparing",
            "runId": context["run_id"],
            "checkpointId": checkpoint_id,
            "parentCheckpointId": context["parent_checkpoint_id"],
            "parentCheckpointIteration":
                context["parent_checkpoint_iteration"],
            "iteration": int(iteration),
            "eventEndExclusive": int(iteration),
            "checkpointFiles": checkpoint_files,
            "segments": [],
        }
        if comm.rank == 0:
            atomic_write_json(manifest_path, preparing_manifest)
        comm.barrier()
        # Existing field and particle writers remain responsible for the
        # checkpoint payload. The commit record below is the acceptance marker.
        for component in self.components:
            component.write(iteration)

        checkpoint_file = os.path.join(
            self.checkpoint_dir, "proc%d" % comm.rank, "hdf5",
            "data%08d.h5" % int(iteration))
        _stamp_checkpoint_file(checkpoint_file, context)
        comm.barrier()

        segment_references = []
        for diagnostic in self.sim.diags:
            close = getattr(diagnostic, "close_segment", None)
            if close is not None:
                reference = close(context)
                if reference is not None:
                    segment_references.append(reference)

        manifest = {
            "checkpointManifestSchemaVersion": _CHECKPOINT_MANIFEST_SCHEMA,
            "checkpointStatus": "committed",
            "runId": context["run_id"],
            "checkpointId": checkpoint_id,
            "parentCheckpointId": context["parent_checkpoint_id"],
            "parentCheckpointIteration":
                context["parent_checkpoint_iteration"],
            "iteration": int(iteration),
            "eventEndExclusive": int(iteration),
            "checkpointFiles": checkpoint_files,
            "segments": segment_references,
        }
        if comm.rank == 0:
            atomic_write_json(manifest_path, manifest)
            atomic_write_json(active_manifest_path(self.checkpoint_dir), manifest)
        comm.barrier()

        next_context = {
            "run_id": context["run_id"],
            "checkpoint_id": checkpoint_id,
            "checkpoint_iteration": int(iteration),
            "iteration": int(iteration),
            "restart": False,
            "segments": segment_references,
            "committed_manifest": os.path.abspath(manifest_path),
        }
        for diagnostic in self.sim.diags:
            start = getattr(diagnostic, "start_segment", None)
            if start is not None:
                start(next_context)

        self.sim._checkpoint_parent_id = checkpoint_id
        self.sim._checkpoint_parent_iteration = int(iteration)
        self.sim._checkpoint_restart_context = None
        return True

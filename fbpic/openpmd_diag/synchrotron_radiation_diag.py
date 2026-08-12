# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""Public configuration for observer-frame synchrotron products."""

import os
import uuid

import numpy as np
from scipy.constants import c

from fbpic.particles.tracking import ParticleTracker
from fbpic.utils.mpi import comm as comm_world
from .generic_diag import OpenPMDDiagnostic
from .observer_radiation_diag import ObserverRadiationWriter
from .radiation_segments import (
    code_revision, diagnostic_configuration, radiation_segment_status,
    write_segment,
)
from .segment_checkpoint import (
    atomic_write_json, relative_file_reference,
    relative_segment_reference,
)

def _stable_random_namespace(name):
    """Return a reproducible uint64 namespace for one output species."""
    value = 0xCBF29CE484222325
    for byte in str(name).encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & ((1 << 64) - 1)
    return value



class SynchrotronRadiationDiagnostic(OpenPMDDiagnostic):
    """Write passive, incoherent observer-frame synchrotron diagnostics.

    The particle species must first call
    :meth:`fbpic.particles.Particles.activate_synchrotron`. Product arrays
    are allocated only for the requested channels.

    Parameters
    ----------
    period, dt_period : int or float, optional
        Output cadence, expressed in iterations or simulation-frame seconds.
        Specify exactly one through the standard diagnostic convention.
    species : dict
        Mapping from output names to activated electron or positron species.
    comm : BoundaryCommunicator, optional
        Communicator used by the simulation.
    write_dir : str, optional
        Parent output directory.
    iteration_min, iteration_max : int, optional
        Half-open iteration range in which output is written.
    observer_frame : {'laboratory', 'simulation'}
        Frame in which products, cuts, source coordinates, and bin edges are
        defined.
    boost : BoostConverter, optional
        Explicit simulation-to-laboratory longitudinal boost. When omitted,
        the transform supplied at species activation is used.
    observer_translation : array-like, optional
        Four-translation ``(ct, x, y, z)`` applied after the boost.
    photon_energy_edges : array-like, optional
        Photon-energy bin edges in joules.
    theta_x_edges, theta_y_edges : array-like, optional
        Projected-angle bin edges. They must lie strictly inside
        ``(-pi/2, pi/2)``.
    angular_measure : {'solid_angle', 'projected_angles'}
        Density measure for products containing both angular axes.
    detectors : sequence of dicts, optional
        Fixed directions or circular apertures with observer-time edges and
        optional energy bands. A detector may override energy_band_mode.
    observer_time_edges : array-like, optional
        Default detector and source observer-time edges in seconds.
    source_coordinate_edges : dict, optional
        Bin edges for any of ``x``, ``y``, and ``z``.
    source_projections : sequence, optional
        Source-distribution projections over available position, angle,
        photon-energy, and time axes. Dictionaries containing a time axis may
        set time_reference to photon_direction (the default) or a configured
        detector name.
    source_moments : sequence, optional
        Named radiation selections for mergeable source statistics. Each may
        use the same photon-direction or detector time_reference convention.
    source_z_intervals : float or sequence of float, optional
        Equal-tail central radiation-energy fractions to report in source
        ``z`` for every source-moment selection, for example ``(0.5, 0.9)``.
    source_z_interval_edges : array-like, optional
        Histogram edges used by the mergeable source-``z`` interval estimator.
        Defaults to ``source_coordinate_edges['z']`` when available.
    channels : sequence or dict, optional
        Any of ``angular_spectral``, ``observer_time``, ``source``,
        ``source_moments``, and ``accounting``. They are inferred when
        omitted.
    output_mode : {'cumulative', 'interval', 'both'}
        Whether records contain all emission to date, only the last output
        interval, or both.
    samples_per_particle : int
        Fixed number of stratified spectral-angular packets per emitting
        particle and PIC event.
    particle_batch_size : int
        Maximum particle count in temporary CPU/GPU arrays.
    particle_selection : dict, optional
        Observer-frame particle ranges applied before emission accumulation.
    gamma_cutoff : float, optional
        Diagnostic-specific observer-frame threshold. Defaults to the value
        supplied at species activation and must be greater than one.
    energy_band_mode : {'joint', 'separable'}
        Default energy-band detector closure. Joint retains the local photon
        energy/emission-angle correlation; separable is the explicitly
        labeled fast approximation.
    random_seed : int
        Diagnostic seed for stateless sampling. Keys also contain persistent
        tracker ID, event index, independent stream ID, and sample index.
    particle_sampling_fraction : float
        Diagnostic-only Bernoulli sampling probability. Retained particle
        weights are divided by this probability, preserving every linear
        incoherent observable in expectation.
    resolution_warning_thresholds : dict, optional
        Nonnegative warning thresholds for ``delta_eta``, ``delta_theta_u``,
        and ``chi_turn``. Indicators are reported but never alter events.
    max_allocation_bytes : int or None
        Maximum aggregate peak diagnostic footprint across all configured
        species, including persistent products, bounded workspaces, lookup
        tables, writer/reduction copies, identity state, and pusher coupling.
    restart_policy : {'require_segment', 'new_segment'}
        Require a compatible committed predecessor on restart, or explicitly
        begin a discontinuous new radiation lineage from a legacy checkpoint.
    """

    def __init__(
            self, period=None, dt_period=None, species=None, comm=None,
            write_dir=None, iteration_min=0, iteration_max=np.inf,
            observer_frame="laboratory", boost=None,
            observer_translation=None, photon_energy_edges=None,
            theta_x_edges=None, theta_y_edges=None,
            angular_measure="solid_angle", detectors=None,
            observer_time_edges=None, source_coordinate_edges=None,
            source_projections=None, source_moments=None,
            source_z_intervals=None, source_z_interval_edges=None,
            channels=None, output_mode="cumulative", samples_per_particle=1,
            particle_batch_size=262144, particle_selection=None,
            gamma_cutoff=None, energy_band_mode="joint", random_seed=0,
            particle_sampling_fraction=1.0,
            resolution_warning_thresholds=None,
            max_allocation_bytes=1073741824,
            restart_policy="require_segment"):
        if not species:
            raise ValueError(
                "`SynchrotronRadiationDiagnostic` requires at least one "
                "species.")
        if output_mode not in ("cumulative", "interval", "both"):
            raise ValueError(
                "`output_mode` must be 'cumulative', 'interval', or 'both'.")
        if observer_frame not in ("laboratory", "lab", "simulation"):
            raise ValueError(
                "`observer_frame` must be 'laboratory' or 'simulation'.")
        if restart_policy not in ("require_segment", "new_segment"):
            raise ValueError(
                "`restart_policy` must be 'require_segment' or "
                "'new_segment'.")
        self.restart_policy = restart_policy
        if max_allocation_bytes is not None:
            try:
                max_allocation_bytes = int(max_allocation_bytes)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "max_allocation_bytes must be positive or None.")
            if max_allocation_bytes < 1:
                raise ValueError(
                    "max_allocation_bytes must be positive or None.")
        if observer_frame == "simulation" and boost is not None:
            raise ValueError(
                "Do not pass `boost` when `observer_frame='simulation'`.")

        self.species = species
        self.species_names = list(species)
        radiators = {}
        for species_name, particle_species in species.items():
            radiator = particle_species.synchrotron_radiator
            if radiator is None:
                raise ValueError(
                    "Species `%s` must activate synchrotron radiation first."
                    % species_name)
            radiators[species_name] = radiator

        first = radiators[self.species_names[0]]
        self.use_cuda = first.use_cuda
        self.dt_sim = first.dt
        for species_name, radiator in radiators.items():
            if radiator.use_cuda != self.use_cuda:
                raise ValueError(
                    "All species in one radiation diagnostic must use the "
                    "same CPU/GPU backend.")
            if not np.isclose(
                    radiator.dt, self.dt_sim, rtol=2.0e-14, atol=0.0):
                raise ValueError(
                    "All species in one radiation diagnostic must have the "
                    "same timestep.")
            if radiator.observer_accumulator is not None:
                raise RuntimeError(
                    "Species `%s` is already attached to a synchrotron "
                    "diagnostic." % species_name)

        # Stateless streams require immutable integer identity. Record which
        # species need automatic tracking now, but defer the allocation until
        # every accumulator has passed its complete memory preflight.
        identity_comm = comm_world if comm is None else comm
        identity_missing = {}
        identity_sorting_buffer_missing = {}
        for species_name, particle_species in self.species.items():
            tracker = getattr(particle_species, "tracker", None)
            identity_missing[species_name] = tracker is None
            identity_sorting_buffer_missing[species_name] = bool(
                tracker is None and particle_species.use_cuda
                and hasattr(particle_species, "track")
                and not hasattr(particle_species, "int_sorting_buffer"))
            if tracker is not None:
                if (not hasattr(tracker, "id")
                        or int(tracker.id.size) != int(
                            particle_species.Ntot)):
                    raise ValueError(
                        "Persistent particle-ID and particle arrays have "
                        "unequal sizes for species `%s`." % species_name)

        if observer_frame != "simulation" and boost is None:
            expected_transform = (first.gamma_boost, first.beta_boost)
            for species_name, radiator in radiators.items():
                transform = (radiator.gamma_boost, radiator.beta_boost)
                if not np.allclose(
                        transform, expected_transform, rtol=2.0e-14,
                        atol=2.0e-15):
                    raise ValueError(
                        "Activated species use different observer boosts; "
                        "pass one explicit `boost`.")

        if isinstance(channels, dict):
            channels = [name for name, enabled in channels.items() if enabled]
        boost_gamma = None if boost is None else boost.gamma0
        boost_beta = None if boost is None else boost.beta0
        self.accumulators = {}
        remaining_allocation = (
            None if max_allocation_bytes is None
            else int(max_allocation_bytes))
        mpi_size = 1 if comm is None else comm.size
        automatically_tracked = []
        try:
            for species_name in self.species_names:
                if (remaining_allocation is not None
                        and remaining_allocation < 1):
                    raise MemoryError(
                        "The aggregate radiation diagnostic memory budget "
                        "was exhausted before configuring species `%s`."
                        % species_name)
                radiator = radiators[species_name]
                configuration = {
                    "observer_frame": observer_frame,
                    "observer_translation": observer_translation,
                    "enabled_channels": channels,
                    "photon_energy_edges": photon_energy_edges,
                    "theta_x_edges": theta_x_edges,
                    "theta_y_edges": theta_y_edges,
                    "angular_measure": angular_measure,
                    "detectors": detectors,
                    "observer_time_edges": observer_time_edges,
                    "source_coordinate_edges": source_coordinate_edges,
                    "source_projections": source_projections,
                    "source_moments": source_moments,
                    "source_z_intervals": source_z_intervals,
                    "source_z_interval_edges": source_z_interval_edges,
                    "samples_per_particle": samples_per_particle,
                    "particle_batch_size": particle_batch_size,
                    "particle_selection": particle_selection,
                    "energy_band_mode": energy_band_mode,
                    "random_seed": random_seed,
                    "random_namespace":
                        _stable_random_namespace(species_name),
                    "particle_sampling_fraction":
                        particle_sampling_fraction,
                    "resolution_warning_thresholds":
                        resolution_warning_thresholds,
                    "output_mode": output_mode,
                    "mpi_size": mpi_size,
                    "max_allocation_bytes": max_allocation_bytes,
                    "allocation_budget_bytes": remaining_allocation,
                    "identity_tracking_will_be_activated":
                        identity_missing[species_name],
                    "identity_sorting_buffer_will_be_allocated":
                        identity_sorting_buffer_missing[species_name],
                }
                if gamma_cutoff is not None:
                    configuration["gamma_cutoff"] = gamma_cutoff
                if boost_gamma is not None:
                    configuration["gamma_boost"] = boost_gamma
                    configuration["beta_boost"] = boost_beta
                accumulator = radiator.configure_observer_diagnostic(
                    **configuration)
                self.accumulators[species_name] = accumulator
                if remaining_allocation is not None:
                    remaining_allocation -= (
                        accumulator.estimated_total_allocation_bytes)

            # Only now allocate IDs: the complete aggregate footprint has
            # passed the configured limit for every species. The standard
            # tracker carries IDs through sorting, migration, injection, and
            # ionization-generated particle reallocation.
            for species_name, particle_species in self.species.items():
                if not identity_missing[species_name]:
                    continue
                used_track_method = hasattr(particle_species, "track")
                created_sorting_buffer = (
                    identity_sorting_buffer_missing[species_name])
                if used_track_method:
                    particle_species.track(identity_comm)
                else:
                    particle_species.tracker = ParticleTracker(
                        identity_comm.size, identity_comm.rank,
                        particle_species.Ntot)
                automatically_tracked.append((
                    particle_species, used_track_method,
                    created_sorting_buffer))
                if (particle_species.use_cuda
                        and not isinstance(particle_species.ux, np.ndarray)):
                    particle_species.tracker.send_to_gpu()
                if int(particle_species.tracker.id.size) != int(
                        particle_species.Ntot):
                    raise RuntimeError(
                        "Automatic persistent particle-ID allocation failed "
                        "for species `%s`." % species_name)
        except Exception:
            for radiator in radiators.values():
                radiator.observer_accumulator = None
            self.accumulators.clear()
            for (particle_species, used_track_method,
                    created_sorting_buffer) in automatically_tracked:
                particle_species.tracker = None
                if (used_track_method
                        and hasattr(particle_species,
                                    "n_integer_quantities")):
                    particle_species.n_integer_quantities -= 1
                if (created_sorting_buffer
                        and hasattr(particle_species,
                                    "int_sorting_buffer")):
                    del particle_species.int_sorting_buffer
            raise
        self.memory_estimate = {
            name: {
                "total_bytes":
                    accumulator.estimated_total_allocation_bytes,
                "components": dict(accumulator.allocation_breakdown),
            }
            for name, accumulator in self.accumulators.items()}
        self.estimated_total_allocation_bytes = sum(
            item["total_bytes"] for item in self.memory_estimate.values())

        OpenPMDDiagnostic.__init__(
            self, period, comm, write_dir, iteration_min, iteration_max,
            dt_period=dt_period, dt_sim=self.dt_sim,
        )
        self.observer_writer = ObserverRadiationWriter(self, output_mode)
        self.diagnostic_id = "observer_radiation:%s" % ",".join(
            sorted(self.species_names))
        (self.segment_configuration,
         self.configuration_fingerprint) = diagnostic_configuration(self)
        self.code_revision = code_revision()
        self.segment_dir = os.path.join(self.write_dir, "segments")
        self.segment_manifest_dir = os.path.join(
            self.segment_dir, "manifests")
        self.create_dir("segments")
        self.create_dir(os.path.join("segments", "manifests"))

        # Segment identity is supplied by Simulation on first use. Keeping
        # these fields small and separate from the accumulator is what makes
        # checkpoint metadata independent of radiation array layout.
        self._segment_initialized = False
        self._segment_finalized = False
        self._segment_state = "uninitialized"
        self._segment_id = None
        self._segment_run_id = None
        self._segment_event_begin = None
        self._segment_parent_checkpoint_id = None
        self._segment_parent_checkpoint_iteration = None
        self._segment_continuity = None
        self._closed_segment_reference = None
        self._last_segment_reference = None
        # Leave established diagnostics at their pre-push phase; the PIC loop
        # schedules this diagnostic only after a completed pusher impulse.
        self.write_after_momentum_push = True

    def write_hdf5(self, iteration):
        """Write a scheduled snapshot only when a new event is available."""
        if self._segment_finalized:
            raise RuntimeError(
                "Observer radiation was finalized; start a new run before "
                "writing more scheduled snapshots.")
        if not self._segment_initialized:
            self._initialize_standalone_segment()
        return self.observer_writer.write(iteration, final_flush=False)

    def _collective_segment_id(self):
        """Return one UUID shared by all ranks participating in the write."""
        value = uuid.uuid4().hex if self.rank == 0 else None
        size = 1 if self.comm is None else int(self.comm.size)
        if size > 1:
            value = comm_world.bcast(value, root=0)
        return value

    def _find_checkpoint_segment(self, context):
        matches = [
            item for item in context.get("segments", [])
            if item.get("diagnosticId") == self.diagnostic_id]
        if len(matches) > 1:
            raise RuntimeError(
                "The checkpoint contains duplicate radiation segment "
                "references for `%s`." % self.diagnostic_id)
        return matches[0] if matches else None

    def _receive_accumulators(self):
        initially_on_gpu = {}
        for species_name, accumulator in self.accumulators.items():
            initially_on_gpu[species_name] = bool(accumulator._on_gpu)
            if initially_on_gpu[species_name]:
                self.species[species_name].synchrotron_radiator.receive_from_gpu()
        return initially_on_gpu

    def _restore_accumulators(self, initially_on_gpu):
        for species_name, was_on_gpu in initially_on_gpu.items():
            if was_on_gpu:
                self.species[species_name].synchrotron_radiator.send_to_gpu()

    def _reset_segment_accumulators(self):
        initially_on_gpu = self._receive_accumulators()
        try:
            for accumulator in self.accumulators.values():
                accumulator.reset_segment_state()
            self.observer_writer.previous = {}
            self.observer_writer.pending_previous = {}
            self.observer_writer.active_timing = {}
            self.observer_writer.last_written_events = {
                name: int(self.accumulators[name].completed_event_count)
                for name in self.species_names}
        finally:
            self._restore_accumulators(initially_on_gpu)

    def _open_segment(self, run_id, event_begin, parent_id,
                      parent_iteration, continuity):
        self._segment_run_id = str(run_id)
        self._segment_id = self._collective_segment_id()
        self._segment_event_begin = int(event_begin)
        for accumulator in self.accumulators.values():
            if (accumulator.last_completed_event_index is None
                    and self._segment_event_begin > 0):
                accumulator.last_completed_event_index = (
                    self._segment_event_begin - 1)
        self._segment_parent_checkpoint_id = parent_id
        self._segment_parent_checkpoint_iteration = (
            None if parent_iteration is None else int(parent_iteration))
        self._segment_continuity = continuity
        self._segment_initialized = True
        self._segment_state = "open"
        self._closed_segment_reference = None

    def _restart_problems(self, context, reference):
        problems = []
        if reference is None:
            problems.append("no predecessor segment is recorded")
        else:
            if reference.get("configurationFingerprint") != (
                    self.configuration_fingerprint):
                problems.append("the diagnostic fingerprint changed")
            if reference.get("closingCheckpointId") != (
                    context.get("checkpoint_id")):
                problems.append("the segment closes at a different checkpoint")
            if int(reference.get("eventEndExclusive", -1)) != int(
                    context.get("iteration", -2)):
                problems.append("the event boundary differs from the checkpoint")
            if radiation_segment_status(reference.get("path", "")) != (
                    "committed"):
                problems.append("the predecessor segment is not committed")
        missing_ids = [
            name for name, particle_species in self.species.items()
            if not bool(getattr(
                particle_species, "_persistent_ids_restored", False))]
        if missing_ids:
            problems.append(
                "persistent particle IDs were not restored for %s"
                % ", ".join(sorted(missing_ids)))
        return problems

    def start_segment(self, context):
        """Start a zeroed segment at genesis, restart, or a committed boundary."""
        context = dict(context)
        if self._segment_finalized:
            raise RuntimeError(
                "This radiation diagnostic was finalized and cannot be resumed.")

        if context.get("initial", False):
            if self._segment_initialized:
                return False
            event_begin = int(context.get("iteration", 0))
            run_id = context.get("run_id") or self._collective_segment_id()
            continuity = "genesis"
            if context.get("restart", False):
                reference = self._find_checkpoint_segment(context)
                problems = self._restart_problems(context, reference)
                if problems and self.restart_policy != "new_segment":
                    detail = "; ".join(problems)
                    raise RuntimeError(
                        "Cannot continue observer radiation seamlessly: %s. "
                        "For a legacy or intentionally discontinuous restart, "
                        "construct the diagnostic with "
                        "restart_policy='new_segment'." % detail)
                if problems:
                    run_id = self._collective_segment_id()
                    continuity = (
                        "discontinuous_legacy_restart"
                        if context.get("legacy", False) else
                        "discontinuous_restart")
                else:
                    run_id = reference["runId"]
                    continuity = "seamless"
            if any(
                    accumulator.completed_event_count != 0
                    for accumulator in self.accumulators.values()):
                raise RuntimeError(
                    "A Simulation-managed radiation segment must start from "
                    "a zeroed accumulator.")
            self._open_segment(
                run_id, event_begin, context.get("checkpoint_id"),
                context.get("checkpoint_iteration"), continuity)
            return True

        if not self._segment_initialized or self._segment_state != "closed":
            raise RuntimeError(
                "A new radiation segment can start only after its predecessor "
                "has closed.")
        reference = self._find_checkpoint_segment(context)
        if reference is None or reference.get("segmentId") != self._segment_id:
            raise RuntimeError(
                "The committed checkpoint does not reference the segment that "
                "was just closed.")
        if radiation_segment_status(reference["path"]) != "committed":
            self._segment_state = "orphaned"
            raise RuntimeError(
                "The closed radiation segment was not committed by its "
                "checkpoint manifest.")
        if reference.get("configurationFingerprint") != (
                self.configuration_fingerprint):
            raise RuntimeError(
                "The committed segment fingerprint changed in memory.")
        if int(context["iteration"]) != int(
                reference["eventEndExclusive"]):
            raise RuntimeError(
                "The next radiation segment does not begin at the committed "
                "predecessor boundary.")
        self._last_segment_reference = dict(reference)
        self._reset_segment_accumulators()
        self._open_segment(
            reference["runId"], int(context["iteration"]),
            context.get("checkpoint_id"),
            context.get("checkpoint_iteration"), "seamless")
        return True

    def prepare_step(self, iteration):
        """Ensure stepping always targets a writable, unambiguous segment."""
        if self._segment_finalized:
            # Historically a diagnostic could be finalized and then reused.
            # Preserve that convenience without reopening a committed run:
            # resumed events form a new independently mergeable run.
            self._reset_segment_accumulators()
            self._segment_finalized = False
            self._open_segment(
                self._collective_segment_id(), int(iteration),
                None, None, "genesis_after_finalize")
        if self._segment_state in ("closed", "orphaned"):
            raise RuntimeError(
                "Observer radiation has no writable open segment.")
        if (self._segment_initialized
                and int(iteration) < self._segment_event_begin):
            raise RuntimeError(
                "Simulation iteration precedes the open radiation segment.")

    def _validate_segment_range(self, event_end_exclusive):
        begin = int(self._segment_event_begin)
        end = int(event_end_exclusive)
        if end < begin:
            raise RuntimeError("Radiation segment event range is reversed.")
        expected = end - begin
        for species_name, accumulator in self.accumulators.items():
            timing = accumulator.cumulative_timing
            count = int(timing["event_count"])
            if count != expected:
                raise RuntimeError(
                    "Radiation segment [%d, %d) for `%s` contains %d of %d "
                    "required pusher events. Refusing to commit a gap."
                    % (begin, end, species_name, count, expected))
            if expected == 0:
                continue
            first = int(timing["first_event_index"])
            timed_last = int(timing["last_event_index"])
            last = accumulator.last_completed_event_index
            if (first != begin or timed_last != end - 1
                    or last is None or int(last) != timed_last):
                raise RuntimeError(
                    "Radiation timing for `%s` does not exactly represent "
                    "the half-open event range [%d, %d)."
                    % (species_name, begin, end))
        return begin, end

    def _reduced_segment_states(self):
        initially_on_gpu = self._receive_accumulators()
        try:
            return {
                species_name: self.observer_writer._reduce_snapshot(
                    accumulator.snapshot())
                for species_name, accumulator in self.accumulators.items()}
        finally:
            self._restore_accumulators(initially_on_gpu)

    @staticmethod
    def _timing_metadata(timing):
        count = int(timing["event_count"])
        if count == 0:
            return {
                "completedEventCount": 0,
                "firstCompletedEventIndex": None,
                "lastCompletedEventIndex": None,
                "firstEventCenterSimulation": None,
                "lastEventCenterSimulation": None,
                "representedIntervalStartSimulation": None,
                "representedIntervalEndSimulation": None,
            }
        return {
            "completedEventCount": count,
            "firstCompletedEventIndex": int(timing["first_event_index"]),
            "lastCompletedEventIndex": int(timing["last_event_index"]),
            "firstEventCenterSimulation": float(
                timing["first_event_center"]),
            "lastEventCenterSimulation": float(timing["last_event_center"]),
            "representedIntervalStartSimulation": float(
                timing["represented_interval_start"]),
            "representedIntervalEndSimulation": float(
                timing["represented_interval_end"]),
        }

    def _segment_metadata(self, context, begin, end, states):
        frame_times = {}
        actual_timing = {}
        random_namespaces = {}
        has_events = end > begin
        for species_name in sorted(self.species_names):
            accumulator = self.accumulators[species_name]
            translation_time = accumulator.observer_translation[0] / c
            timing = states[species_name]["timing"]
            if has_events:
                first_center = float(timing["first_event_center"])
                last_center = float(timing["last_event_center"])
                interval_start = float(timing["represented_interval_start"])
                interval_end = float(timing["represented_interval_end"])
            else:
                first_center = None
                last_center = None
                interval_start = begin * self.dt_sim
                interval_end = interval_start
            frame_times[species_name] = {
                "simulationEventCenterStart": first_center,
                "simulationEventCenterEnd": last_center,
                "simulationImpulseIntervalStart": interval_start,
                "simulationImpulseIntervalEnd": interval_end,
                "observerOriginEventCenterStart": (
                    None if first_center is None else
                    accumulator.gamma_boost * first_center + translation_time),
                "observerOriginEventCenterEnd": (
                    None if last_center is None else
                    accumulator.gamma_boost * last_center + translation_time),
                "observerOriginImpulseIntervalStart": (
                    accumulator.gamma_boost * interval_start
                    + translation_time),
                "observerOriginImpulseIntervalEnd": (
                    accumulator.gamma_boost * interval_end
                    + translation_time),
            }
            actual_timing[species_name] = self._timing_metadata(
                states[species_name]["timing"])
            random_namespaces[species_name] = {
                "seed": int(accumulator.random_seed),
                "speciesNamespace": int(accumulator.random_namespace),
                "streamIdentifiers": dict(accumulator.random_stream_ids),
                "eventKey": (
                    "absolute_simulation_event_index;"
                    "persistent_particle_id;species_namespace;seed;stream;"
                    "sample_index"),
            }
        return {
            "artifactType": "radiation_segment",
            "radiationStateSchemaVersion": self.segment_configuration[
                "radiationStateSchemaVersion"],
            "runId": self._segment_run_id,
            "segmentId": self._segment_id,
            "diagnosticId": self.diagnostic_id,
            "parentCheckpointId": self._segment_parent_checkpoint_id,
            "parentCheckpointIteration":
                self._segment_parent_checkpoint_iteration,
            "closingCheckpointId": context.get("checkpoint_id"),
            "closingCheckpointIteration": context.get(
                "checkpoint_iteration"),
            "eventConvention": "half_open_[eventBegin,eventEndExclusive)",
            "eventBegin": begin,
            "eventEndExclusive": end,
            "firstRadiationEventIndex": begin if has_events else None,
            "lastRadiationEventIndex": end - 1 if has_events else None,
            "frameTimes": frame_times,
            "completedEventTiming": actual_timing,
            "segmentStatus": "closed",
            "closeReason": context["close_reason"],
            "accumulationScope": "segment",
            "continuity": self._segment_continuity,
            "configurationFingerprint": self.configuration_fingerprint,
            "randomness": random_namespaces,
            "persistentParticleIdentity": "fbpic_particle_tracker_uint64",
            "commitManifest": context["commit_manifest"],
            "codeRevision": self.code_revision,
        }

    def close_segment(self, context):
        """Persist raw mergeable state for one half-open event interval."""
        context = dict(context)
        if not self._segment_initialized:
            raise RuntimeError("No radiation segment is open.")
        if self._segment_state == "closed":
            reference = self._closed_segment_reference
            if (reference is not None
                    and reference.get("closingCheckpointId")
                    == context.get("checkpoint_id")):
                return dict(reference)
            raise RuntimeError("Radiation segment is already closed.")
        if self._segment_state != "open":
            raise RuntimeError(
                "Radiation segment cannot close from state `%s`."
                % self._segment_state)

        begin, end = self._validate_segment_range(
            context["event_end_exclusive"])
        states = self._reduced_segment_states()
        path = os.path.abspath(os.path.join(
            self.segment_dir, "segment-%s.h5" % self._segment_id))
        if self.rank == 0:
            metadata = self._segment_metadata(
                context, begin, end, states)
            metadata["commitManifest"] = relative_file_reference(
                context["commit_manifest"], path)
            write_segment(
                path, metadata, self.segment_configuration,
                self.configuration_fingerprint, states)
        reference = {
            "runId": self._segment_run_id,
            "segmentId": self._segment_id,
            "diagnosticId": self.diagnostic_id,
            "path": path,
            "parentCheckpointId": self._segment_parent_checkpoint_id,
            "closingCheckpointId": context.get("checkpoint_id"),
            "parentCheckpointIteration":
                self._segment_parent_checkpoint_iteration,
            "closingCheckpointIteration": context.get(
                "checkpoint_iteration"),
            "eventBegin": begin,
            "eventEndExclusive": end,
            "configurationFingerprint": self.configuration_fingerprint,
            "closeReason": context["close_reason"],
        }
        self._segment_state = "closed"
        self._closed_segment_reference = dict(reference)
        return reference

    def _initialize_standalone_segment(self):
        first_indices = [
            int(accumulator.cumulative_timing["first_event_index"])
            for accumulator in self.accumulators.values()
            if int(accumulator.cumulative_timing["event_count"]) > 0]
        event_begin = min(first_indices) if first_indices else 0
        self._open_segment(
            self._collective_segment_id(), event_begin, None, None, "genesis")

    def finalize_segment(self):
        """Close and commit the final segment without a simulation checkpoint."""
        if self._segment_finalized:
            return False
        if not self._segment_initialized:
            self._initialize_standalone_segment()
        if self._segment_state != "open":
            raise RuntimeError(
                "Only an open radiation segment can be finalized.")
        last_indices = [
            int(accumulator.last_completed_event_index)
            for accumulator in self.accumulators.values()
            if accumulator.last_completed_event_index is not None]
        event_end = (
            max(last_indices) + 1 if last_indices
            else int(self._segment_event_begin))
        manifest_path = os.path.abspath(os.path.join(
            self.segment_manifest_dir,
            "final-%s.json" % self._segment_id))
        context = {
            "checkpoint_id": None,
            "checkpoint_iteration": None,
            "event_end_exclusive": event_end,
            "close_reason": "finalize",
            "commit_manifest": manifest_path,
        }
        reference = self.close_segment(context)
        manifest = {
            "radiationSegmentManifestSchemaVersion": 1,
            "manifestType": "radiation_finalization",
            "segmentStatus": "committed",
            "runId": self._segment_run_id,
            "diagnosticId": self.diagnostic_id,
            "configurationFingerprint": self.configuration_fingerprint,
            "eventEndExclusive": event_end,
            "segments": [
                relative_segment_reference(reference, manifest_path)],
        }
        if self.rank == 0:
            atomic_write_json(manifest_path, manifest)
        if (1 if self.comm is None else int(self.comm.size)) > 1:
            comm_world.barrier()
        if radiation_segment_status(reference["path"]) != "committed":
            self._segment_state = "orphaned"
            raise RuntimeError(
                "Final radiation segment did not acquire a commit manifest.")
        self._segment_state = "committed"
        self._segment_finalized = True
        self._last_segment_reference = dict(reference)
        return True

    @property
    def segment_status(self):
        """Return the derived state of the current persisted segment."""
        if self._segment_state == "uninitialized":
            return "open"
        if self._segment_state == "closed":
            persisted = radiation_segment_status(
                self._closed_segment_reference["path"])
            return "committed" if persisted == "committed" else "closed"
        return self._segment_state

    def get_segment_status(self):
        """Return current and most recently closed segment identities."""
        last_status = None
        if self._last_segment_reference is not None:
            last_status = radiation_segment_status(
                self._last_segment_reference["path"])
        return {
            "current": self.segment_status,
            "runId": self._segment_run_id,
            "segmentId": self._segment_id,
            "eventBegin": self._segment_event_begin,
            "lastClosed": self._last_segment_reference,
            "lastClosedStatus": last_status,
        }

    def finalize(self):
        """Flush the latest snapshot and commit the terminal segment.

        Finalization is terminal for this diagnostic and is idempotent. Normal
        :meth:`Simulation.step` calls never invoke it implicitly.
        """
        if not self._segment_initialized:
            self._initialize_standalone_segment()
        snapshot_written = self.observer_writer.write(
            None, final_flush=True)
        segment_committed = self.finalize_segment()
        return bool(snapshot_written or segment_committed)

    def flush(self, iteration=None):
        """Compatibility alias for :meth:`finalize`.

        ``iteration`` is accepted for source compatibility but deliberately
        ignored; the file iteration and time are always taken from the latest
        event included in the snapshot.
        """
        return self.finalize()

    def get_memory_estimate(self):
        """Return the pre-allocation estimate exposed by this diagnostic."""
        return {
            name: {
                "total_bytes": estimate["total_bytes"],
                "components": dict(estimate["components"]),
            }
            for name, estimate in self.memory_estimate.items()
        }

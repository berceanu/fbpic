# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""Public configuration for observer-frame synchrotron products."""

import numpy as np

from fbpic.particles.tracking import ParticleTracker
from fbpic.utils.mpi import comm as comm_world
from .generic_diag import OpenPMDDiagnostic
from .observer_radiation_diag import ObserverRadiationWriter

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
            max_allocation_bytes=1073741824):
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
        # Leave established diagnostics at their pre-push phase; the PIC loop
        # schedules this diagnostic only after a completed pusher impulse.
        self.write_after_momentum_push = True

    def write_hdf5(self, iteration):
        """Write a scheduled snapshot only when a new event is available."""
        return self.observer_writer.write(iteration, final_flush=False)

    def finalize(self):
        """Explicitly write the latest unwritten completed pusher event.

        This operation is independent of the scheduled cadence and iteration
        window. It is idempotent: it returns ``False`` without writing when
        all completed events are already represented by an output file.
        """
        return self.observer_writer.write(None, final_flush=True)

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

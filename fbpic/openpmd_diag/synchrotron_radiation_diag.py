# Copyright 2023, FBPIC contributors
# Authors: Igor A Andriyash, Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file defines the class SRDiagnostic.
"""
import os
import numpy as np
from scipy.constants import hbar
from .generic_diag import OpenPMDDiagnostic
from .observer_radiation_diag import ObserverRadiationWriter
from fbpic.utils.mpi import comm as comm_simple

class SynchrotronRadiationDiagnostic(OpenPMDDiagnostic):
    """
    Class that defines the synchrotron radiation diagnostics to be performed.
    """

    def __init__(
            self, period=None, dt_period=None, species=None, comm=None,
            write_dir=None, iteration_min=0, iteration_max=np.inf,
            observer_frame=None, boost=None, observer_translation=None,
            photon_energy_bin_edges=None, photon_energy_edges=None,
            angular_measure="solid_angle", angular_bin_edges=None,
            theta_x_bin_edges=None, theta_y_bin_edges=None,
            theta_x_edges=None, theta_y_edges=None,
            detectors=None, detector_directions=None,
            detector_apertures=None, observer_time_bin_edges=None,
            observer_time_edges=None, source_coordinate_bin_edges=None,
            source_coordinate_edges=None,
            source_distribution_projections=None, source_projections=None,
            source_moment_selections=None, source_moments=None,
            output_channels=None, channels=None, cumulative=True,
            interval=False, output_mode=None,
            local_angular_model="synchrotron", samples_per_particle=1,
            particle_batch_size=262144,
            particle_selection=None, gamma_threshold=None,
            gamma_cutoff=None):
        """
        Initialize the synchrotron radiation diagnostic

        Parameters
        ----------
        period : int, optional
            The period of the diagnostics, in number of timesteps.
            (i.e. the diagnostics are written whenever the number
            of iterations is divisible by `period`). Specify either this or
            `dt_period`

        dt_period : float (in seconds), optional
            The period of the diagnostics, in physical time of the simulation.
            Specify either this or `period`

        species: a dictionary of :any:`Particles` objects
            The object that is written (e.g. elec)
            is assigned to the particle name of this species.
            (e.g. {"electrons": elec }). All species must have synchrotron
            radiation activated with the same specral-angular grids.

        comm : an fbpic BoundaryCommunicator object or None
            If this is not None, the data is gathered on the first proc,
            and the guard cells are removed from the output.
            Otherwise, each proc writes its own data, including guard cells
            (Make sure to use different write_dir in this case)

        write_dir : string, optional
            The POSIX path to the directory where the results are
            to be written. If none is provided, this will be the path
            of the current working directory

        iteration_min, iteration_max: ints, optional
            The iterations between which data should be written
            (`iteration_min` is inclusive, `iteration_max` is exclusive)

        observer_frame: {'laboratory', 'simulation'}, optional
            Frame in which every advanced radiation observable is defined.
            Passing any observer-frame product option selects the advanced
            interface; omitting all of them preserves the legacy writer.

        boost: a BoostConverter, optional
            Lorentz transform from the simulation frame to the laboratory
            observer frame.  If omitted, the transform supplied when the
            species was activated is used.

        observer_translation: array-like of four floats, optional
            Translation ``(ct, x, y, z)`` applied after the Lorentz transform.

        photon_energy_bin_edges: array-like, optional
            Observer-frame photon-energy bin edges in joules.

        angular_bin_edges: pair or dict, optional
            Observer-frame ``theta_x`` and ``theta_y`` bin edges.  A dict uses
            the keys ``theta_x`` and ``theta_y``.

        angular_measure: {'solid_angle', 'projected_angles'}, optional
            Whether angular-spectral densities are per ``dOmega`` or per
            ``dtheta_x dtheta_y``.

        detectors: sequence of dicts, optional
            Far-field channels.  Each dict contains a three-vector
            ``direction`` (or ``theta_x`` and ``theta_y``), optional circular
            ``half_angle``, ``aperture_quadrature``, ``energy_bands``, and
            per-detector time edges or pulse interval.

        observer_time_bin_edges: array-like, optional
            Default edges for ``t - n.r/c`` profiles, in seconds.

        source_coordinate_bin_edges: dict, optional
            Observer-frame edges for any of ``x``, ``y``, and ``z``.

        source_distribution_projections: sequence, optional
            Named dicts or axis tuples selected from ``x``, ``y``, ``z``,
            ``theta_x``, ``theta_y``, ``energy``, and ``time``.  ``'full'``
            selects all axes whose edges are available.

        source_moment_selections: sequence of dicts, optional
            Energy, angle, coordinate, direction, and observer-time regions
            for radiation-weighted source moments.

        output_channels: sequence or dict, optional
            Independently enable ``angular_spectral``, ``observer_time``,
            ``source``, ``source_moments``, and ``accounting``.  When omitted,
            channels are inferred from the supplied configurations.

        cumulative, interval: bool, optional
            Select cumulative and/or per-output-interval records.  The
            equivalent ``output_mode`` values are ``cumulative``, ``interval``,
            and ``both``.

        local_angular_model: {'synchrotron', 'legacy_gaussian'}, optional
            Local spectral-angular closure.  The default uses the local orbit
            plane and a photon-energy-dependent Schwinger angular kernel.

        samples_per_particle: int, optional
            Fixed number of stratified local spectral-angular samples per
            emitting particle and step.  It does not depend on output bins.

        particle_batch_size: int, optional
            Maximum particles processed together.  This bounds temporary
            CPU/GPU memory independently of the total population.

        particle_selection: dict, optional
            Observer-frame ranges for particle position, momentum, angle,
            weight, or Lorentz factor.

        gamma_threshold, gamma_cutoff: float, optional
            Observer-frame Lorentz-factor threshold.  ``gamma_threshold`` is
            the preferred diagnostic spelling; ``gamma_cutoff`` is an alias.
        """
        if species is None:
            species = {}
        # Check input
        if len(species) == 0:
            raise ValueError(
            "`SRDiagnostic` requires the dictionary with the species.")

        # Register the arguments
        self.species = species
        self.species_names = list( species.keys() )

        for species_name in self.species_names:
            if species[ species_name ].synchrotron_radiator is None:
                raise ValueError(
                    f"{species_name} must have synchrotron radiation active")

        radiators = {
            name: species[name].synchrotron_radiator
            for name in self.species_names
        }
        sr_object = radiators[self.species_names[0]]

        self.use_cuda = sr_object.use_cuda
        self.dt_sim = sr_object.dt
        for species_name, radiator in radiators.items():
            if radiator.use_cuda != self.use_cuda:
                raise ValueError(
                    "All synchrotron species in one diagnostic must use the "
                    "same CPU/GPU backend.")
            if not np.isclose(
                    radiator.dt, self.dt_sim, rtol=2.e-14, atol=0.0):
                raise ValueError(
                    "All synchrotron species in one diagnostic must have the "
                    "same timestep.")
        self.advanced = any(value is not None for value in (
            observer_frame, boost, observer_translation,
            photon_energy_bin_edges, photon_energy_edges,
            angular_bin_edges, theta_x_bin_edges, theta_y_bin_edges,
            theta_x_edges, theta_y_edges, detectors, detector_directions,
            detector_apertures, observer_time_bin_edges, observer_time_edges,
            source_coordinate_bin_edges, source_coordinate_edges,
            source_distribution_projections, source_projections,
            source_moment_selections, source_moments,
            output_channels, channels, gamma_threshold, gamma_cutoff,
            particle_selection,
        )) or output_mode is not None or interval or not cumulative \
            or particle_batch_size != 262144 \
            or angular_measure != "solid_angle" \
            or local_angular_model != "synchrotron" \
            or samples_per_particle != 1 \
            or any(not getattr(radiator, "legacy_enabled", True)
                   for radiator in radiators.values())

        if self.advanced:
            aliases = (
                ("photon_energy_bin_edges", photon_energy_bin_edges,
                 "photon_energy_edges", photon_energy_edges),
                ("theta_x_bin_edges", theta_x_bin_edges,
                 "theta_x_edges", theta_x_edges),
                ("theta_y_bin_edges", theta_y_bin_edges,
                 "theta_y_edges", theta_y_edges),
                ("observer_time_bin_edges", observer_time_bin_edges,
                 "observer_time_edges", observer_time_edges),
                ("source_coordinate_bin_edges", source_coordinate_bin_edges,
                 "source_coordinate_edges", source_coordinate_edges),
                ("source_distribution_projections",
                 source_distribution_projections,
                 "source_projections", source_projections),
                ("source_moment_selections", source_moment_selections,
                 "source_moments", source_moments),
                ("output_channels", output_channels, "channels", channels),
                ("gamma_threshold", gamma_threshold,
                 "gamma_cutoff", gamma_cutoff),
            )
            for first_name, first_value, second_name, second_value in aliases:
                if first_value is not None and second_value is not None:
                    raise ValueError(
                        "Specify only one of `%s` and `%s`." %
                        (first_name, second_name))
            energy_edges = (
                photon_energy_bin_edges if photon_energy_bin_edges is not None
                else photon_energy_edges)
            x_edges = (theta_x_bin_edges if theta_x_bin_edges is not None
                       else theta_x_edges)
            y_edges = (theta_y_bin_edges if theta_y_bin_edges is not None
                       else theta_y_edges)
            if angular_bin_edges is not None:
                if x_edges is not None or y_edges is not None:
                    raise ValueError(
                        "Use either `angular_bin_edges` or separate theta "
                        "edge arguments, not both.")
                if isinstance(angular_bin_edges, dict):
                    x_edges = angular_bin_edges.get(
                        "theta_x", angular_bin_edges.get("x"))
                    y_edges = angular_bin_edges.get(
                        "theta_y", angular_bin_edges.get("y"))
                else:
                    x_edges, y_edges = angular_bin_edges

            time_edges = (
                observer_time_bin_edges
                if observer_time_bin_edges is not None
                else observer_time_edges)
            coordinate_edges = (
                source_coordinate_bin_edges
                if source_coordinate_bin_edges is not None
                else source_coordinate_edges)
            projections = (
                source_distribution_projections
                if source_distribution_projections is not None
                else source_projections)
            moments = (
                source_moment_selections
                if source_moment_selections is not None else source_moments)
            requested_channels = (
                output_channels if output_channels is not None else channels)
            if isinstance(requested_channels, dict):
                requested_channels = [
                    name for name, enabled in requested_channels.items()
                    if enabled]

            detector_config = self._combine_detector_configuration(
                detectors, detector_directions, detector_apertures)
            if observer_frame is None:
                observer_frame = "laboratory"
            if output_mode is None:
                if cumulative and interval:
                    output_mode = "both"
                elif interval:
                    output_mode = "interval"
                elif cumulative:
                    output_mode = "cumulative"
                else:
                    raise ValueError(
                        "At least one of cumulative or interval output is "
                        "required.")
            if output_mode not in ("cumulative", "interval", "both"):
                raise ValueError(
                    "`output_mode` must be cumulative, interval, or both.")

            if observer_frame != "simulation" and boost is None:
                first_transform = (
                    sr_object.gamma_boost, sr_object.beta_boost)
                for species_name, radiator in radiators.items():
                    transform = (radiator.gamma_boost, radiator.beta_boost)
                    if not np.allclose(
                            transform, first_transform, rtol=2.e-14,
                            atol=2.e-15):
                        raise ValueError(
                            "All species must use the same inferred observer "
                            "boost; pass an explicit `boost` otherwise.")
            for species_name, radiator in radiators.items():
                if radiator.radiation_reaction:
                    raise NotImplementedError(
                        "Observer-frame radiation products are passive and "
                        "cannot be combined with radiation reaction.")
                if radiator.observer_accumulator is not None:
                    raise RuntimeError(
                        "Only one observer-frame synchrotron diagnostic may "
                        "configure species `%s`." % species_name)

            explicit_gamma = (
                gamma_threshold if gamma_threshold is not None else gamma_cutoff)
            boost_gamma = None if boost is None else boost.gamma0
            boost_beta = None if boost is None else boost.beta0
            self.accumulators = {}
            for species_name in self.species_names:
                radiator = radiators[species_name]
                configuration = {
                    "observer_frame": observer_frame,
                    "observer_translation": observer_translation,
                    "enabled_channels": requested_channels,
                    "photon_energy_edges": energy_edges,
                    "theta_x_edges": x_edges,
                    "theta_y_edges": y_edges,
                    "angular_measure": angular_measure,
                    "detectors": detector_config,
                    "observer_time_edges": time_edges,
                    "source_coordinate_edges": coordinate_edges,
                    "source_projections": projections,
                    "source_moments": moments,
                    "local_angular_model": local_angular_model,
                    "samples_per_particle": samples_per_particle,
                    "particle_batch_size": particle_batch_size,
                    "particle_selection": particle_selection,
                    "gamma_cutoff": (
                        explicit_gamma if explicit_gamma is not None
                        else 1.0 / radiator.gamma_cutoff_inv),
                }
                if boost_gamma is not None:
                    configuration["gamma_boost"] = boost_gamma
                    configuration["beta_boost"] = boost_beta
                self.accumulators[species_name] = \
                    radiator.configure_observer_diagnostic(**configuration)

            OpenPMDDiagnostic.__init__(
                self, period, comm, write_dir, iteration_min, iteration_max,
                dt_period=dt_period, dt_sim=self.dt_sim)
            self.observer_writer = ObserverRadiationWriter(self, output_mode)
            return

        if not getattr(sr_object, "legacy_enabled", True):
            raise ValueError(
                "This species was activated without legacy axes. Configure "
                "observer-frame output channels and bin edges on the "
                "SynchrotronRadiationDiagnostic.")
        self.mesh_shape = (
            sr_object.N_theta_x, sr_object.N_theta_y, sr_object.N_omega
        )

        self.mesh_spacing = np.array([
            sr_object.d_theta_x, sr_object.d_theta_y,
            sr_object.d_omega * hbar ]
        )

        self.mesh_origin = np.array([
            sr_object.theta_x_min, sr_object.theta_y_min,
            sr_object.omega_min * hbar
        ])

        # General setup
        OpenPMDDiagnostic.__init__(self, period, comm, write_dir,
                            iteration_min, iteration_max,
                            dt_period=dt_period, dt_sim=self.dt_sim )

    @staticmethod
    def _combine_detector_configuration(
            detectors, detector_directions, detector_apertures):
        if detectors is None:
            combined = []
        elif isinstance(detectors, dict):
            combined = [dict(detectors)]
        else:
            combined = list(detectors)
        if detector_directions is not None:
            directions = detector_directions
            array = np.asarray(directions)
            if array.shape == (3,):
                directions = [directions]
            for index, direction in enumerate(directions):
                if isinstance(direction, dict):
                    item = dict(direction)
                else:
                    item = {
                        "name": "direction_%d" % index,
                        "direction": direction,
                    }
                combined.append(item)
        if detector_apertures is not None:
            apertures = detector_apertures
            if isinstance(apertures, dict):
                apertures = [apertures]
            elif np.isscalar(apertures):
                if not combined:
                    raise ValueError(
                        "A scalar detector aperture requires at least one "
                        "detector direction.")
                apertures = [apertures] * len(combined)
            if all(isinstance(item, dict) for item in apertures):
                for item in apertures:
                    combined.append(dict(item))
            else:
                if len(apertures) != len(combined):
                    raise ValueError(
                        "Scalar detector aperture values must match the "
                        "number of configured directions.")
                updated = []
                for detector, aperture in zip(combined, apertures):
                    item = (dict(detector) if isinstance(detector, dict)
                            else {"direction": detector})
                    item["half_angle"] = aperture
                    updated.append(item)
                combined = updated
        return combined


    def write_hdf5( self, iteration ):
        """
        Write an HDF5 file that complies with the OpenPMD standard

        Parameter
        ---------
        iteration : int
             The current iteration number of the simulation.
        """

        if self.advanced:
            self.observer_writer.write(iteration)
            return

        # If needed: Receive data from the GPU
        if self.use_cuda :
            for specie_name in self.species_names:
                self.species[specie_name].synchrotron_radiator\
                    .receive_from_gpu()

        # Extract information needed for the openPMD attributes
        time = iteration * self.dt_sim

        # Create the file with these attributes
        filename = "data%08d.h5" %iteration
        fullpath = os.path.join( self.write_dir, "hdf5", filename )
        self.create_file_empty_meshes(
            fullpath, iteration, time )

        # Open the file again, and get the field path
        f = self.open_file( fullpath )
        # (f is None if this processor does not participate in writing data)
        if f is not None:
            field_path = "/data/%d/fields/" %iteration
            field_grp = f[field_path]
        else:
            field_grp = None

        self.write_dataset( field_grp, "radiation" )

        # Close the file (only the first proc does this)
        if f is not None:
            f.close()

        # Send data to the GPU if needed
        if self.use_cuda :
            for specie_name in self.species_names:
                self.species[specie_name].synchrotron_radiator.send_to_gpu()

    # Writing methods
    # ---------------
    def write_dataset( self, field_grp, path) :
        """
        Write a given dataset

        Parameters
        ----------
        field_grp : an h5py.Group object
            The group that corresponds to the path indicated in meshesPath

        path : string
            The relative path where to write the dataset, in field_grp
        """

        for specie_name in self.species_names:
            path_specie = path + '_' + specie_name
            data_array = self.get_dataset( self.species[specie_name] )
            if field_grp is not None:
                dset = field_grp[path_specie]
                dset[:] =  data_array
            else:
                dset = None

    def get_dataset( self, specie ):
        """
        Copy and gather radation data on the first proc, in MPI mode
        """
        # Get the data on each individual proc
        data_one_proc = specie.synchrotron_radiator.radiation_data.copy()

        # Gather the data
        if self.comm.size>1:
            data_all_proc = self.mpi_reduce_radiation( data_one_proc )
        else:
            data_all_proc = data_one_proc

        return( data_all_proc )

    def mpi_reduce_radiation(self, data):
        """
        MPI operation to gather the radiation data
        """
        sendbuf = data
        if self.rank == 0:
            recvbuf = np.empty_like(data)
        else:
            recvbuf = None

        comm_simple.Reduce(sendbuf, recvbuf, root=0)
        return recvbuf

    # OpenPMD setup methods
    # ---------------------

    def create_file_empty_meshes( self, fullpath, iteration, time ):
        """
        Create an openPMD file with empty meshes and setup all its attributes

        Parameters
        ----------
        fullpath: string
            The absolute path to the file to be created

        iteration: int
            The iteration number of this diagnostic

        time: float (seconds)
            The physical time at this iteration
        """
        # Create the file
        f = self.open_file( fullpath )

        # Setup the different layers of the openPMD file
        # (f is None if this processor does not participate is writing data)
        if f is not None:

            # Setup the attributes of the top level of the file
            self.setup_openpmd_file( f, iteration, time, self.dt_sim )

            # Setup the meshes group (contains all the fields)
            field_path = "/data/%d/fields/" %iteration
            field_grp = f.require_group(field_path)

            for specie_name in self.species_names:
                dset = field_grp.require_dataset(
                    f"radiation_{specie_name}", self.mesh_shape, dtype='f8')
                # Setup the record and the component to which it belongs
                self.setup_openpmd_mesh_component_record( dset, "radiation" )
            # Close the file
            f.close()

    def setup_openpmd_mesh_component_record( self, dset, quantity ) :
        """
        Sets the attributes that are specific to a mesh record

        Parameter
        ---------
        dset : an h5py.Dataset or h5py.Group object

        quantity : string
           The name of the record (e.g. "radiation")
        """
        # Generic record attributes
        self.setup_openpmd_record( dset, quantity )

        # Geometry parameters
        dset.attrs['geometry'] = np.bytes_("cartesian")
        dset.attrs['axisLabels'] = np.array([ b'x', b'y', b'z' ])
        dset.attrs['gridSpacing'] = self.mesh_spacing
        dset.attrs["gridGlobalOffset"] = self.mesh_origin

        # Generic attributes
        dset.attrs["dataOrder"] = np.bytes_("C")
        dset.attrs["gridUnitSI"] = 1.
        dset.attrs["fieldSmoothing"] = np.bytes_("none")

        # Generic setup of the component
        self.setup_openpmd_component( dset )

        # Field positions
        dset.attrs["position"] = np.array([0.0, 0.0, 0.0])

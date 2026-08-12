Betatron radiation
==================

FBPIC can accumulate classical synchrotron radiation while a simulation is
running. The diagnostic targets relativistic electrons and positrons in the
strong-wiggler regime,

.. math::

    \gamma \gg 1, \qquad \sigma_\theta \gg \gamma^{-1}.

It is an incoherent, passive diagnostic: each macroparticle contributes its
physical-particle weight times a single-particle radiated energy. It stores
neither field phase nor trajectories, does not sum complex amplitudes, and
does not modify particle momentum.

Configuration
-------------

Activate the model on every radiating species, then request only the products
needed by the simulation:

.. code-block:: python

    electrons.activate_synchrotron(
        gamma_cutoff=10.0,
        boost=sim.boost,
    )

    radiation = SynchrotronRadiationDiagnostic(
        period=200,
        species={"electrons": electrons},
        comm=sim.comm,
        observer_frame="laboratory",
        boost=sim.boost,
        photon_energy_edges=np.geomspace(100*e, 100.e3*e, 129),
        theta_x_edges=np.linspace(-20.e-3, 20.e-3, 81),
        theta_y_edges=np.linspace(-20.e-3, 20.e-3, 81),
        angular_measure="solid_angle",
        observer_time_edges=np.linspace(-50.e-15, 50.e-15, 401),
        detectors=[{
            "name": "on_axis",
            "direction": (0., 0., 1.),
            "half_angle": 2.e-3,
            "energy_bands": [{
                "name": "hard_xray",
                "energy_range": (5.e3*e, 30.e3*e),
            }],
        }],
        source_coordinate_edges={
            "x": np.linspace(-5.e-6, 5.e-6, 65),
            "y": np.linspace(-5.e-6, 5.e-6, 65),
            "z": np.linspace(0., 5.e-3, 101),
        },
        source_projections=[
            {"name": "xz", "axes": ("x", "z")},
            {"name": "x_energy", "axes": ("x", "energy")},
        ],
        source_moments=[
            {"name": "all"},
            {"name": "hard_xray", "energy_range": (5.e3*e, 30.e3*e)},
        ],
        source_z_intervals=(0.5, 0.9),
        resolution_warning_thresholds={
            "delta_eta": 0.1,
            "delta_theta_u": 0.01,
            "chi_turn": 1.0,
        },
        output_mode="both",
        samples_per_particle=2,
        random_seed=1234,
        max_allocation_bytes=1024**3,
    )

The boost may be omitted in an ordinary laboratory simulation. When it is
omitted from the diagnostic, the transform supplied at species activation is
used. An optional observer_translation=(ct, x, y, z) is applied after the
longitudinal Lorentz transform.

Only electron and positron species are accepted:
:math:`|q|=e` and :math:`m=m_e`. The gamma_cutoff must be greater than one;
the local synchrotron closure is not defined as a low-energy radiation model.

Local radiation model
---------------------

For the available PIC event tuple, transformed to the selected observer
frame, FBPIC evaluates

.. math::

    P_\perp =
    \frac{e^2}{6\pi\epsilon_0c}
    \frac{\gamma^2|\boldsymbol\beta\times\dot{\boldsymbol u}|^2}
         {|\boldsymbol\beta|^2},
    \qquad
    P_\parallel =
    \frac{e^2}{6\pi\epsilon_0c}
    \frac{\dot\gamma^2}{|\boldsymbol\beta|^2},

and

.. math::

    \omega_c =
    \frac{3}{2}\gamma^2
    \frac{|\boldsymbol\beta\times\dot{\boldsymbol u}|}
         {|\boldsymbol\beta|^3}.

Only :math:`P_\perp` normalizes the curvature spectrum,

.. math::

    \frac{\mathrm dP_{\rm syn}}{\mathrm dE_\gamma}
    = \frac{P_\perp}{\hbar\omega_c}
      S\!\left(\frac{E_\gamma}{\hbar\omega_c}\right),
    \qquad
    \int_0^\infty S(x)\,\mathrm dx=1.

:math:`P_\parallel` remains a separate accounting channel. Total transverse
and longitudinal energies are integrated directly and therefore do not depend
on a requested bin range or resolution.

The tabulated spectral CDF uses logarithmic positive spacing to resolve
:math:`S(x)\sim x^{1/3}`. The analytic energy fraction above x_max is not
renormalized into retained photons: it is written as
energy_truncated_by_spectral_closure and as record metadata.

Spectral--angular products use a fixed number of stratified energy packets per
emitting particle and event. Conditional on energy, the angle normal to the
instantaneous orbit plane is sampled from the polarization-summed Schwinger
distribution. The in-plane direction is the local velocity tangent. This is a
one-dimensional local strong-wiggler closure, not a phase-resolved harmonic or
complete two-dimensional formation-length calculation. Metadata records the
finite transformed-kernel range and the :math:`\pi/2` sampled-angle cap.

Angular coordinates are

.. math::

    \theta_x=\operatorname{atan2}(n_x,n_z), \qquad
    \theta_y=\operatorname{atan2}(n_y,n_z).

Their inverse projected chart represents only :math:`n_z>0`; bin edges and
projected-angle selections must therefore lie strictly inside
:math:`(-\pi/2,\pi/2)`. Fixed detector directions may be arbitrary unit
vectors. angular_measure="solid_angle" divides each forward angular cell by
its exact spherical area; "projected_angles" uses
:math:`\mathrm d\theta_x\mathrm d\theta_y`.

Observer-time detectors
-----------------------

For a fixed detector direction,

.. math::

    \tau_{\boldsymbol n}
    =t_{\rm obs}-\boldsymbol n\cdot\boldsymbol r_{\rm obs}/c,

and the broadband channel deposits the full local Lienard angular power,

.. math::

    \frac{\mathrm dP}{\mathrm d\Omega}
    =\frac{e^2}{16\pi^2\epsilon_0c}
    \frac{
      |\boldsymbol n\times[
       (\boldsymbol n-\boldsymbol\beta)\times\dot{\boldsymbol\beta}]|^2}
      {(1-\boldsymbol n\cdot\boldsymbol\beta)^5}.

The cancellation-prone denominator is evaluated through a positive stable
identity rather than direct subtraction. Circular broadband detector products
use deterministic equal-solid-angle Fibonacci quadrature. Aperture accounting
also classifies the retained spectral--angular packets into inside and outside
sets. These are deliberately separate estimators: the sampled outside energy
is not presented as the complement of the deterministic broadband integral.

Energy-filtered detector channels use the joint energy--angle closure by
default. Setting energy_band_mode="separable" selects the explicit fast
approximation

.. math::

    \Delta W_{12}(\boldsymbol n)
    \approx w\,\Delta t_{\rm obs}
      \frac{\mathrm dP_\perp}{\mathrm d\Omega}(\boldsymbol n)
      \left[
      C_S\!\left(\frac{E_2}{\hbar\omega_c}\right)
      -C_S\!\left(\frac{E_1}{\hbar\omega_c}\right)
      \right].

The separable mode combines the exact local transverse broadband pattern with
an angle-integrated spectral fraction and does not retain synchrotron
energy--angle coupling. Every affected record carries
bandEnergyAngleCouplingRetained=0 and a bandSpectralAngularClosure
description. The default joint mode instead uses the angle-conditioned
Schwinger energy CDF and records bandEnergyAngleCouplingRetained=1.

Source products
---------------

source_projections accepts any subset of x, y, z, theta_x, theta_y, energy,
and time for which edges exist. Coordinate-only projections with
coordinate-only cuts are deterministic and accumulate
:math:`wP_\perp\Delta t_{\rm obs}`. Projections involving photon energy,
angle, or time use the sampled local closure.

For a sampled photon direction, source time is

.. math::

    \tau_{\boldsymbol n_\gamma}
    =t_{\rm obs}
     -\boldsymbol n_\gamma\cdot\boldsymbol r_{\rm obs}/c.

It is a direction-conditioned radiation-phase coordinate in the joint source
distribution. Integrating out angle mixes distinct null coordinates and does
not yield the pulse measured by one physical detector. A source projection or
moment selection can instead set time_reference to a configured detector name;
then its source time is evaluated with that fixed detector direction. Output
metadata labels these conventions as sampled_photon_direction_source_time or
fixed_detector_referenced_source_time.

Source moments include position centroids and covariance, RMS and principal
transverse sizes, longitudinal extent, angle covariance, position--angle and
position--time correlations. The reported ellipticity is

.. math::

    (\sigma_{\rm major}-\sigma_{\rm minor})/
    (\sigma_{\rm major}+\sigma_{\rm minor}).

The major eigenvector is canonicalized to have a nonnegative x component;
orientation remains physically defined modulo :math:`\pi`.

Each source-moment selection may set its quantities field to any subset of
"position", "angle", and "time". The default requests all three. A
position-only request with coordinate-only cuts is accumulated
deterministically from integrated curvature energy and does not activate
spectral packet sampling. source_z_intervals requests equal-tail central
radiation-energy-containing intervals for every source-moment selection. Its
mergeable histogram has explicit underflow and overflow energy; the output
therefore exposes when the configured source_z_interval_edges do not cover a
requested interval.

Canonical pusher event and boost
--------------------------------

One event is attached to the completed momentum push centered at integer
simulation time :math:`t_n`, after the push and before the position push or any
discrete elementary process:

.. math::

    \left(x_n^\mu,\mathcal U_-^\mu,\mathcal U_+^\mu\right),
    \qquad
    \mathcal U_\pm^\mu=(\gamma_\pm,\boldsymbol u_\pm).

The centered four-velocity, proper-time interval, and four-acceleration are

.. math::

    \mathcal U_n^\mu =
    \frac{\mathcal U_+^\mu+\mathcal U_-^\mu}
         {\sqrt{(\mathcal U_++\mathcal U_-)^2}},
    \qquad
    \Delta\tau_n=\frac{\Delta t_{\rm sim}}{\mathcal U_n^0},

.. math::

    A_n^\mu =
    \frac{c(\mathcal U_+^\mu-\mathcal U_-^\mu)}{\Delta\tau_n}.

This construction preserves :math:`\mathcal U_n^2=1` and
:math:`\mathcal U_n\cdot A_n=0` discretely. The source event,
:math:`\mathcal U_n`, and :math:`A_n` receive the same Poincare transform. In
the detector frame the event duration is
:math:`\Delta t_{\rm obs}=\mathcal U_{\rm obs}^0\Delta\tau_n`.

The pusher streams lower endpoints through a fixed-capacity batch buffer and
consumes each batch immediately after its unchanged Vay push. It does not make
three full-species momentum copies. Ionization, collisions, and Compton
operations remain outside this continuous impulse.

Pusher-resolution indicators
----------------------------

Every eligible event contributes endpoint indicators

.. math::

    \Delta\eta=\operatorname{arcosh}(\mathcal U_+\cdot\mathcal U_-),
    \qquad
    \Delta\theta_u=\arccos(
      \widehat{\boldsymbol u}_+\cdot\widehat{\boldsymbol u}_-),
    \qquad
    \chi_{\rm turn}=\gamma_{n,{\rm sim}}\Delta\theta_u.

The implementation uses a cancellation-safe equivalent for
:math:`\Delta\eta`, clips the angle cosine, and assigns zero turning angle if
either endpoint momentum is zero. Cumulative and interval records contain the
maximum plus transverse-radiation-energy-weighted mean and RMS of each
indicator, and the emitted-energy fraction above each configured warning
threshold. These are quality indicators only and never reject or modify an
event.

Output lifecycle, reproducibility, and memory
---------------------------------------------

Every product is an independent openPMD mesh. Exact nonuniform edges are saved
under /data/<iteration>/radiationAxes and linked by axisEdgePaths; consumers
must use those edges instead of the fallback unit mesh spacing. File iteration
and time always refer to the latest completed pusher event included in that
file, not the latest detector arrival time.

Scheduled writes obey period, iteration_min, and iteration_max. They are
``radiation_scheduled_snapshot`` artifacts whose accumulation scope is the
currently open segment. Returning from ``Simulation.step`` neither closes a
segment nor forces an off-cadence write.

Checkpointed runs use half-open absolute event ranges. A checkpoint at
simulation iteration :math:`k` closes a canonical segment
``[eventBegin, k)``; therefore the completed event :math:`k-1` is included
once, while event :math:`k` is the first event a restart executes. The reduced
radiation state is written to ``<write_dir>/segments/segment-<id>.h5``. It
contains additive arrays and counters, centered source-moment sufficient
statistics, and retained interval histograms. The normal FBPIC checkpoint does
not contain these arrays. A small checkpoint manifest is published only after
both the simulation payload and every diagnostic segment have closed, making
an unreferenced or partially written segment orphaned and unacceptable by
default.

A restart reconstructs the diagnostic from its input configuration, verifies
its deterministic compatibility fingerprint and restored persistent particle
IDs, and opens a zeroed segment at the checkpoint iteration. The random event
key continues to use the absolute simulation iteration. An old checkpoint
without segment metadata is deliberately not treated as seamless; beginning a
new, discontinuous radiation lineage requires an explicit policy:

.. code-block:: python

    radiation = SynchrotronRadiationDiagnostic(
        ...,
        restart_policy="new_segment",
    )


At a true run ending, call ``simulation.finalize_diagnostics()`` (or
``radiation.finalize()``). This performs any dirty off-cadence snapshot and
commits the terminal segment with close reason ``finalize``. It is idempotent.
Continuing to step after finalization preserves the historical API but begins
a new zeroed radiation run with a new run identifier; it cannot silently
extend the already committed result.

Whole-run radiation is an offline product. Pass the committed segment files
for one selected checkpoint lineage to the strict merger (input order is not
significant):

.. code-block:: python

    from fbpic.openpmd_diag import merge_radiation_segments

    merge_radiation_segments(
        ["segment-a.h5", "segment-b.h5"],
        "radiation-whole-run.h5",
    )

The merger rejects uncommitted files, duplicate IDs, gaps, overlaps, broken
checkpoint ancestry, and incompatible fingerprints. It never silently rebins.
Additive state is summed, centered moment statistics are combined with the
parallel covariance formula, and source and observer-time quantiles are
reconstructed from the merged histograms. Only a selection that begins at a
lineage origin and ends in an explicit final segment is labeled
``radiation_merged_whole_run``; other valid selections are clearly labeled as
lineage selections. ``radiation_segment_status(path)`` reports whether a
persisted segment is committed or orphaned, while
``radiation.get_segment_status()`` also exposes the live open/closed state.

Packet work scales with samples_per_particle rather than photon-energy-bin
count. Each stateless random variate is keyed by diagnostic seed, persistent
uint64 particle ID, event index, independent stream ID, sample index, and a
stable species namespace. Consequently sorting, migration, batching, MPI
ownership, and CPU/GPU execution order do not change a particle event's
samples. Tracking is enabled automatically when needed; standard injection and
ionization paths assign new IDs before the first eligible event. Optional
particle thinning remains a Bernoulli Horvitz--Thompson estimator linear in
physical macroparticle weight.

Before product and workspace allocation, the diagnostic evaluates the
aggregate max_allocation_bytes budget across all species. get_memory_estimate()
returns a per-species byte breakdown covering dense products, centered moment
state, source-z histograms, lookup/configuration arrays, persistent identity,
the bounded endpoint and packet workspaces, MPI reductions, GPU transfer
peaks, interval baselines, and writer copies.

API reference
-------------

.. automethod:: fbpic.particles.Particles.activate_synchrotron

.. autoclass:: fbpic.openpmd_diag.SynchrotronRadiationDiagnostic

.. autofunction:: fbpic.openpmd_diag.merge_radiation_segments

.. autofunction:: fbpic.openpmd_diag.radiation_segment_status

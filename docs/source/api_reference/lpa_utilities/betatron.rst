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
        output_mode="both",
        samples_per_particle=2,
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
identity rather than direct subtraction. Circular apertures use deterministic
equal-solid-angle Fibonacci quadrature and do not activate spectral packet
sampling by themselves.

Energy-filtered detector channels use the explicit separable closure

.. math::

    \Delta W_{12}(\boldsymbol n)
    \approx w\,\Delta t_{\rm obs}
      \frac{\mathrm dP_\perp}{\mathrm d\Omega}(\boldsymbol n)
      \left[
      C_S\!\left(\frac{E_2}{\hbar\omega_c}\right)
      -C_S\!\left(\frac{E_1}{\hbar\omega_c}\right)
      \right].

This combines the exact local transverse broadband pattern with an
angle-integrated spectral fraction; it does not retain synchrotron
energy--angle coupling. Every affected record carries
bandEnergyAngleCouplingRetained=0 and a bandSpectralAngularClosure
description. Narrow, hard, or off-axis bands should not be interpreted as the
same joint kernel used by packet products.

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
not yield the pulse measured by one physical detector. The output metadata
states this distinction for every source-time product and source-moment
record.

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
spectral packet sampling.

Boost and PIC staggering
------------------------

The simulation-to-observer transform reconstructs position, emission time,
momentum, and fields and applies the particle-dependent worldline interval

.. math::

    \frac{\mathrm dt_{\rm obs}}{\mathrm dt_{\rm sim}}
    =\frac{\gamma_{\rm obs}}{\gamma_{\rm sim}}.

This is the complete event tuple available to the elementary process, not an
exactly simultaneous continuum event: particle position and momentum are at
the half step, while fields were gathered at the preceding integer step.
Boosted and unboosted results should therefore be compared under timestep
convergence. The staggering is stored as picEventTimeStaggering.

Output and reproducibility
--------------------------

Every product is an independent openPMD mesh. Exact nonuniform edges are saved
under /data/<iteration>/radiationAxes and linked by axisEdgePaths; consumers
must use those edges instead of the fallback unit mesh spacing. Iteration time
is the observer time of the simulation-origin reference event, not the latest
arrival time in a detector profile.

Packet work scales with samples_per_particle rather than photon-energy-bin
count. Random sampling uses the NumPy or CuPy stream controlled by
:func:`fbpic.utils.random_seed.set_random_seed`. Exact realizations can still
change with backend, MPI decomposition, particle ordering, or population
history. Broadband detector directions, deterministic apertures, accounting,
and coordinate-only source projections introduce no packet noise.

API reference
-------------

.. automethod:: fbpic.particles.Particles.activate_synchrotron

.. autoclass:: fbpic.openpmd_diag.SynchrotronRadiationDiagnostic

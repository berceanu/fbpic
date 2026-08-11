Betatron radiation
==================

FBPIC can calculate classical synchrotron radiation on the fly.  The model
is intended for relativistic particles in the strong-wiggler regime,

.. math::

    \gamma \gg 1, \qquad \sigma_\theta \gg \gamma^{-1}.

This includes, for example, plasma betatron radiation.  The calculation is
incoherent: every macroparticle contributes its weight times the
single-electron energy.  It does not retain interparticle phase, sum complex
far-field amplitudes, or perform a trajectory Fourier transform.

Fast observer-frame diagnostic
------------------------------

The production interface separates the radiation products so a simulation
only allocates and updates the channels it needs.  Activate passive emission
without legacy axes, then configure observer-frame bin *edges* on the
diagnostic:

.. code-block:: python

    electrons.activate_synchrotron(
        gamma_cutoff=10.0, boost=sim.boost
    )

    radiation = SynchrotronRadiationDiagnostic(
        period=200,
        species={"electrons": electrons},
        comm=sim.comm,
        observer_frame="laboratory",
        boost=sim.boost,
        photon_energy_bin_edges=np.geomspace(100*e, 100.e3*e, 129),
        angular_bin_edges={
            "theta_x": np.linspace(-20.e-3, 20.e-3, 81),
            "theta_y": np.linspace(-20.e-3, 20.e-3, 81),
        },
        angular_measure="solid_angle",
        observer_time_bin_edges=np.linspace(-50.e-15, 50.e-15, 401),
        detectors=[{
            "name": "on_axis",
            "direction": (0., 0., 1.),
            "half_angle": 2.e-3,
            "energy_bands": [{
                "name": "hard_xray",
                "energy_range": (5.e3*e, 30.e3*e),
            }],
        }],
        source_coordinate_bin_edges={
            "x": np.linspace(-5.e-6, 5.e-6, 65),
            "y": np.linspace(-5.e-6, 5.e-6, 65),
            "z": np.linspace(0., 5.e-3, 101),
        },
        source_distribution_projections=[
            {"name": "xz", "axes": ("x", "z")},
            {"name": "x_energy", "axes": ("x", "energy")},
        ],
        source_moment_selections=[
            {"name": "all"},
            {"name": "hard_xray", "energy_range": (5.e3*e, 30.e3*e)},
        ],
        output_mode="both",
        local_angular_model="synchrotron",
        samples_per_particle=2,
    )

The boost may be omitted in an ordinary laboratory simulation.  When it is
also passed to :meth:`~fbpic.particles.Particles.activate_synchrotron`, the
diagnostic can infer it, but specifying it at the diagnostic makes the output
frame explicit.  ``observer_translation=(ct, x, y, z)`` optionally supplies
the four-translation :math:`b^\mu`.  All bin edges, cutoffs, selections,
directions, source coordinates, and times above are observer-frame values.

Local power and spectral closure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the selected observer frame, FBPIC evaluates

.. math::

    P_\perp = \frac{e^2}{6\pi\epsilon_0c}
      \frac{\gamma^2|\boldsymbol\beta\mathbin{\times}
      \dot{\boldsymbol u}|^2}{|\boldsymbol\beta|^2}, \qquad
    P_\parallel = \frac{e^2}{6\pi\epsilon_0c}
      \frac{\dot\gamma^2}{|\boldsymbol\beta|^2}.

The curvature critical frequency is

.. math::

    \omega_c=\frac{3}{2}\gamma^2
      \frac{|\boldsymbol\beta\mathbin{\times}\dot{\boldsymbol u}|}
           {|\boldsymbol\beta|^3}.

Only :math:`P_\perp` normalizes the curvature spectrum,

.. math::

    \frac{\mathrm dP_{\rm syn}}{\mathrm dE_\gamma}
      = \frac{P_\perp}{\hbar\omega_c}
        S\!\left(\frac{E_\gamma}{\hbar\omega_c}\right),
    \qquad \int_0^\infty S(x)\,\mathrm dx=1.

:math:`P_\parallel` and its integrated fraction are separate accounting
records; it is never inserted into the curvature spectrum.  Total emitted
energy is accumulated directly as
:math:`w(P_\perp+P_\parallel)\Delta t_{\rm obs}` and therefore does not
depend on a requested bin width or range.

The default ``synchrotron`` angular closure draws a fixed, configurable
number of stratified samples from :math:`S`.  For each sampled photon energy,
the transverse curvature fixes the local orbit plane.  The angle normal to
that plane is sampled from a pretabulated, polarization-summed Schwinger
spectral--angular conditional distribution, including its dependence on
:math:`E_\gamma/(\hbar\omega_c)`; the local velocity supplies the tangent
direction in the plane.  Thus a particle event does not place its whole
spectrum at one energy-independent random angle.  This is a local
strong-wiggler closure, not a phase-resolved harmonic model.
``legacy_gaussian`` retains the former energy-independent Gaussian cone as an
explicit compatibility choice.

``angular_measure="projected_angles"`` writes density per
:math:`\mathrm d\theta_x\mathrm d\theta_y`.  ``solid_angle`` uses the same
projected-angle coordinates but divides every angular cell by its exact
spherical quadrilateral area :math:`\mathrm d\Omega`.  The selected measure
and the original edges are stored on every relevant record.
The corresponding record is
:math:`\mathcal R(\boldsymbol\theta,E_\gamma)=
\mathrm d^3W/(\mathrm d\mu_\theta\mathrm dE_\gamma)`; it therefore
integrates back to represented curvature energy under the selected angular
measure and photon-energy edges.

Observer-time detectors
~~~~~~~~~~~~~~~~~~~~~~~

For each direction, emission is placed at

.. math::

    \tau_{\rm obs}=t_{\rm obs}
       -\boldsymbol n\mathbin{\cdot}\boldsymbol r_{\rm obs}/c

with the exact local broadband angular power

.. math::

    \frac{\mathrm dP}{\mathrm d\Omega}
    =\frac{e^2}{16\pi^2\epsilon_0c}
    \frac{|\boldsymbol n\mathbin{\times}[(\boldsymbol n-
    \boldsymbol\beta)\mathbin{\times}\dot{\boldsymbol\beta}]|^2}
    {(1-\boldsymbol n\mathbin{\cdot}\boldsymbol\beta)^5}.

A positive ``half_angle`` adds a circular aperture.  Its broadband energy is
integrated with equal-solid-angle deterministic quadrature; increase
``aperture_quadrature`` when resolving a cone with sharp angular structure.
The output includes bin-averaged power, integrated pulse energy, peak power,
and the requested cumulative-energy interval (5--95 percent by default).
Optional energy bands apply the normalized *curvature* spectral closure and
the exact Liénard angular pattern formed from the transverse acceleration.
They remain explicitly marked as excluding longitudinal power and do not
require spectral Monte-Carlo packets for a direction-only detector.

Source distributions and moments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``source_distribution_projections`` accepts any requested subset of
``x``, ``y``, ``z``, ``theta_x``, ``theta_y``, ``energy``, and ``time`` for
which edges were supplied.  The string ``"full"`` requests their full joint
distribution; reduced tuples such as ``("x", "z")`` avoid that memory cost.
Each bin stores curvature-radiation energy before the writer divides by its
physical bin measure.  Projections containing only source coordinates and
coordinate cuts are accumulated deterministically from
:math:`wP_\perp\Delta t_{\rm obs}` and require no photon packet; adding an
energy, angle, or observer-time axis activates the local sampled closure.

Conceptually, these records and projections discretize

.. math::

    B(\boldsymbol x_s,\boldsymbol\theta,E_\gamma)
      = \sum_{p,n} w_p
        \frac{\mathrm d^3P_{pn}}
             {\mathrm dE_\gamma\mathrm d\mu_\theta}
        \delta^{(3)}(\boldsymbol x_s-\boldsymbol r_{pn,\rm obs})
        \delta^{(2)}(\boldsymbol\theta-\boldsymbol\theta_{pn,\rm obs})
        \Delta t_{pn,\rm obs},

with every quantity evaluated in the selected observer frame.

Moment selections accept ``energy_range``, ``angular_range``,
``observer_time_range``, coordinate ranges, or a direction and half angle.
``energy_bins`` and paired ``theta_x_bins``/``theta_y_bins`` expand into a
series of selections.  For each one the output includes source energy,
centroid, the full position covariance, RMS sizes and longitudinal extent,
transverse principal sizes, ellipticity and orientation, angle covariance,
position-angle covariance and correlations, and position-observer-time
covariance and correlations.

Boosted emission events
~~~~~~~~~~~~~~~~~~~~~~~

The new diagnostic transforms the complete synchronized event, not only its
momentum.  For a simulation-to-observer Lorentz transform :math:`\Lambda` and
translation :math:`b`, it applies

.. math::

    x_d^\mu=\Lambda^\mu{}_{\nu}x_s^\nu+b^\mu,\qquad
    U_d^\mu=\Lambda^\mu{}_{\nu}U_s^\nu,\qquad
    F_d^{\mu\nu}=\Lambda^\mu{}_{\alpha}\Lambda^\nu{}_{\beta}
      F_s^{\alpha\beta},

and uses the particle-dependent interval

.. math::

    \Delta t_d=\frac{\gamma_d}{\gamma_s}\Delta t_s.

Consequently source position, emission time, momentum, gathered fields,
critical energy, angle, and every selection are in one frame.  The openPMD
iteration time is the observer time of the simulation-origin reference event;
the record metadata states that reference explicitly.

Output and performance
~~~~~~~~~~~~~~~~~~~~~~

The angular spectrum, each detector/profile, and each source projection is an
independent openPMD mesh.  Arbitrary edges are stored under
``/data/<iteration>/radiationAxes`` and linked by ``axisEdgePaths`` metadata.
Accounting, source-moment, and pulse-metric records carry component-specific
SI unit dimensions, selections, frame transform, model, and accumulation
mode.  MPI ranks are reduced before writing.  ``output_mode`` can be
``cumulative``, ``interval``, or ``both``.

Accounting records include transverse and longitudinal emitted energy and
their fraction, the deterministic curvature energy below, within, and above
the requested photon-energy range, and the energy represented by every mesh
or detector channel.  Loss outside the angular grid or union of configured
apertures is estimated from the sampled local curvature closure and is labeled
as such.  Every quantity is available in both cumulative and interval form
when ``output_mode="both"``.

Per-step spectral work is a fixed
``samples_per_particle`` rather than the number of photon-energy bins.  No
particle-by-energy temporary array is formed.  Detector cost scales with the
number of requested directions and aperture quadrature points.  Disabled
channels allocate no product arrays and perform no channel-specific work.
Full source distributions can still be large by choice; moments and reduced
projections are the intended large-population defaults.

Legacy maximal histogram
------------------------

The original activation call with three ``(min, max, count)`` axes remains
available unchanged.  It allocates one maximal
``theta_x x theta_y x photon_energy`` histogram, evaluates every energy at
every particle step, and uses the historical Gaussian angular sample.  The
following sections document that compatibility path.  New simulations should
prefer the edge-based, independently selectable interface above.

Legacy laboratory-frame radiation model
---------------------------------------

Let :math:`\boldsymbol u=\boldsymbol p/(m c)`,
:math:`\gamma=(1+|\boldsymbol u|^2)^{1/2}`, and
:math:`\boldsymbol\beta=\boldsymbol u/\gamma`.  At the particle event,

.. math::

    \dot{\boldsymbol u} = -\frac{e}{m c}
    \left(\boldsymbol E+\boldsymbol\beta\mathbin{\times}c\boldsymbol B\right),
    \qquad
    \dot\gamma = -\frac{e}{m c}\boldsymbol\beta\mathbin{\cdot}
    \boldsymbol E.

The relativistic Larmor power is

.. math::

    P = \frac{e^2}{6\pi\epsilon_0 c}\,\mathcal A^2,

where the squared proper acceleration can be evaluated without subtracting
nearly equal terms as

.. math::

    \mathcal A^2
    = \gamma^6\left(
      |\dot{\boldsymbol\beta}|^2
      -|\boldsymbol\beta\mathbin{\times}
        \dot{\boldsymbol\beta}|^2\right)
    = \frac{
      \gamma^2|\boldsymbol\beta\mathbin{\times}
        \dot{\boldsymbol u}|^2+\dot\gamma^2}
      {|\boldsymbol\beta|^2}.

The critical angular frequency is determined by the trajectory curvature,

.. math::

    \omega_c = \frac{3}{2}\gamma^3
    \frac{|\boldsymbol\beta\mathbin{\times}
      \dot{\boldsymbol\beta}|}{|\boldsymbol\beta|^3}
    = \frac{3}{2}\gamma^2
    \frac{|\boldsymbol\beta\mathbin{\times}
      \dot{\boldsymbol u}|}{|\boldsymbol\beta|^3}.

For a laboratory time interval :math:`\Delta t`, the spectral energy is

.. math::

    \frac{\partial E_{\mathrm{rad}}}{\partial(\hbar\omega)}
    = \frac{P\,\Delta t}{\hbar\omega_c}
      S\left(\frac{\omega}{\omega_c}\right),

with

.. math::

    S(x) = \frac{9\sqrt{3}}{8\pi}\,x
    \int_x^\infty K_{5/3}(\xi)\,\mathrm d\xi.

The central photon direction is the particle direction.  FBPIC represents
the model's angular profile by adding independent normal samples to the two
projected angles,

.. math::

    \theta_x=\operatorname{atan2}(u_x,u_z), \qquad
    \theta_y=\operatorname{atan2}(u_y,u_z), \qquad
    \sigma_\theta=2^{-3/2}\gamma^{-1}.

The samples are deposited bilinearly on the angular grid.  The energy
spectrum is evaluated directly at every photon energy requested by the user.

Legacy boosted-frame simulations
--------------------------------

The diagnostic can calculate this laboratory-frame radiation while the PIC
simulation runs in a longitudinally boosted frame.  Pass the simulation's
:class:`~fbpic.lpa_utils.boosted_frame.BoostConverter` to ``boost`` when
activating synchrotron radiation.  Photon energies, angles,
``gamma_cutoff``, and the accumulated output are then all interpreted in the
laboratory frame.

FBPIC's simulation frame, denoted by a prime below, moves along :math:`+z`
with laboratory velocity :math:`\beta_b c` and Lorentz factor
:math:`\Gamma_b`.  Its laboratory-to-simulation momentum convention is

.. math::

    \gamma' = \Gamma_b(\gamma-\beta_b u_z), \qquad
    u'_z = \Gamma_b(u_z-\beta_b\gamma).

Consequently, the diagnostic reconstructs the laboratory momentum with the
inverse transform

.. math::

    \gamma = \Gamma_b(\gamma'+\beta_b u'_z), \qquad
    u_z = \Gamma_b(u'_z+\beta_b\gamma'), \qquad
    u_x=u'_x, \quad u_y=u'_y.

The local fields are reconstructed with the same sign convention,

.. math::

    \begin{aligned}
    E_x &= \Gamma_b(E'_x+\beta_b cB'_y), &
    E_y &= \Gamma_b(E'_y-\beta_b cB'_x), & E_z &= E'_z,\\
    cB_x &= \Gamma_b(cB'_x-\beta_b E'_y), &
    cB_y &= \Gamma_b(cB'_y+\beta_b E'_x), & cB_z &= cB'_z.
    \end{aligned}

Coordinate time is not Lorentz invariant.  Along each particle worldline,

.. math::

    \frac{\mathrm dt}{\mathrm dt'}
    = \Gamma_b(1+\beta_b\beta'_z)
    = \frac{\gamma}{\gamma'},

where the last equality also follows from
:math:`\mathrm d\tau=\mathrm dt/\gamma=\mathrm dt'/\gamma'`.  FBPIC
therefore multiplies each simulation-frame step by this particle-dependent
factor before accumulating :math:`P\,\mathrm dt`.  This worldline factor is
required even after momenta and fields have been transformed.

As in the original laboratory-frame diagnostic, the finite-step PIC
evaluation uses synchronized particle positions and momenta at a half step
with fields gathered at the preceding integer step.  It also approximates
the worldline integral with the step-local value of
:math:`\gamma/\gamma'`.  This staggered finite-step tuple and quadrature are
not exactly Lorentz covariant.  Laboratory- and boosted-frame PIC results
therefore agree in the timestep-converged limit rather than being expected
to match exactly at finite resolution.  Comparisons between frames should
include a timestep-convergence check.

All radiation formulas above are then evaluated from the reconstructed
laboratory quantities.  In particular, the cutoff and angular spread use
the laboratory :math:`\gamma`, and :math:`\omega_c` and the photon-energy
axis use laboratory frequencies.  The diagnostic does not compute a
boosted-frame histogram and relabel it.  It deposits directly on the user's
laboratory energy-angle grid, so no photon energy-angle Jacobian is needed.

A macroparticle weight is the number of represented particles and is a
Lorentz scalar.  It therefore multiplies the reconstructed single-particle
emission without an additional boost factor.  The output quantity is the
spectral-angular density

.. math::

    R = \frac{\partial^3 E_{\mathrm{rad}}}
      {\partial(\hbar\omega)\,\partial\theta_x\,\partial\theta_y}.

Since radians are dimensionless in SI, :math:`R` is dimensionless (and may
also be read as per radian squared).  The accumulated energy represented by
the output grid is obtained by numerical quadrature.  For example, using
trapezoidal weights :math:`q_k` on the linearly spaced photon-energy axis,

.. math::

    E_{\mathrm{grid}} \simeq \sum_{i,j,k} q_k R_{ijk}\,
    \Delta\theta_x\,\Delta\theta_y\,\Delta(\hbar\omega),

where the endpoint weights are one half.  The angular sum is histogram-like,
whereas the energy dependence is sampled at grid points; downstream analysis
should therefore state its energy-quadrature convention.

This normalization is unchanged between laboratory and boosted simulations.
MPI ranks are summed before the standard openPMD output is written.  The
openPMD energy and angular coordinates are laboratory coordinates.  The
file time remains the simulation-frame time, and an intermediate cumulative
output generally ends at different laboratory times along different
particle worldlines; it is not a laboratory simultaneity snapshot.

For example, in a boosted simulation use

.. code-block:: python

    electrons.activate_synchrotron(
        photon_energy_axis, theta_x_axis, theta_y_axis,
        gamma_cutoff=10.0, boost=sim.boost
    )

In an unboosted simulation, omitting ``boost`` retains the legacy
laboratory-frame behavior.

Legacy random sampling and radiation reaction
---------------------------------------------

Call :func:`fbpic.utils.random_seed.set_random_seed` before the simulation
to make angular sampling repeatable.  The sampled sequences are repeatable
for a fixed backend, MPI rank count and decomposition, particle ordering,
and population history.  The CPU and CUDA generators are different, so their
individual angular samples are not expected to be identical.  CUDA atomic
accumulation may also vary at floating-point roundoff.  Their distributions
and integrated spectra are equivalent.  Calling ``set_random_seed`` again
restarts synchrotron sampling at the next accumulation, including persistent
CUDA streams.

The existing laboratory-frame implementation can optionally apply a
classical recoil along the electron direction.  That recoil cannot be
subtracted directly from simulation-frame momentum.  Therefore FBPIC raises
``NotImplementedError`` if ``radiation_reaction=True`` is combined with a
non-identity ``boost``.  Boosted laboratory-frame diagnosis is supported
with ``radiation_reaction=False`` and does not modify particle dynamics.

Legacy scope and output
-----------------------

This diagnostic keeps the original incoherent, time-integrated,
strong-wiggler approximation.  It is not a phase-resolved undulator,
coherent-radiation, Compton, or retarded-field calculation.  The finite
energy-angle region supplied by the user determines which part of the
radiation is recorded.

Activate the plugin for an electron species with

.. automethod:: fbpic.particles.Particles.activate_synchrotron

Write the resulting laboratory-frame spectral-angular density with

.. autoclass:: fbpic.openpmd_diag.SynchrotronRadiationDiagnostic

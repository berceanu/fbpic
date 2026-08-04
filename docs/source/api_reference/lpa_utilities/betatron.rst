Betatron radiation
==================

FBPIC can calculate classical synchrotron radiation on the fly.  The model
is intended for relativistic particles in the strong-wiggler regime,

.. math::

    \gamma \gg 1, \qquad \sigma_\theta \gg \gamma^{-1}.

This includes, for example, plasma betatron radiation.  The calculation is
incoherent and time integrated: it does not retain the radiation phase or
evaluate retarded fields.

Laboratory-frame radiation model
--------------------------------

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

Boosted-frame simulations
-------------------------

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
the output grid is

.. math::

    E_{\mathrm{grid}} = \sum_{i,j,k} R_{ijk}\,
    \Delta\theta_x\,\Delta\theta_y\,\Delta(\hbar\omega).

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

Random sampling and radiation reaction
--------------------------------------

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

Scope and output
----------------

This diagnostic keeps the original incoherent, time-integrated,
strong-wiggler approximation.  It is not a phase-resolved undulator,
coherent-radiation, Compton, or retarded-field calculation.  The finite
energy-angle region supplied by the user determines which part of the
radiation is recorded.

Activate the plugin for an electron species with

.. automethod:: fbpic.particles.Particles.activate_synchrotron

Write the resulting laboratory-frame spectral-angular density with

.. autoclass:: fbpic.openpmd_diag.SynchrotronRadiationDiagnostic

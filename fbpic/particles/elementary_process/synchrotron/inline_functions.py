"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines inline functions that are compiled for both GPU and CPU, and
used in the syncrotron radiation code.
"""
import math
from scipy.constants import c, e, m_e

e_mc = e / ( m_e * c)


def get_particle_lab_frame(
    ux, uy, uz, gamma_inv, gamma_boost, beta_boost
    ):
    """
    Transform particle momentum from the simulation frame to the lab frame.

    The simulation frame moves along ``+z`` in the lab. Therefore the inverse
    transform used here has a plus sign. The returned time factor converts a
    simulation-frame worldline element to the corresponding lab-frame
    element: ``dt_lab = dt_ratio * dt_sim``.

    Parameters
    ----------
    ux, uy, uz: floats
        Components of normalized momentum in the simulation frame.

    gamma_inv: float
        Reciprocal Lorentz factor in the simulation frame.

    gamma_boost, beta_boost: floats
        Lorentz factor and normalized velocity of the simulation frame.

    Returns
    -------
    ux_lab, uy_lab, uz_lab: floats
        Components of normalized momentum in the lab frame.

    gamma_inv_lab: float
        Reciprocal Lorentz factor in the lab frame.

    dt_ratio: float
        Ratio ``dt_lab/dt_sim`` along the particle worldline.
    """
    if beta_boost == 0.0:
        return ux, uy, uz, gamma_inv, 1.0

    gamma_sim = 1.0 / gamma_inv
    lightfront_boost = gamma_boost * (1.0 + beta_boost)
    transverse_mass2 = 1.0 + ux * ux + uy * uy

    # Form the small light-front component from the mass shell instead of
    # subtracting nearly equal numbers. This remains accurate when a large
    # boost transforms a low-energy lab particle into one moving near -c.
    if uz >= 0.0:
        p_plus_sim = gamma_sim + uz
        p_minus_sim = transverse_mass2 / p_plus_sim
    else:
        p_minus_sim = gamma_sim - uz
        p_plus_sim = transverse_mass2 / p_minus_sim

    p_plus_lab = lightfront_boost * p_plus_sim
    p_minus_lab = p_minus_sim / lightfront_boost
    gamma_lab = 0.5 * (p_plus_lab + p_minus_lab)
    uz_lab = 0.5 * (p_plus_lab - p_minus_lab)
    # The subtraction above is unavoidable near exactly transverse motion.
    # Remove only cancellation noise at the scale of floating-point roundoff;
    # otherwise atan2(0, -epsilon) would spuriously return pi instead of zero.
    if abs(uz_lab) <= 8.0 * 2.220446049250313e-16 * gamma_lab:
        uz_lab = 0.0

    dt_ratio = gamma_lab * gamma_inv

    return ux, uy, uz_lab, 1.0 / gamma_lab, dt_ratio


def get_fields_lab_frame(
    Ex, Ey, Ez, cBx, cBy, cBz, gamma_boost, beta_boost
    ):
    """
    Transform local electromagnetic fields to the laboratory frame.

    Magnetic-field components are passed multiplied by the speed of light,
    so that all field arguments have the same units.

    Parameters
    ----------
    Ex, Ey, Ez: floats
        Electric-field components in the simulation frame.

    cBx, cBy, cBz: floats
        Magnetic-field components in the simulation frame, multiplied by
        the speed of light.

    gamma_boost, beta_boost: floats
        Lorentz factor and normalized velocity of the simulation frame.

    Returns
    -------
    Ex_lab, Ey_lab, Ez_lab: floats
        Electric-field components in the lab frame.

    cBx_lab, cBy_lab, cBz_lab: floats
        Lab-frame magnetic-field components multiplied by the speed of light.
    """
    if beta_boost == 0.0:
        return Ex, Ey, Ez, cBx, cBy, cBz

    lightfront_boost = gamma_boost * (1.0 + beta_boost)

    # Transform field light-front pairs before recombining them. This avoids
    # cancellation between a large boost factor and ``1 - beta_boost``.
    Ex_plus_cBy = lightfront_boost * (Ex + cBy)
    Ex_minus_cBy = (Ex - cBy) / lightfront_boost
    Ey_minus_cBx = lightfront_boost * (Ey - cBx)
    Ey_plus_cBx = (Ey + cBx) / lightfront_boost

    Ex_lab = 0.5 * (Ex_plus_cBy + Ex_minus_cBy)
    cBy_lab = 0.5 * (Ex_plus_cBy - Ex_minus_cBy)
    Ey_lab = 0.5 * (Ey_plus_cBx + Ey_minus_cBx)
    cBx_lab = 0.5 * (Ey_plus_cBx - Ey_minus_cBx)

    return Ex_lab, Ey_lab, Ez, cBx_lab, cBy_lab, cBz


def get_angles( ux, uy, uz ):
    """
    Calculate angular projections of particle momentum vector on
    `(theta_x, theta_y)` plane.

    Parameters
    ----------
    ux, uy, uz: floats
        Components of particles normalized momentum.

    Returns
    -------
    theta_x, theta_y: floats
        angular projections of particle momentum vector in radians
    """

    theta_x = math.atan2( ux, uz )
    theta_y = math.atan2( uy, uz )

    return( theta_x, theta_y )

def get_linear_coefficients(x, xmin, dx):
    """
    Calculate shape coefficients and index for the 1D linear
    interpolation on a uniform grid (used for angle projections)

    Parameters
    ----------
    x: float
        Coordinate to be projected.

    xmin: float
         Grid origin

    dx: float
        Grid step

    Returns
    -------
    ix: integer
        Index of the cell that contains the point

    s0, s1: floats
        Weights projected to the left and right nodes of the cell
    """
    s_ix = ( x - xmin ) / dx
    ix = math.floor( s_ix )
    s1 = s_ix - ix
    s0 = 1.0 - s1
    ix = int(ix)

    return ix, s0, s1

def get_particle_radiation(
    ux, uy, uz, w,
    Ex, Ey, Ez,
    cBx, cBy, cBz,
    gamma_inv,
    Larmore_factor_density,
    Larmore_factor_momentum,
    SR_dxi, SR_xi_data,
    omega_ax, spect_loc
    ):

    """
    Calculate spectal energy distribution emitted by the particle.

    Parameters
    ----------
    ux, uy, uz, w: floats
        Components momentum and weight of the particle

    Ex, Ey, Ez: float
         Components of electric field on the particle (V/m)

    cBx, cBy, cBz: float
         Components of magnetic field on the particle multiplied by
         the speed of light (V/m)

    gamma_inv: float
        Reciprocal of particle Lorentz factor

    Larmore_factor_density: float
        Normalization factor for spectral-angular density,
        `e**2 * dt / (6 * np.pi * epsilon_0 * c * hbar * d_theta_x * d_theta_y)`

    Larmore_factor_momentum: float
        Normalization factor for the photon momentum,
        ``e**2 * dt / (6 * pi * epsilon_0 * m_e * c**3)``

    omega_ax: 1D vector of floats
        frequencies on which spectrum is calculated

    spect_loc: 1D vector of floats
        calculated spectral density of the radiation

    Returns
    -------
    spect_loc: 1D vector of floats
        calculated spectral density of the radiation
    """
    d_omega = omega_ax[1] - omega_ax[0]
    N_omega_src = SR_xi_data.size
    gamma = 1. / gamma_inv

    # get normalized velocity
    beta_x = ux * gamma_inv
    beta_y = uy * gamma_inv
    beta_z = uz * gamma_inv

    # get momentum time derivative
    dt_ux = - e_mc * ( Ex + beta_y * cBz - beta_z * cBy )
    dt_uy = - e_mc * ( Ey + beta_z * cBx - beta_x * cBz )
    dt_uz = - e_mc * ( Ez + beta_x * cBy - beta_y * cBx )

    # get Lorentz factor derivative
    dt_gamma = - e_mc * (Ex * beta_x + Ey * beta_y + Ez * beta_z)

    beta_abs2 = beta_x**2 + beta_y**2 + beta_z**2
    if beta_abs2 <= 0.0:
        spect_loc[:] = 0.0
        return( spect_loc, 0.0, 0.0, 0.0 )
    beta_abs2_inv = 1. / beta_abs2

    # Stable form of the squared proper acceleration. Decomposing du/dt
    # parallel and normal to beta avoids subtracting ultrarelativistic terms.
    cross_u_x = beta_y * dt_uz - beta_z * dt_uy
    cross_u_y = beta_z * dt_ux - beta_x * dt_uz
    cross_u_z = beta_x * dt_uy - beta_y * dt_ux
    cross_u_abs2 = cross_u_x**2 + cross_u_y**2 + cross_u_z**2
    Energy_norm = w * (
        gamma**2 * cross_u_abs2 + dt_gamma**2
    ) * beta_abs2_inv

    # calculate emitted radiation momentum
    Momentum_Larmor = Larmore_factor_momentum * Energy_norm / w
    u_abs_inv = 1. / math.sqrt(ux * ux + uy * uy + uz * uz )
    # or
    # u_abs_inv = 1. / math.sqrt(1 + gamma*gamma )
    ux_ph = Momentum_Larmor * ux * u_abs_inv
    uy_ph = Momentum_Larmor * uy * u_abs_inv
    uz_ph = Momentum_Larmor * uz * u_abs_inv

    # Normal coordinate acceleration, again without a parallel subtraction.
    curvature_accel2 = cross_u_abs2 * gamma_inv**2 * beta_abs2_inv
    if curvature_accel2 == 0.0:
        spect_loc[:] = 0.0
        return( spect_loc, 0.0, 0.0, 0.0 )

    omega_c = 1.5 * gamma**3 * beta_abs2_inv  * \
        math.sqrt( curvature_accel2 )

    # discard too low critical frequencies as not resolved
    if (omega_c < 4 * d_omega):
        spect_loc[:] = 0.0
        return( spect_loc, 0.0, 0.0, 0.0 )

    omega_c_inv = 1. / omega_c

    Density_Larmore = Larmore_factor_density * Energy_norm  * omega_c_inv

    # Loop over the frequencies to project the spectrum
    for i_omega in range(omega_ax.size):
        xi_loc = omega_ax[i_omega] * omega_c_inv
        s_ix_src = xi_loc / SR_dxi
        ix_src = math.floor( s_ix_src )

        if ( ix_src >= N_omega_src - 1 ):
            spect_loc[i_omega] = 0.0
        else:
            s1 = s_ix_src - ix_src
            s0 = 1.0 - s1
            ix_src_int = int(ix_src)
            S_xi_loc = SR_xi_data[ix_src_int] * s0 + SR_xi_data[ix_src_int+1] * s1
            spect_loc[i_omega] = Density_Larmore * S_xi_loc

    return( spect_loc, ux_ph, uy_ph, uz_ph )

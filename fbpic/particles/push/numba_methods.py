# Copyright 2016, FBPIC contributors
# Authors: Remi Lehe, Manuel Kirchen
# License: 3-Clause-BSD-LBNL
"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the particle push methods on the CPU with numba.
"""
import numba
from fbpic.utils.threading import njit_parallel, prange
from scipy.constants import c, e
# Import inline functions
from .inline_functions import push_p_vay
# Compile the inline functions for CPU
push_p_vay = numba.njit(push_p_vay)

@njit_parallel
def push_x_numba( x, y, z, ux, uy, uz, inv_gamma, Ntot, dt,
                push_x, push_y, push_z ):
    """
    Advance the particles' positions over `dt` using the momenta ux, uy, uz,
    multiplied by the scalar coefficients x_push, y_push, z_push.
    """
    # Half timestep, multiplied by c
    chdt = c*dt

    # Particle push (in parallel if threading is installed)
    for ip in prange(Ntot) :
        x[ip] += chdt * inv_gamma[ip] * push_x * ux[ip]
        y[ip] += chdt * inv_gamma[ip] * push_y * uy[ip]
        z[ip] += chdt * inv_gamma[ip] * push_z * uz[ip]

    return x, y, z

@njit_parallel
def push_p_numba( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using numba
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m

    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst )

    return ux, uy, uz, inv_gamma



@njit_parallel
def push_p_with_endpoint_numba(ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, q, m, i_start, N_batch, dt,
                ux_minus, uy_minus, uz_minus):
    """Push one bounded batch and retain its lower momentum endpoint.

    The endpoint arrays are indexed locally within the batch. This is the
    pusher/radiation coupling path: it avoids making three full-species
    copies while leaving the arithmetic of push_p_vay unchanged.
    """
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    for i_batch in prange(N_batch):
        ip = i_start + i_batch
        ux_i = ux[ip]
        uy_i = uy[ip]
        uz_i = uz[ip]
        ux_minus[i_batch] = ux_i
        uy_minus[i_batch] = uy_i
        uz_minus[i_batch] = uz_i
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux_i, uy_i, uz_i, inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst)


@njit_parallel
def push_p_after_plane_with_endpoint_numba(
                z, z_plane, ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, q, m, i_start, N_batch, dt,
                ux_minus, uy_minus, uz_minus):
    """Push a bounded ballistic-plane batch and retain lower endpoints."""
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m
    for i_batch in prange(N_batch):
        ip = i_start + i_batch
        ux_i = ux[ip]
        uy_i = uy[ip]
        uz_i = uz[ip]
        ux_minus[i_batch] = ux_i
        uy_minus[i_batch] = uy_i
        uz_minus[i_batch] = uz_i
        if z[ip] > z_plane:
            ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
                ux_i, uy_i, uz_i, inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
                econst, bconst)


@njit_parallel
def push_p_ioniz_with_endpoint_numba(ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, m, i_start, N_batch, dt,
                ionization_level, ux_minus, uy_minus, uz_minus):
    """Push a bounded variable-charge batch and retain lower endpoints."""
    prefactor_econst = e*dt/(m*c)
    prefactor_bconst = 0.5*e*dt/m
    for i_batch in prange(N_batch):
        ip = i_start + i_batch
        ux_i = ux[ip]
        uy_i = uy[ip]
        uz_i = uz[ip]
        ux_minus[i_batch] = ux_i
        uy_minus[i_batch] = uy_i
        uz_minus[i_batch] = uz_i
        if ionization_level[ip] == 0:
            continue
        econst = prefactor_econst * ionization_level[ip]
        bconst = prefactor_bconst * ionization_level[ip]
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux_i, uy_i, uz_i, inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
            econst, bconst)

@njit_parallel
def push_p_after_plane_numba( z, z_plane, ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, q, m, Ntot, dt ) :
    """
    Advance the particles' momenta, using numba.
    Only the particles that are located beyond the plane z=z_plane
    have their momentum modified ; the others particles move ballistically.
    """
    # Set a few constants
    econst = q*dt/(m*c)
    bconst = 0.5*q*dt/m

    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :
        if z[ip] > z_plane:
            ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
                ux[ip], uy[ip], uz[ip], inv_gamma[ip],
                Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip], econst, bconst)


@njit_parallel
def push_p_ioniz_numba( ux, uy, uz, inv_gamma,
                Ex, Ey, Ez, Bx, By, Bz, m, Ntot, dt, ionization_level ) :
    """
    Advance the particles' momenta, using numba
    """
    # Set a few constants
    prefactor_econst = e*dt/(m*c)
    prefactor_bconst = 0.5*e*dt/m

    # Loop over the particles (in parallel if threading is installed)
    for ip in prange(Ntot) :

        # For neutral macroparticles, skip this step
        if ionization_level[ip] == 0:
            continue

        # Calculate the charge dependent constants
        econst = prefactor_econst * ionization_level[ip]
        bconst = prefactor_bconst * ionization_level[ip]
        # Perform the push
        ux[ip], uy[ip], uz[ip], inv_gamma[ip] = push_p_vay(
            ux[ip], uy[ip], uz[ip], inv_gamma[ip],
            Ex[ip], Ey[ip], Ez[ip], Bx[ip], By[ip], Bz[ip],
            econst, bconst )

    return ux, uy, uz, inv_gamma

"""
This file is part of the Fourier-Bessel Particle-In-Cell code (FB-PIC)
It defines the structure and methods associated with the fields.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, mu_0, epsilon_0
import pyfftw
from fbpic.hankel_dt import DHT

# If numbapro is installed, it potentially allows to use the GPU
try :
    from cuda_methods import *
except :
    cuda_installed = False
else :
    cuda_installed = True
    # Define number of threads per block as global variables
    tpb = 32

class Fields(object) :
    """
    Class that contains the fields data of the simulation

    Methods
    -------
    - push : Advances the fields over one timestep
    - interp2spect : Transforms the fields from the
           interpolation grid to the spectral grid
    - spect2interp : Transforms the fields from the
           spectral grid to the interpolation grid
    - correct_currents : Corrects the currents so that
           they satisfy the conservation equation
    - erase : sets the fields to zero on the interpolation grid
    - divide_by_volume : divide the fields by the cell volume
    
    Main attributes
    ----------
    All the following attributes are lists,
    with one element per azimuthal mode
    - interp : a list of InterpolationGrid objects
        Contains the field data on the interpolation grid
    - spect : a list of SpectralGrid objects
        Contains the field data on the spectral grid
    - trans : a list of SpectralTransformer objects
        Allows to transform back and forth between the
        interpolation and spectral grid
    - psatd : a list of PsatdCoeffs
        Contains the coefficients to solve the Maxwell equations
    """

    def __init__( self, Nz, zmax, Nr, rmax, Nm, dt, use_cuda=False ) :
        """
        Initialize the components of the Fields object

        Parameters
        ----------
        Nz : int
            The number of gridpoints in z

        zmax : float
            The size of the box along z
            
        Nr : int
            The number of gridpoints in r

        rmax : float
            The size of the box along r

        Nm : int
            The number of azimuthal modes

        dt : float
            The timestep of the simulation, required for the
            coefficients of the psatd scheme

        use_cuda : bool, optional
            Wether to use the GPU or not
        """

        # Convert Nz to the nearest odd integer
        # (easier for the interpretation of the FFT)
        Nz = 2*int(Nz/2) + 1
        
        # Register the arguments inside the object
        self.Nz = Nz
        self.Nr = Nr
        self.rmax = rmax
        self.Nm = Nm
        self.dt = dt

        # Define wether or not to use the GPU
        self.use_cuda = use_cuda
        if (self.use_cuda==True) and (cuda_installed==False) :
            print 'Cuda for numba is not installed ; running on the CPU.'
            self.use_cuda = False
        if self.use_cuda == True:
            print 'Using the GPU for the fields.'

        # Infer the values of the z and kz grid
        dz = zmax/Nz
        z = dz * ( np.arange( 0, Nz ) + 0.5 )
        kz = 2*np.pi* np.fft.fftfreq( Nz, dz ) 
        # (According to FFT conventions, the kz array starts with
        # positive frequencies and ends with negative frequency.)
        
        # Create the list of the transformers, which convert the fields
        # back and forth between the spatial and spectral grid
        # (one object per azimuthal mode)
        self.trans = []
        for m in range(Nm) :
            self.trans.append( SpectralTransformer(Nz, Nr, m, rmax,
                                        use_cuda=self.use_cuda ) )

        # Create the interpolation grid for each modes
        # (one grid per azimuthal mode)
        self.interp = [ ]
        for m in range(Nm) :
            # Extract the radial grid for mode m
            r = self.trans[m].dht0.get_r()
            # Create the object
            self.interp.append( InterpolationGrid( z, r, m,
                                        use_cuda=self.use_cuda ) )

        # Create the spectral grid for each mode, as well as
        # the psatd coefficients
        # (one grid per azimuthal mode)
        self.spect = [ ]
        self.psatd = [ ]
        for m in range(Nm) :
            # Extract the inhomogeneous spectral grid for mode m
            kr = 2*np.pi * self.trans[m].dht0.get_nu()
            # Create the object
            self.spect.append( SpectralGrid( kz, kr, m,
                                        use_cuda=self.use_cuda ) )
            self.psatd.append( PsatdCoeffs( self.spect[m].kz,
                                self.spect[m].kr, m, dt, Nz, Nr,
                                use_cuda = self.use_cuda ) )

    def push(self, ptcl_feedback=True) :
        """
        Push the different azimuthal modes over one timestep,
        in spectral space.

        ptcl_feedback : bool, optional
            Whether to use the particles' densities and currents
            when pushing the fields
        """
        # Push each azimuthal grid individually, by passing the
        # corresponding psatd coefficients
        for m in range(self.Nm) :
            self.spect[m].push_eb_with( self.psatd[m], ptcl_feedback )
            self.spect[m].push_rho()

    def correct_currents(self) :
        """
        Correct the currents so that they satisfy the
        charge conservation equation
        """
        # Correct each azimuthal grid individually
        for m in range(self.Nm) :
            self.spect[m].correct_currents( self.dt )

    def interp2spect(self, fieldtype) :
        """
        Transform the fields `fieldtype` from the interpolation
        grid to the spectral grid

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to transform
            (either 'E', 'B', 'J', 'rho_next', 'rho_prev')
        """
        # Use the appropriate transformation depending on the fieldtype.
        if fieldtype == 'E' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].Ez, self.spect[m].Ez )
                self.trans[m].interp2spect_vect(
                    self.interp[m].Er, self.interp[m].Et,
                    self.spect[m].Ep, self.spect[m].Em )
        elif fieldtype == 'B' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].Bz, self.spect[m].Bz )
                self.trans[m].interp2spect_vect(
                    self.interp[m].Br, self.interp[m].Bt,
                    self.spect[m].Bp, self.spect[m].Bm )
        elif fieldtype == 'J' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].Jz, self.spect[m].Jz )
                self.trans[m].interp2spect_vect(
                    self.interp[m].Jr, self.interp[m].Jt,
                    self.spect[m].Jp, self.spect[m].Jm )
        elif fieldtype == 'rho_next' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].rho, self.spect[m].rho_next )
        elif fieldtype == 'rho_prev' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].interp2spect_scal(
                    self.interp[m].rho, self.spect[m].rho_prev )
        else :
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )

    def spect2interp(self, fieldtype) :
        """
        Transform the fields `fieldtype` from the spectral grid
        to the spectral grid

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to transform
            (either 'E', 'B', 'J', 'rho')
        """
        # Use the appropriate transformation depending on the fieldtype.
        if fieldtype == 'E' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].Ez, self.interp[m].Ez )
                self.trans[m].spect2interp_vect(
                    self.spect[m].Ep,  self.spect[m].Em,
                    self.interp[m].Er, self.interp[m].Et )
        elif fieldtype == 'B' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].Bz, self.interp[m].Bz )
                self.trans[m].spect2interp_vect(
                    self.spect[m].Bp, self.spect[m].Bm,
                    self.interp[m].Br, self.interp[m].Bt )
        elif fieldtype == 'J' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].Jz, self.interp[m].Jz )
                self.trans[m].spect2interp_vect(
                    self.spect[m].Jp,  self.spect[m].Jm,
                    self.interp[m].Jr, self.interp[m].Jt )
        elif fieldtype == 'rho' :
            # Transform each azimuthal grid individually
            for m in range(self.Nm) :
                self.trans[m].spect2interp_scal(
                    self.spect[m].rho_next, self.interp[m].rho )
        else :
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )

    def erase(self, fieldtype ) :
        """
        Sets the field `fieldtype` to zero on the interpolation grid

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be erased
            (either 'E', 'B', 'J', 'rho')
        """
        if self.use_cuda :
            # Erase the arrays on the GPU
            bpgz = int( self.Nz/tpz + 1 )
            bpgr = int( self.Nr/tpr + 1 )
            
            if fieldtype == 'rho' :
                cuda_erase_scalar[(bpgz, tpb), (bpgr, tbp)](
                    self.interp[0].rho, self.interp[1].rho, self.Nz, self.Nr )
            elif fieldtype == 'J' :
                cuda_erase_vector[(bpgz, bpgr), (tpb, tbp)](
                    self.interp[0].Jr, self.interp[1].Jr,
                    self.interp[0].Jt, self.interp[1].Jt,
                    self.interp[0].Jz, self.interp[1].Jz, self.Nz, self.Nr )
            elif fieldtype == 'E' :
                cuda_erase_vector[(bpgz, bpgr), (tpb, tbp)](
                    self.interp[0].Er, self.interp[1].Er,
                    self.interp[0].Et, self.interp[1].Et,
                    self.interp[0].Ez, self.interp[1].Ez, self.Nz, self.Nr )
            elif fieldtype == 'B' :
                cuda_erase_vector[(bpgz, bpgr), (tpb, tbp)](
                    self.interp[0].Br, self.interp[1].Br,
                    self.interp[0].Bt, self.interp[1].Bt,
                    self.interp[0].Bz, self.interp[1].Bz, self.Nz, self.Nr )
            else :
                raise ValueError('Invalid string for fieldtype: %s' %fieldtype)
        else :
            # Erase the arrays on the CPU
            if fieldtype == 'rho' :
                for m in range(self.Nm) :
                    self.interp[m].rho[:,:] = 0.
            elif fieldtype == 'J' :
                for m in range(self.Nm) :
                    self.interp[m].Jr[:,:] = 0.
                    self.interp[m].Jt[:,:] = 0.
                    self.interp[m].Jz[:,:] = 0.
            elif fieldtype == 'E' :
                for m in range(self.Nm) :
                    self.interp[m].Er[:,:] = 0.
                    self.interp[m].Et[:,:] = 0.
                    self.interp[m].Ez[:,:] = 0.
            elif fieldtype == 'B' :
                for m in range(self.Nm) :
                    self.interp[m].Br[:,:] = 0.
                    self.interp[m].Bt[:,:] = 0.
                    self.interp[m].Bz[:,:] = 0.
            else :
                raise ValueError('Invalid string for fieldtype: %s' %fieldtype)

    def filter_interp( self, fieldtype, direction='r' ) :
        """
        Filter the field `fieldtype` on the interpolation grid

        This uses a binomial filter

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be filtered
            (either 'E', 'B', 'J' or 'rho')

        direction : string, optional
           The direction in which to filter
           (either 'r' or 'z')
        """
        # Filter fields on the GPU
        for m in range(self.Nm) :
            self.interp[m].filter( fieldtype, direction )

    def filter_spect( self, fieldtype ) :
        """
        Filter the field `fieldtype` on the spectral grid

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be filtered
            (either 'E', 'B', 'J', 'rho_next' or 'rho_prev')
        """
        for m in range(self.Nm) :
            self.spect[m].filter( fieldtype )

    def divide_by_volume( self, fieldtype ) :
        """
        Divide the field `fieldtype` in each cell by the cell volume,
        on the interpolation grid.

        This is typically done for rho and J, after the charge and
        current deposition.

        Parameter
        ---------
        fieldtype :
            A string which represents the kind of field to be erased
            (either 'rho' or 'J')
        """
        if self.use_cuda :
            # Perform division on the GPU
            bpgz = int( self.Nz/tpz + 1 )
            bpgr = int( self.Nr/tpr + 1 )

            if fieldtype == 'rho' :
                cuda_divide_scalar_by_volume[(bpgz, bpgr),(tpb, tbp)](
                    self.interp[0].rho, self.interp[1].rho,
                    self.interp[0].invvol, self.interp[1].invvol,
                    self.Nz, self.Nr )
            elif fieldtype == 'J' :
                cuda_divide_vector_by_volume[(bpgz, bpgr),(tpb, tbp)](
                    self.interp[0].Jr, self.interp[1].Jr,
                    self.interp[0].Jt, self.interp[1].Jt,
                    self.interp[0].Jz, self.interp[1].Jz,
                    self.interp[0].invvol, self.interp[1].invvol,
                    self.Nz, self.Nr )
            else :
                raise ValueError('Invalid string for fieldtype: %s' %fieldtype)
        else :
            # Perform division on the CPU
            if fieldtype == 'rho' :
                for m in range(self.Nm) :
                    self.interp[m].rho = \
                    self.interp[m].rho * self.interp[m].invvol[np.newaxis,:]
            elif fieldtype == 'J' :
                for m in range(self.Nm) :
                    self.interp[m].Jr = \
                    self.interp[m].Jr * self.interp[m].invvol[np.newaxis,:]
                    self.interp[m].Jt = \
                    self.interp[m].Jt * self.interp[m].invvol[np.newaxis,:]
                    self.interp[m].Jz = \
                    self.interp[m].Jz * self.interp[m].invvol[np.newaxis,:]
            else :
                raise ValueError('Invalid string for fieldtype: %s' %fieldtype)


class InterpolationGrid(object) :
    """
    Contains the fields and coordinates of the spatial grid.

    Main attributes :
    - z,r : 1darrays containing the positions of the grid
    - Er, Et, Ez, Br, Bt, Bz, Jr, Jt, Jz, rho :
      2darrays containing the fields.
    """

    def __init__(self, z, r, m, use_cuda=False ) :
        """
        Allocates the matrices corresponding to the spatial grid
        
        Parameters
        ----------
        z : 1darray of float
            The positions of the longitudinal, spatial grid
        
        r : 1darray of float
            The positions of the radial, spatial grid

        m : int
            The index of the mode

        use_cuda : bool, optional
            Wether to use the GPU or not
        """

        # Register the arrays and their length
        Nz = len(z)
        Nr = len(r)
        self.Nz = Nz
        self.z = z.copy()
        self.Nr = Nr
        self.r = r.copy()
        self.m = m

        # Check whether the GPU should be used
        self.use_cuda = use_cuda
        
        # Register a few grid properties
        dr = r[1] - r[0]
        dz = z[1] - z[0]
        self.dr = dr
        self.dz = dz
        self.invdr = 1./dr
        self.invdz = 1./dz
        self.rmin = self.r.min()
        self.rmax = self.r.max()
        self.zmin = self.z.min()
        self.zmax = self.z.max()
        # Cell volume (assuming an evenly-spaced grid)
        vol = np.pi*dz*( (r+0.5*dr)**2 - (r-0.5*dr)**2 )
        # NB : No Verboncoeur-type correction required
        self.invvol = 1./vol
        
        # Allocate the fields arrays
        self.Er = np.zeros( (Nz, Nr), dtype='complex' )
        self.Et = np.zeros( (Nz, Nr), dtype='complex' )
        self.Ez = np.zeros( (Nz, Nr), dtype='complex' )
        self.Br = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bt = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bz = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jr = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jt = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jz = np.zeros( (Nz, Nr), dtype='complex' )
        self.rho = np.zeros( (Nz, Nr), dtype='complex' )
        
        
    def filter(self, fieldtype, direction ) :
        """
        Filter the field `fieldtype`

        This uses a binomial filter

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be filtered
            (either 'E', 'B', 'J' or 'rho')

        direction : string, optional
           The direction in which to filter
           (either 'r' or 'z')
        """
        if fieldtype == 'rho' :
            binomial_filter( self.rho, direction, (-1)**self.m )
        elif fieldtype == 'J' :
            binomial_filter( self.Jr, direction, -(-1)**self.m )
            binomial_filter( self.Jt, direction, -(-1)**self.m )
            binomial_filter( self.Jz, direction, (-1)**self.m )
        elif fieldtype == 'E' :
            binomial_filter( self.Er, direction, -(-1)**self.m )
            binomial_filter( self.Et, direction, -(-1)**self.m )
            binomial_filter( self.Ez, direction, (-1)**self.m )
        elif fieldtype == 'B' :
            binomial_filter( self.Br, direction, -(-1)**self.m )
            binomial_filter( self.Bt, direction, -(-1)**self.m )
            binomial_filter( self.Bz, direction, (-1)**self.m )
        else :
            raise ValueError( 'Invalid string for fieldtype: %s' %fieldtype )
        
    def show(self, fieldtype, below_axis=True, scale=1,
             gridscale=1.e-6, **kw) :
        """
        Show the field `fieldtype` on the interpolation grid

        Parameters
        ----------
        fieldtype : string
            Name of the field to be plotted.
            (either 'Er', 'Et', 'Ez', 'Br', 'Bt', 'Bz',
            'Jr', 'Jt', 'Jz', 'rho')

        scale : float, optional
            Value by which the field should be divided before plotting

        gridscale : float, optional
            Value by which to scale the z and r axis
            Default : scale it in microns
            
        kw : dictionary
            Options to be passed to matplotlib's imshow
        """
        # Select the field to plot
        plotted_field = getattr( self, fieldtype)
        # Show the field also below the axis for a more realistic picture
        if below_axis == True :
            plotted_field = np.hstack( (plotted_field[:,::-1],plotted_field) )
            extent = np.array([ self.zmin-0.5*self.dz, self.zmax+0.5*self.dz,
                      -self.rmax - 0.5*self.dr, self.rmax + 0.5*self.dr ])
        else :
            extent = np.array([self.zmin-0.5*self.dz, self.zmax+0.5*self.dz,
                      self.rmin - 0.5*self.dr, self.rmax + 0.5*self.dr])
        extent = extent/gridscale
        # Title
        plt.suptitle(
            '%s on the interpolation grid, for mode %d' %(fieldtype, self.m) )
            
        # Plot the real part
        plt.subplot(211)
        plt.imshow( plotted_field.real.T[::-1]/scale, aspect='auto',
                    interpolation='nearest', extent = extent, **kw )
        plt.xlabel('z')
        plt.ylabel('r')
        cb = plt.colorbar()
        cb.set_label('Real part')

        # Plot the imaginary part
        plt.subplot(212)
        plt.imshow( plotted_field.imag.T[::-1]/scale, aspect='auto',
                    interpolation='nearest', extent = extent, **kw )
        plt.xlabel('z')
        plt.ylabel('r')
        cb = plt.colorbar()
        cb.set_label('Imaginary part')
        
class SpectralGrid(object) :
    """
    Contains the fields and coordinates of the spectral grid.
    """

    def __init__(self, kz, kr, m, use_cuda=False ) :
        """
        Allocates the matrices corresponding to the spectral grid
        
        Parameters
        ----------
        kz : 1darray of float
            The wavevectors of the longitudinal, spectral grid
        
        kr : 1darray of float
            The wavevectors of the radial, spectral grid

        m : int
            The index of the mode

        use_cuda : bool, optional
            Wether to use the GPU or not
        """
        # Register the arrays and their length
        Nz = len(kz)
        Nr = len(kr)
        self.Nr = Nr
        self.Nz = Nz
        self.m = m
        self.kz, self.kr = np.meshgrid( kz, kr, indexing='ij' )

        # Check whether to use the GPU
        self.use_cuda = use_cuda
        
        # Allocate the fields arrays
        self.Ep = np.zeros( (Nz, Nr), dtype='complex' )
        self.Em = np.zeros( (Nz, Nr), dtype='complex' )
        self.Ez = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bp = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bm = np.zeros( (Nz, Nr), dtype='complex' )
        self.Bz = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jp = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jm = np.zeros( (Nz, Nr), dtype='complex' )
        self.Jz = np.zeros( (Nz, Nr), dtype='complex' )
        self.rho_prev = np.zeros( (Nz, Nr), dtype='complex' )
        self.rho_next = np.zeros( (Nz, Nr), dtype='complex' )

        # Auxiliary arrays
        # - for current correction
        self.F = np.zeros( (Nz, Nr), dtype='complex' )
        # Note : F is not needed on the GPU (on-the-fly variable)
        self.inv_k2 = 1./np.where( ( self.kz == 0 ) & (self.kr == 0),
                                   1., self.kz**2 + self.kr**2 )
        # No correction for k=0
        self.inv_k2[ ( self.kz == 0 ) & (self.kr == 0) ] = 0.
        # - for filtering
        self.filter_array = get_filter_array( kz, kr )

    def correct_currents (self, dt) :
        """
        Correct the currents so that they satisfy the
        charge conservation equation

        Parameters
        ----------
        dt : float
            Timestep of the simulation
        """
        # Precalculate useful coefficient
        inv_dt = 1./dt
        
        if self.use_cuda :
            # Correct the currents on the GPU
            bpgz = int( self.Nz/tpz + 1 )
            bpgr = int( self.Nr/tpr + 1 )
                        
            cuda_correct_currents[(bpgz, bpgr), (tpb, tbp)](
                self.rho_prev, self.rho_next, self.Jp, self.Jm, self.Jz,
                self.kz, self.kr, self.inv_k2, inv_dt, self.Nz, self.Nr)
        else :
            # Correct the currents on the CPU

            # Calculate the intermediate variable F
            self.F[:,:] = - self.inv_k2 * (
                (self.rho_next - self.rho_prev)*inv_dt \
                + 1.j*self.kz*self.Jz + self.kr*( self.Jp - self.Jm ) ) 
            
            # Correct the current accordingly
            self.Jp += 0.5*self.kr*self.F
            self.Jm += -0.5*self.kr*self.F
            self.Jz += -1.j*self.kz*self.F

    def push_eb_with(self, ps, ptcl_feedback=True, use_true_rho=False ) :
        """
        Push the fields over one timestep, using the psatd coefficients.

        Parameters
        ----------
        ps : PsatdCoeffs object
            psatd object corresponding to the same m mode

        ptcl_feedback : bool, optional
            Whether to take into the densities and currents when
            pushing the fields

        use_true_rho : bool, optional
            Whether to use the rho projected on the grid.
            If set to False, this will use div(E) and div(J)
            to evaluate rho and its time evolution.
            In the case use_true_rho==False, the rho projected
            on the grid is used only to correct the currents, and
            the simulation can be run without the neutralizing ions.
        """
        # Check that psatd object passed as argument is the right one
        # (i.e. corresponds to the right mode)
        assert( self.m == ps.m )

        if self.use_cuda :
            # Push the fields on the GPU
            bpgz = int( self.Nz/tpz + 1 )
            bpgr = int( self.Nr/tpr + 1 )
            
            cuda_push_eb_with[(bpgz, tpb), (bpgr, tbp)](
                self.Ep, self.Em, self.Ez, self.Bp, self.Bm, self.Bz,
                self.Jp, self.Jm, self.Jz, self.rho_prev, self.rho_next,
                ps.rho_prev_coef, ps.rho_next_coef, ps.j_coef, ps.dt,
                ptcl_feedback, use_true_rho, Nz, Nr )

        else :
            # Push the fields on the CPU
            
            # Define a few constants
            i = 1.j
            c2 = c**2

            # Save the electric fields, since it is needed for the B push
            ps.Ep[:,:] = self.Ep[:,:]
            ps.Em[:,:] = self.Em[:,:]
            ps.Ez[:,:] = self.Ez[:,:]

            # With particle feedback
            if ptcl_feedback :

                # Calculate useful auxiliary arrays
                if use_true_rho :
                    # Evaluation using the rho projected on the grid
                    rho_diff = ps.rho_next_coef*self.rho_next \
                        - ps.rho_prev_coef*self.rho_prev
                else :
                    # Evaluation using div(E) and div(J)
                    rho_diff = (ps.rho_next_coef-ps.rho_prev_coef)*epsilon_0* \
                    ( self.kr*self.Ep - self.kr*self.Em + i*self.kz*self.Ez ) \
                    - ps.rho_next_coef * ps.dt * \
                    ( self.kr*self.Jp - self.kr*self.Jm + i*self.kz*self.Jz )

                # Push the E field
                self.Ep[:,:] = ps.C*self.Ep + 0.5*self.kr*rho_diff \
                    + c2*ps.S_w*( -i*0.5*self.kr*self.Bz + self.kz*self.Bp \
                              - mu_0*self.Jp )

                self.Em[:,:] = ps.C*self.Em - 0.5*self.kr*rho_diff \
                    + c2*ps.S_w*( -i*0.5*self.kr*self.Bz - self.kz*self.Bm \
                              - mu_0*self.Jm )

                self.Ez[:,:] = ps.C*self.Ez - i*self.kz*rho_diff \
                    + c2*ps.S_w*( i*self.kr*self.Bp + i*self.kr*self.Bm \
                      - mu_0*self.Jz )

                # Push the B field
                self.Bp[:,:] = ps.C*self.Bp \
                    - ps.S_w*( -i*0.5*self.kr*ps.Ez + self.kz*ps.Ep ) \
                    + ps.j_coef*( -i*0.5*self.kr*self.Jz + self.kz*self.Jp )

                self.Bm[:,:] = ps.C*self.Bm \
                    - ps.S_w*( -i*0.5*self.kr*ps.Ez - self.kz*ps.Em ) \
                    + ps.j_coef*( -i*0.5*self.kr*self.Jz - self.kz*self.Jm )

                self.Bz[:,:] = ps.C*self.Bz \
                    - ps.S_w*( i*self.kr*ps.Ep + i*self.kr*ps.Em ) \
                    + ps.j_coef*( i*self.kr*self.Jp + i*self.kr*self.Jm )

            # Without particle feedback
            else :

                # Push the E field
                self.Ep[:,:] = ps.C*self.Ep \
                + c2*ps.S_w*( -i*0.5*self.kr*self.Bz + self.kz*self.Bp )
        
                self.Em[:,:] = ps.C*self.Em \
                + c2*ps.S_w*( -i*0.5*self.kr*self.Bz - self.kz*self.Bm )
    
                self.Ez[:,:] = ps.C*self.Ez \
                + c2*ps.S_w*( i*self.kr*self.Bp + i*self.kr*self.Bm )            
        
                # Push the B field
                self.Bp[:,:] = ps.C*self.Bp \
                    - ps.S_w*( -i*0.5*self.kr*ps.Ez + self.kz*ps.Ep ) 
    
                self.Bm[:,:] = ps.C*self.Bm \
                    - ps.S_w*( -i*0.5*self.kr*ps.Ez - self.kz*ps.Em ) 

                self.Bz[:,:] = ps.C*self.Bz \
                    - ps.S_w*( i*self.kr*ps.Ep + i*self.kr*ps.Em )

    def push_rho(self) :
        """
        Transfer the values of rho_next to rho_prev,
        and set rho_next to zero
        """
        if self.use_cuda :
            # Push the fields on the GPU
            bpgz = int( self.Nz/tpz + 1 )
            bpgr = int( self.Nr/tpr + 1 )

            cuda_push_rho[(bpgz, bpgr), (tpbz, tpbr)](
                self.rho_prev, self.rho_next )
        else :
            # Push the fields on the CPU
            self.rho_prev[:,:] = self.rho_next[:,:]
            self.rho_next[:,:] = 0.

    def filter(self, fieldtype) :
        """
        Filter the field `fieldtype`

        Parameter
        ---------
        fieldtype : string
            A string which represents the kind of field to be filtered
            (either 'E', 'B', 'J', 'rho_next' or 'rho_prev')
        """
        if self.use_cuda :
            # Filter fields on the GPU
            bpgz = int( self.Nz/tpz + 1 )
            bpgr = int( self.Nr/tpr + 1 )
            
            if fieldtype == 'rho_prev' :
                cuda_filter_scalar[(bpgz, bpgr), (tpbz, tpbr)](
                    self.rho_prev, self.filter_array, self.Nz, self.Nr )
            elif fieldtype == 'rho_next' :
                cuda_filter_scalar[(bpgz, bpgr), (tpbz, tpbr)](
                    self.rho_next, self.filter_array, self.Nz, self.Nr )
            elif fieldtype == 'J' :
                cuda_filter_vector[(bpgz, bpgr), (tpbz, tpbr)](
                self.Jp, self.Jm, self.Jz, self.filter_array, self.Nz, self.Nr)
            elif fieldtype == 'E' :
                cuda_filter_vector[(bpgz, bpgr), (tpbz, tpbr)](
                self.Ep, self.Em, self.Ez, self.filter_array, self.Nz, self.Nr)
            elif fieldtype == 'B' :
                cuda_filter_vector[(bpgz, bpgr), (tpbz, tpbr)](
                self.Bp, self.Bm, self.Bz, self.filter_array, self.Nz, self.Nr)
            else :
                raise ValueError('Invalid string for fieldtype: %s' %fieldtype)
        else :
            # Filter fields on the CPU
            
            if fieldtype == 'rho_prev' :
                self.rho_prev = self.rho_prev * self.filter_array
            elif fieldtype == 'rho_next' :
                self.rho_next = self.rho_next * self.filter_array
            elif fieldtype == 'J' :
                self.Jp = self.Jp * self.filter_array
                self.Jm = self.Jm * self.filter_array
                self.Jz = self.Jz * self.filter_array
            elif fieldtype == 'E' :
                self.Ep = self.Ep * self.filter_array
                self.Em = self.Em * self.filter_array
                self.Ez = self.Ez * self.filter_array
            elif fieldtype == 'B' :
                self.Bp = self.Bp * self.filter_array
                self.Bm = self.Bm * self.filter_array
                self.Bz = self.Bz * self.filter_array
            else :
                raise ValueError('Invalid string for fieldtype: %s' %fieldtype)
        
    def show(self, fieldtype, below_axis=True, scale=1, **kw) :
        """
        Show the field `fieldtype` on the spectral grid

        Parameters
        ----------
        fieldtype : string
            Name of the field to be plotted.
            (either 'Ep', 'Em', 'Ez', 'Bp', 'Bm', 'Bz',
            'Jp', 'Jm', 'Jz', 'rho_prev', 'rho_next')

        scale : float
            Value by which the field should be divide before plotting
            
        kw : dictionary
            Options to be passed to matplotlib's imshow
        """
        # Select the field to plot
        plotted_field = getattr( self, fieldtype)
        # Fold it so as to center the 0 frequency
        plotted_field = np.fft.fftshift( plotted_field, axes=0 )
        if below_axis == True :
            plotted_field = np.hstack( (plotted_field[:,::-1], plotted_field) )
            extent = [ self.kz[:,0].min(), self.kz[:,0].max(),
                    -self.kr[0,:].max(), self.kr[0,:].max() ]
        else :
            extent = [ self.kz[:,0].min(), self.kz[:,0].max(),
                    self.kr[0,:].min(), self.kr[0,:].max() ]
        # Title
        plt.suptitle(
            '%s on the spectral grid, for mode %d' %(fieldtype, self.m) )
        
        # Plot the real part
        plt.subplot(211)
        plt.imshow( plotted_field.real.T[::-1]/scale, aspect='auto',
                    interpolation='nearest', extent = extent, **kw )
        plt.xlabel('kz')
        plt.ylabel('kr')
        cb = plt.colorbar()
        cb.set_label('Real part')
        
        # Plot the imaginary part
        plt.subplot(212)
        plt.imshow( plotted_field.imag.T[::-1]/scale, aspect='auto',
                    interpolation='nearest', extent = extent, **kw )
        plt.xlabel('kz')
        plt.ylabel('kr')
        cb = plt.colorbar()
        cb.set_label('Imaginary part')



class PsatdCoeffs(object) :
    """
    Contains the coefficients of the PSATD scheme for a given mode.
    """
    
    def __init__( self, kz, kr, m, dt, Nz, Nr, use_cuda=False ) :
        """
        Allocates the coefficients matrices for the psatd scheme.
        
        Parameters
        ----------
        kz : 2darray of float
            The positions of the longitudinal, spectral grid
        
        kr : 2darray of float
            The positions of the radial, spectral grid

        m : int
            Index of the mode
            
        dt : float
            The timestep of the simulation
        """
        
        # Register m and dt
        self.m = m
        self.dt = dt
    
        # Construct the omega and inverse omega array
        w = c*np.sqrt( kz**2 + kr**2 )
        inv_w = 1./np.where( w == 0, 1., w ) # Avoid division by 0 

        # Construct the C coefficient arrays
        self.C = np.cos( w*dt )
        
        # Construct the S/w coefficient arrays
        self.S_w = np.sin( w*dt )*inv_w
        # Enforce the right value for w==0
        self.S_w[ w==0 ] = dt
        
        # Construct the mu0 c2 (1-C)/w2 array
        self.j_coef =  mu_0*c**2*(1.-self.C)*inv_w**2
        # Enforce the right value for w==0
        self.j_coef[ w==0 ] = mu_0*c**2*(0.5*dt**2)

        # Construct rho_prev coefficient array
        inv_dt = 1./dt
        self.rho_prev_coef = c**2/epsilon_0*(self.C - inv_dt*self.S_w)*inv_w**2
        # Enforce the right value for w==0
        self.rho_prev_coef[ w==0 ] = c**2/epsilon_0*(-1./3*dt**2)

        # Construct rho_next coefficient array
        self.rho_next_coef = c**2/epsilon_0*(1 - inv_dt*self.S_w)*inv_w**2
        # Enforce the right value for w==0
        self.rho_next_coef[ w==0 ] = c**2/epsilon_0*(1./6*dt**2)
        
        # Allocate useful auxiliary matrices
        self.Ep = np.zeros( (Nz, Nr), dtype='complex' )
        self.Em = np.zeros( (Nz, Nr), dtype='complex' )
        self.Ez = np.zeros( (Nz, Nr), dtype='complex' )
        # Note : no need to copy these variables to the GPU
        # (on-the-fly variables)

class SpectralTransformer(object) :
    """
    Object that allows to transform the fields back and forth between the
    spectral and interpolation grid.

    Attributes :
    - dht : the discrete Hankel transform object that operates along r

    Main methods :
    - spect2interp_scal :
        converts a scalar field from the spectral to the interpolation grid
    - spect2interp_vect :
        converts a vector field from the spectral to the interpolation grid
    - interp2spect_scal :
        converts a scalar field from the interpolation to the spectral grid
    - interp2spect_vect :
        converts a vector field from the interpolation to the spectral grid
    """

    def __init__(self, Nz, Nr, m, rmax, nthreads=4, use_cuda=False ) :
        """
        Initializes the dht attributes, which contain auxiliary
        matrices allowing to transform the fields quickly

        Parameters
        ----------
        Nz, Nr : int
            Number of points along z and r respectively

        m : int
            Index of the mode (needed for the Hankel transform)

        rmax : float
            The size of the simulation box along r.

        nthreads : int, optional
            Number of threads for the FFTW transform
        """
        # Check whether to use the GPU
        self.use_cuda = use_cuda
        
        # Initialize the DHT (local implementation, see hankel_dt.py)
        print('Preparing the Discrete Hankel Transforms for mode %d' %m)
        self.dht0 = DHT(   m, Nr, Nz, rmax, 'MDHT(m,m)', d=0.5, Fw='inverse',
                           use_cuda=self.use_cuda )
        self.dhtp = DHT( m+1, Nr, Nz, rmax, 'MDHT(m+1,m)', d=0.5, Fw='inverse',
                           use_cuda=self.use_cuda )
        self.dhtm = DHT( m-1, Nr, Nz, rmax, 'MDHT(m-1,m)', d=0.5, Fw='inverse',
                           use_cuda=self.use_cuda )

        # Initialize the FFTW
        print('Preparing FFTW for mode %d' %m)
        # Two buffers and FFTW objects are initialized, since spect2interp_vect
        # and interp2spect_vect require two separate FFTs.
        # First buffer and FFTW transform
        self.interp_buffer_r = \
            pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
        self.spect_buffer_r = \
            pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
        self.spect_buffer_p = self.spect_buffer_r # Two names for the same array
        self.fft_r = pyfftw.FFTW( self.interp_buffer_r, self.spect_buffer_r,
                    axes=(0,), direction='FFTW_FORWARD', threads=nthreads )
        self.ifft_r = pyfftw.FFTW( self.spect_buffer_r, self.interp_buffer_r,
                    axes=(0,), direction='FFTW_BACKWARD', threads=nthreads )
        # Second buffer and FFTW transform
        self.interp_buffer_t = \
            pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
        self.spect_buffer_t = \
            pyfftw.n_byte_align_empty( (Nz,Nr), 16, 'complex128' )
        self.spect_buffer_m = self.spect_buffer_t # Two names for the same array
        self.fft_t = pyfftw.FFTW( self.interp_buffer_t, self.spect_buffer_t,
                    axes=(0,), direction='FFTW_FORWARD', threads=nthreads )
        self.ifft_t = pyfftw.FFTW( self.spect_buffer_t, self.interp_buffer_t,
                    axes=(0,), direction='FFTW_BACKWARD', threads=nthreads )
        
    def spect2interp_scal( self, spect_array, interp_array ) :
        """
        Convert a scalar field from the spectral grid
        to the interpolation grid.

        Parameters
        ----------
        spect_array : 2darray of complexs
           A complex array representing the fields in spectral space, from 
           which to compute the values of the interpolation grid
           The first axis should correspond to z and the second axis to r.

        interp_array : 2darray of complexs
           A complex array representing the fields on the interpolation
           grid, and which is overwritten by this function.
        """
        # Perform the inverse DHT first (along axis -1, which corresponds to r)
        self.dht0.inverse_transform( spect_array, self.spect_buffer_r )

        # Then perform the inverse FFT (along axis 0, which corresponds to z)
        self.interp_buffer_r = self.ifft_r()
        #Copy to the output array
        interp_array[:,:] = self.interp_buffer_r[:,:]  

    def spect2interp_vect( self, spect_array_p, spect_array_m,
                          interp_array_r, interp_array_t ) :
        """
        Convert a transverse vector field in the spectral space (e.g. Ep, Em)
        to the interpolation grid (e.g. Er, Et)

        Parameters
        ----------
        spect_array_p, spect_array_m : 2darray
           Complex arrays representing the fields in spectral space, from 
           which to compute the values of the interpolation grid
           The first axis should correspond to z and the second axis to r.

        interp_array_r, interp_array_t : 2darray
           Complex arrays representing the fields on the interpolation
           grid, and which are overwritten by this function.
        """
        # Perform the inverse DHT first (along axis -1, which corresponds to r)
        self.dhtp.inverse_transform( spect_array_p, self.spect_buffer_p )
        self.dhtm.inverse_transform( spect_array_m, self.spect_buffer_m )
    
        # Combine them to obtain the actual r and t components
        # (It is important to write the affectation in the following way, since
        # self.spect_buffer_p and self.spect_buffer_r actually point to the same
        # object, for memory economy)
        self.spect_buffer_p[:,:], self.spect_buffer_m[:,:] = \
            self.spect_buffer_r + self.spect_buffer_t, \
            1.j*( self.spect_buffer_r - self.spect_buffer_t)

        # Finally perform the FFT (along axis 0, which corresponds to z)
        self.interp_buffer_r = self.ifft_r()
        # Copy to the output array
        interp_array_r[:,:] = self.interp_buffer_r[:,:] 
        self.interp_buffer_t = self.ifft_t()
        # Copy to the output array
        interp_array_t[:,:] = self.interp_buffer_t[:,:] 

    def interp2spect_scal( self, interp_array, spect_array ) :
        """
        Convert a scalar field from the interpolation grid
        to the spectral grid.

        Parameters
        ----------
        interp_array : 2darray
           A complex array representing the fields on the interpolation
           grid, from which to compute the values of the interpolation grid
           The first axis should correspond to z and the second axis to r.
        
        spect_array : 2darray
           A complex array representing the fields in spectral space,
           and which is overwritten by this function.
        """
        # Perform the FFT first (along axis 0, which corresponds to z)
        self.interp_buffer_r[:,:] = interp_array #Copy the input array
        self.spect_buffer_r = self.fft_r()
        
        # Then perform the DHT (along axis -1, which corresponds to r)
        self.dht0.transform( self.spect_buffer_r, spect_array )

    def interp2spect_vect( self, interp_array_r, interp_array_t,
                           spect_array_p, spect_array_m ) :
        """
        Convert a transverse vector field from the interpolation grid
        (e.g. Er, Et) to the spectral space (e.g. Ep, Em)

        Parameters
        ----------
        interp_array_r, interp_array_t : 2darray
           Complex arrays representing the fields on the interpolation
           grid, from which to compute the values in spectral space
           The first axis should correspond to z and the second axis to r.
        
        spect_array_p, spect_array_m : 2darray
           Complex arrays representing the fields in spectral space,
           and which are overwritten by this function.
        """
        # Perform the FFT first (along axis 0, which corresponds to z)
        self.interp_buffer_r[:,:] = interp_array_r #Copy the input array
        self.spect_buffer_r = self.fft_r()
        self.interp_buffer_t[:,:] = interp_array_t #Copy the input array
        self.spect_buffer_t = self.fft_t()

        # Combine the r and t components to obtain the p and m components
        # (It is important to write the affectation in the following way, since
        # self.spect_buffer_p and self.spect_buffer_r actually point to the same
        # object, for memory economy.)
        self.spect_buffer_p[:,:], self.spect_buffer_m[:,:] = \
          0.5*( self.spect_buffer_r - 1.j*self.spect_buffer_t ), \
          0.5*( self.spect_buffer_r + 1.j*self.spect_buffer_t )
        
        # Perform the inverse DHT first (along axis -1, which corresponds to r)
        self.dhtp.transform( self.spect_buffer_p, spect_array_p )
        self.dhtm.transform( self.spect_buffer_m, spect_array_m )


# -----------------
# Utility function
# -----------------

def binomial_filter( F, direction, sign_guard ) :
    """
    Apply a binomial filter to the array F

    Parameters
    ----------
    F : 2darray
        An array whose first axis corresponds to z and
        second axis corresponds to r

    direction : string
        Indicates in which direction to perform the filter
        (Either 'r' or 'z')

    sign_guard : int
        Indicates with what is the sign of the values below
        the axis, as compared to the value above the axis.
        (Only used in the case direction='r')
    """
    F_unfiltered = F.copy()
    
    if direction == 'z' : # Periodic boundaries
        F[1:-1,:] = 0.25*F_unfiltered[:-2,:] \
                   + 0.5*F_unfiltered[1:-1,:] \
                   + 0.25*F_unfiltered[2:,:]
        F[0,:] =  0.25*F_unfiltered[-1,:] \
                   + 0.5*F_unfiltered[0,:] \
                   + 0.25*F_unfiltered[1,:]
        F[-1,:] =  0.25*F_unfiltered[-2,:] \
                   + 0.5*F_unfiltered[-1,:] \
                   + 0.25*F_unfiltered[0,:]

    elif direction == 'r' : # Non-periodic boundaries
        F[:,1:-1] = 0.25*F_unfiltered[:,:-2] \
                   + 0.5*F_unfiltered[:,1:-1] \
                   + 0.25*F_unfiltered[:,2:]
        # Assume that the guard cell below the axis has
        # the same value as the cell above the axis, up
        # to the sign `sign_guard`
        F[:,0] =  0.25*sign_guard*F_unfiltered[:,0] \
                 + 0.5*F_unfiltered[:,0] \
                 + 0.5*F_unfiltered[:,1]
        # Assume that the guard cell above the boundary has
        # the same value as the cell below it
        F[:,-1] =  0.25*F_unfiltered[:,-1] \
                   + 0.5*F_unfiltered[:,-1] \
                   + 0.25*F_unfiltered[:,-2]
    else :
        raise ValueError("Unrecognized `direction` : %s" %direction)


def get_filter_array( kz, kr ) :
    """
    Return the array that multiplies the fields in k space

    The filtering function is 1-sin( k/kmax * pi/2 )**2.
    (equivalent to a one-pass binomial filter in real space,
    for the longitudinal direction)

    Parameters
    ----------
    kz, kr : 1darrays
       The longitudinal and transverse wavevectors on the spectral grid

    Returns
    -------
    A 2darray of shape ( len(kz), len(kr) )
    """
    # Find the 1D filter in z
    coef_z = 1./kz.max() * np.pi/2
    filt_z = 1. - np.sin( kz * coef_z )**2

    # Find the 1D filter in r
    coef_r = 1./kr.max() * np.pi/2
    filt_r = 1. - np.sin( kr * coef_r )**2

    # Build the 2D filter by takin the product
    filter_array = filt_z[:, np.newaxis] * filt_r[np.newaxis, :]

    return( filter_array )
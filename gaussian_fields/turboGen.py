# TURBOGEN collection of method to generate fluctuation density field
# I can imagine to implement 1D, 2D and 3D turbulence generator method

"""
Author: Stefano Merlini
Created: 07/05/2020
"""

# function that computes the turbulence base don a given parameters:
#  - grid and domain definitions
#  - spectra

# -----------------------------------------------------------------------------------------
#  ____  _  _  ____  ____   __    ___  ____  __ _  ____  ____   __  ____  __  ____ 
# (_  _)/ )( \(  _ \(  _ \ /  \  / __)(  __)(  ( \(  __)(  _ \ / _\(_  _)/  \(  _ \
#   )(  ) \/ ( )   / ) _ ((  O )( (_ \ ) _) /    / ) _)  )   //    \ )( (  O ))   /
#  (__) \____/(__\_)(____/ \__/  \___/(____)\_)__)(____)(__\_)\_/\_/(__) \__/(__\_)
# -----------------------------------------------------------------------------------------
# 

import numpy as np

# ---------- 1D Gaussian Generator -----------------
# this method is from reference: 1988, Yamasaki, "Digital Generation of Non-Goussian Stochastic Fields"
# Additional reference: Shinozuka, M. and Deodatis, G. (1996) 

def gaussian1D(lx, nx, nmodes, wn1, especf):
    
    """
     Given a specific energy spectrum, this function generates
     1-D Gaussian field whose energy spectrum corresponds to the  
     the input energy spectrum.

     Parameters:
    -------------------
    lx: float
        the domain size in the x-direction.
    nx: integer
        the number of grid points in the x-direction
    nmodes: integer
        Number of modes
    wn1: float
        Smallest wavenumber. Typically dictated by spectrum or domain
    espec: function
        A callback function representing the energy spectrum in input

    EXAMPLE:
    # define spectrum
    
    class k41:
    def evaluate(self, k):
        espec = pow(k,-5.0/3.0)
        return espec

    # user input

    nx = 64
    lx = 1
    nmodes = 100
    inputspec = 'k41'
    whichspect = k41().evaluate
    wn1 = 2.0*np.pi/lx

    rx = tg.gaussian1D(lx, nx, nmodes, wn1, whichspect)

    dx = lx/nx
    X = np.arange(0, lx, dx)
    plt.plot(X,rx, 'k-', label='computed')    
    """

    dx = lx/nx
    # Compute the highest wavenumber (wavenumber cutoff)
    wnn = np.pi/dx
    print("This function will generate data up to wavenumber: ", wnn)
    # compute the infinitesiaml wavenumber (step dk)
    dk = (wnn-wn1)/nmodes
    # compute an array of equal-distance wavenumbers at the cells centers
    wn = wn1 + 0.5*dk +  np.arange(0,nmodes)*dk
    dkn = np.ones(nmodes)*dk
    # Calculating the proportional factor (using the input power spectrum)
    espec = especf(wn)
    espec = espec.clip(0.0)
    A_m = np.sqrt(2.0*espec*dkn) # for each mode I need a proportional factor ('colouring' the spectrum)
    # Generate Random phase angles from a normal distribution between 0 and 2pi
    phi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
    psi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)

    kx = wn

    # computing the center position of the cell

    xc = dx/2.0 + np.arange(0,nx)*dx

    _rx = np.zeros(nx)

    print("Generating 1-D turbulence...")

    for i in range(0,nx):
        # for every step i along x-direction do the fourier summation
        
        arg1 = kx*xc[i] + phi
        arg2 = kx*xc[i] + psi

        bmx = A_m * np.sqrt(2.0) *(np.cos(arg1 - kx*dx/2.0) + np.cos(arg2 - kx*dx/2.0))
        
        _rx[i] = np.sum(bmx)

    print("Done! 1-D Turbulence has been generated!")

    return _rx 

def gaussian1D_FFT(x,a):
    """A FFT based generator for scalar gaussian fields in 1D
    Reference:Timmer, J and König, M. “On Generating Power Law Noise.” Astronomy & Astrophysics 300 (1995):
     1–30. https://doi.org/10.1017/CBO9781107415324.004.

    Arguments:
        x {1D float array} -- uniformly spaced array of 1D grid of coordinates over which to generate turbulence
        a {float} -- power law index, a=-5/3 for Kolmogorov, S(k)=k**a

    Returns:
        x {1D array of floats} -- unchanged from input, for convinience (direct to plotting)
        signal {1D array of float} -- a realisation of a 1D gaussian process with power law k**a.

    Example:
        x = np.linspace(0,10,1001)
        x, sig = gaussian1D_FFT(x,a = -11/3)
        fig,ax=plt.subplots()
        ax.plot(x,sig)
    """

    k=np.fft.fftfreq(k.size, d=x[1]-x[0]) #these are the frequencies, starting from 0 up to f_max, then -f_max to 0.
    k_pos=k[:k.size//2] #take the positive wavenumbers only - our desired signal is real, so the FT is hermitian

    kr=np.random.randn(k_pos.size) # random number from Gaussian for both 
    ki=np.random.randn(k_pos.size) # real and imaginary components

    k=(kr+1j*ki)*k_pos**(a/2) # colour the initially white noise with the power spectrum
    k[0]=0 # set 0 wavenumber to 0, so mean (DC) value is 0.

    #our desired signal is real, so we can use a hermitian FFT to get the output.
    # we could also create the negative frequencies as the complex conjugate of the positive frequencies
    # and then use the full Fourier transform, but there's no need.
    signal=np.fft.hfft(k_total, n=x.size) # specify n to ensure correct size of output

    return x, signal
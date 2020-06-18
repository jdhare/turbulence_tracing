# TURBOGEN collection of method to generate fluctuation density field
# I can imagine to implement 1D, 2D and 3D turbulence generator method

"""
Author: Stefano Merlini
Created: 14/05/2020
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

#   __      ____     ___   __   _  _  ____  ____  __   __   __ _     ___  __   ____ 
#  /  \ ___(    \   / __) / _\ / )( \/ ___)/ ___)(  ) / _\ (  ( \   / __)/  \ / ___)
# (_/ /(___)) D (  ( (_ \/    \) \/ (\___ \\___ \ )( /    \/    /  ( (__(  O )\___ \
#  (__)    (____/   \___/\_/\_/\____/(____/(____/(__)\_/\_/\_)__)   \___)\__/ (____/

# this method is from reference: 1988, Yamasaki, "Digital Generation of Non-Goussian Stochastic Fields"
# Additional reference: Shinozuka, M. and Deodatis, G. (1996) 

def gaussian1Dcos(lx, nx, nmodes, wn1, especf):
    """
     Given a specific energy spectrum, this function generates
     1-D Gaussian field whose energy spectrum corresponds to the  
     the input energy spectrum.

     Parameters:
    -----------------------------------------------------------------
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
    -----------------------------------------------------------------
    
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
    # -----------------------------------------------------------------

    # cell size in X-direction
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
    _r = np.zeros(nx)
    print("Generating 1-D turbulence...")
    for i in range(0,nx):
        # for every step i along x-direction do the fourier summation
        arg1 = kx*xc[i] + phi
        bmx = A_m * np.sqrt(2.0) *(np.cos(arg1))
        _r[i] = np.sum(bmx)
    print("Done! 1-D Turbulence has been generated!")
    return _r







#  ____      ____     ___   __   _  _  ____  ____  __   __   __ _     ___  __   ____ 
# (___ \ ___(    \   / __) / _\ / )( \/ ___)/ ___)(  ) / _\ (  ( \   / __)/  \ / ___)
#  / __/(___)) D (  ( (_ \/    \) \/ (\___ \\___ \ )( /    \/    /  ( (__(  O )\___ \
# (____)    (____/   \___/\_/\_/\____/(____/(____/(__)\_/\_/\_)__)   \___)\__/ (____/

# this method is from reference: 1988, Yamasaki, "Digital Generation of Non-Goussian Stochastic Fields"
# Additional reference: Shinozuka, M. and Deodatis, G. (1996) 

def gaussian2Dcos(lx, ly, nx, ny, nmodes, wn1, especf):
    """
     Given a specific energy spectrum, this function generates
     2-D Gaussian field whose energy spectrum corresponds to the  
     the input energy spectrum.

     Parameters:
    ----------------------------------------------------------------
    lx: float
        the domain size in the x-direction.
    ly: float
        the domain size in the y-direction.
    nx: integer
        the number of grid points in the x-direction
    ny: integer
        the number of grid points in the y-direction
    nmodes: integer
        Number of modes
    wn1: float
        Smallest wavenumber. Typically dictated by spectrum or domain
    espec: function
        A callback function representing the energy spectrum in input
    -----------------------------------------------------------------

    EXAMPLE:
    import turboGen as tg
    import calcspec

    # define spectrum
    class k41:
    def evaluate(self, k):
        espec = pow(k,-5.0/3.0)
        return espec
    
    # user input
    
    nx = 64
    ny = 64
    lx = 1
    ly = 1
    nmodes = 100
    inputspec = 'k41'
    whichspect = k41().evaluate
    wn1 = min(2.0*np.pi/lx, 2.0*np.pi/ly)

    r = tg.gaussian1D(lx, ly, nx, ny, nmodes, wn1, whichspect)
    
    dx = lx/nx
    dy = ly/ny
    X = np.arange(0, lx, dx)
    Y = np.arange(0, ly, dy)
    X, Y = np.meshgrid(np.arange(0,lx,dx), np.arange(0,ly,dy))
    cp = plt.contourf(X, Y, r)
    cb = plt.colorbar(cp)

    # I you want to calculate the spectrum

    knyquist, wavenumbers, tkespec = calcspec.compute2Dspectum(r, lx, ly, False)

    """
    # --------------------------------------------------------------------------

    # cell size in X and Y directions
    dx = lx/nx
    dy = ly/ny
    # Compute the highest wavenumber (wavenumber cutoff)
    wnn = max(np.pi/dx,np.pi/dy)
    print("This function will generate data up to wavenumber: ", wnn)
    # compute the infinitesiaml wavenumber (step dk)
    dk = (wnn - wn1)/nmodes
    # compute an array of equal-distance wavenumbers at the cells centers
    wn = wn1 + 0.5*dk +  np.arange(0,nmodes)*dk
    dkn = np.ones(nmodes)*dk
    # Calculating the proportional factor (using the input power spectrum)
    espec = especf(wn)
    espec = espec.clip(0.0)
    A_m = np.sqrt(2.0*espec*(dkn)**2) # for each mode I need a proportional factor ('colouring' the spectrum)
    # Generate Random phase angles from a normal distribution between 0 and 2pi
    phi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
    psi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
    theta = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
    #
    kx = np.cos(theta)*wn
    ky = np.sin(theta)*wn
    # Computing the vector [xc,yc]
    xc = dx / 2.0 + np.arange(0, nx) * dx
    yc = dy / 2.0 + np.arange(0, ny) * dy
    # Looping through 2-Dimensions nx, ny and perfom the Fourier Summation

    # computing the center position of the cell
    xc = dx/2.0 + np.arange(0,nx)*dx
    yc = dy/2.0 + np.arange(0,ny)*dy

    _r = np.zeros((nx,ny))

    print("Generating 2-D turbulence...")
    for j in range(0,ny):
        for i in range(0,nx):
            # for every step i along x-y direction do the fourier summation
            arg1 = kx*xc[i] + ky*yc[j] + phi
            arg2 = kx*xc[i] - ky*yc[j] + psi
            bm = A_m * np.sqrt(2.0) *(np.cos(arg1) + np.cos(arg2))
            _r[i,j] = np.sum(bm)
    print("Done! 2-D Turbulence has been generated!")
    return _r









#  ____      ____     ___   __   _  _  ____  ____  __   __   __ _     ___  __   ____ 
# ( __ \ ___(    \   / __) / _\ / )( \/ ___)/ ___)(  ) / _\ (  ( \   / __)/  \ / ___)
#  (__ ((___)) D (  ( (_ \/    \) \/ (\___ \\___ \ )( /    \/    /  ( (__(  O )\___ \
# (____/    (____/   \___/\_/\_/\____/(____/(____/(__)\_/\_/\_)__)   \___)\__/ (____/

# this method is from reference: 1988, Yamasaki, "Digital Generation of Non-Goussian Stochastic Fields"
# Additional reference: Shinozuka, M. and Deodatis, G. (1996) 



def gaussian3Dcos(lx, ly, lz, nx, ny, nz, nmodes, wn1, especf):
    """
     Given a specific energy spectrum, this function generates
     3-D Gaussian field whose energy spectrum corresponds to the  
     the input energy spectrum.

     Parameters:
    ----------------------------------------------------------------
    lx: float
        the domain size in the x-direction.
    ly: float
        the domain size in the y-direction.
    lz: float
        the domain size in the z-direction.
    nx: integer
        the number of grid points in the x-direction
    ny: integer
        the number of grid points in the y-direction
    nz: integer
        the number of grid points in the z-direction
    nmodes: integer
        Number of modes
    wn1: float
        Smallest wavenumber. Typically dictated by spectrum or domain
    espec: function
        A callback function representing the energy spectrum in input
    -----------------------------------------------------------------

    EXAMPLE:
    import turboGen as tg
    import calcspec

    # define spectrum
    class k41:
    def evaluate(self, k):
        espec = pow(k,-5.0/3.0)
        return espec
    
    # user input
    
    nx = 64
    ny = 64
    nz = 64
    lx = 1
    ly = 1
    lz = 1

    nmodes = 100
    inputspec = 'k41'
    whichspect = k41().evaluate
    wn1 = min(2.0*np.pi/lx, 2.0*np.pi/ly, 2.0*np.pi/lz)

    r = tg.gaussian1D(lx, ly, lz, nx, ny, nz, nmodes, wn1, whichspect)
    
    dx = lx/nx
    dy = ly/ny
    dz = lz/nz
    X = np.arange(0, lx, dx)
    Y = np.arange(0, ly, dy)
    Z = np.arange(0, lz, dz)

    X, Y = np.meshgrid(np.arange(0,lx,dx), np.arange(0,ly,dy))
    cp = plt.contourf(X, Y, r(:,:,1))
    cb = plt.colorbar(cp)

    # I you want to calculate the spectrum

    knyquist, wavenumbers, tkespec = calcspec.compute3Dspectum(r, lx, ly, lz, False)

    """
    # --------------------------------------------------------------------------

    # cell size in X, Y, Z directions
    dx = lx/nx
    dy = ly/ny
    dz = lz/nz

    # Compute the highest wavenumber (wavenumber cutoff)
    wnn = max(np.pi/dx,np.pi/dy,np.pi/dz)
    print("This function will generate data up to wavenumber: ", wnn)
    # compute the infinitesiaml wavenumber (step dk)
    dk = (wnn - wn1)/nmodes
    # compute an array of equal-distance wavenumbers at the cells centers
    wn = wn1 + 0.5*dk +  np.arange(0,nmodes)*dk
    dkn = np.ones(nmodes)*dk
    # Calculating the proportional factor (using the input power spectrum)
    espec = especf(wn)
    espec = espec.clip(0.0)
    A_m = np.sqrt(2.0*espec*(dkn)**3) # for each mode I need a proportional factor ('colouring' the spectrum)
    # Generate Random phase angles from a normal distribution between 0 and 2pi
    
    psi_1 = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
    psi_2 = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
    psi_3 = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
    psi_4 = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)

    theta = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)
    phi = 2.0*np.pi*np.random.uniform(0.0,1.0,nmodes)

    #
    kx = np.sin(theta) * np.cos(phi) * wn;
    ky = np.sin(theta) * np.sin(phi) * wn;
    kz = np.cos(theta) * wn;

    # Computing the vector [xc,yc,zc]
    xc = dx / 2.0 + np.arange(0, nx) * dx
    yc = dy / 2.0 + np.arange(0, ny) * dy
    zc = dz / 2.0 + np.arange(0, nz) * dz

    # Looping through 3-Dimensions nx, ny, nz and perfom the Fourier Summation

    # computing the center position of the cell
    xc = dx/2.0 + np.arange(0,nx)*dx
    yc = dy/2.0 + np.arange(0,ny)*dy
    zc = dz/2.0 + np.arange(0,nz)*dz

    _r = np.zeros((nx,ny,nz))

    print("Generating 3-D turbulence...")
    for k in range(0,nz):
        for j in range(0,ny):
            for i in range(0,nx):
                # for every step i along x-y-z direction do the fourier summation
                arg1 = kx*xc[i] + ky*yc[j] + kz*zc[k] + psi_1
                arg2 = kx*xc[i] + ky*yc[j] - kz*zc[k] + psi_2
                arg3 = kx*xc[i] - ky*yc[j] + kz*zc[k] + psi_3
                arg4 = kx*xc[i] - ky*yc[j] - kz*zc[k] + psi_4
                bm = A_m * np.sqrt(2.0) * (np.cos(arg1) + np.cos(arg2) + np.cos(arg3) + np.cos(arg4))
                _r[i,j,k] = np.sum(bm)

    print("Done! 3-D Turbulence has been generated!")
    return _r


def gaussian1D_FFT(N, k_func):
    """A FFT based generator for scalar gaussian fields in 1D
    Reference:Timmer, J and König, M. “On Generating Power Law Noise.” Astronomy & Astrophysics 300 (1995):
     1–30. https://doi.org/10.1017/CBO9781107415324.004.
    Arguments:
        L_drive {float} -- Driving length scale
        N {int}  -- size of domain will be (2*N+1)
        k_func {function} -- a function which takes an input k 
    Returns:
        signal {1D array of floats} -- a realisation of a 1D Gaussian process.
    Example:
        L_drive = 1e-2
        def power_spectrum(k,a):
            return k**-a

        def k41(k):
            return power_spectrum(k, 5/3)        
        
        sig = gaussian1D_FFT( N, k41)
        
        fig,ax = plt.subplots()
        x = np.linspace(-N,N, 2*N+1)
        ax.plot(x, sig)
    """
    M=2*N+1
    k=np.fft.fftfreq(M) #these are the frequencies, starting from 0 up to f_max, then -f_max to 0.

    K=np.sqrt(k**2)
    K=np.fft.fftshift(K)#numpy convention, highest frequencies at the centre

    Wr=np.random.randn(M) # random number from Gaussian for both 
    Wi=np.random.randn(M) # real and imaginary components

    Wr = Wr + np.flip(Wr) #f(-k)=f*(k)
    Wi = Wi - np.flip(Wi)

    W = Wr+1j*Wi

    F = W*np.sqrt(k_func(K)) # power spectra follows power law, so sqrt here.

    F_shift=np.fft.ifftshift(F)

    F_shift[0]=0 # 0 mean

    signal=np.fft.ifftn(F_shift)
    
    return signal.real

def gaussian2D_FFT(N, k_func):
    """A FFT based generator for scalar gaussian fields in 1D
    Reference:Timmer, J and König, M. “On Generating Power Law Noise.” Astronomy & Astrophysics 300 (1995):
     1–30. https://doi.org/10.1017/CBO9781107415324.004.
    Arguments:
        L_drive {float} -- Driving length scale
        N {int}  -- size of domain will be (2*N+1)^2
        k_func {function} -- a function which takes an input k 
    Returns:
        signal {2D array of floats} -- a realisation of a 2D Gaussian process.
    Example:
        N = 100
        L_drive = 1e-2
        def power_spectrum(k,a):
            return k**-a

        def k41(k):
            return power_spectrum(k, 5/3)        
        
        sig = gaussian2D_FFT(N, k41)
        
        fig,ax=plt.subplots()
        ax.imshow(sig, cmap='bwr', extent=[-N,N,-N,N])
        
    """

    M=2*N+1
    k=np.fft.fftfreq(M) #these are the frequencies, starting from 0 up to f_max, then -f_max to 0.


    KX,KY=np.meshgrid(k,k)
    K=np.sqrt(KX**2+KY**2)
    K=np.fft.fftshift(K)#numpy convention, highest frequencies at the centre

    Wr=np.random.randn(M, M) # random number from Gaussian for both 
    Wi=np.random.randn(M, M) # real and imaginary components

    Wr = Wr + np.flip(Wr) #f(-k)=f*(k)
    Wi = Wi - np.flip(Wi)

    W = Wr+1j*Wi

    F = W*np.sqrt(k_func(K)) # power spectra follows power law, so sqrt here.

    F_shift=np.fft.ifftshift(F)

    F_shift[0,0]=0 # 0 mean

    signal=np.fft.ifftn(F_shift)
    
    return signal.real

def gaussian3D_FFT(N, k_func):
    """A FFT based generator for scalar gaussian fields in 1D
    Reference:Timmer, J and König, M. “On Generating Power Law Noise.” Astronomy & Astrophysics 300 (1995):
     1–30. https://doi.org/10.1017/CBO9781107415324.004.
    Arguments:
        N {int}  -- size of domain will be (2*N+1)^3
        k_func {function} -- a function which takes an input k 
    Returns:
        signal {3D array of floats} -- a realisation of a 3D Gaussian process.
    Example:
        N = 100
        def power_spectrum(k,a):
            return k**-a

        def k41(k):
            return power_spectrum(k, 5/3)        
        
        sig = gaussian3D_FFT(N, k41)
        
        fig,ax=plt.subplots(3,3, figsize=(8,8), sharex=True, sharey=True)
        ax=ax.flatten()
        
        for a in ax:
            r=np.random.randint(0,ny)
            d=sig[r,:,:]
            a.imshow(d, cmap='bwr', extent=[-N,N,-N,N])
            a.set_title("y="+str(r))    """
    M = 2*N+1
    k = np.fft.fftfreq(M) #these are the frequencies, starting from 0 up to f_max, then -f_max to 0.

    KX,KY,KZ = np.meshgrid(k,k,k)
    K = np.sqrt(KX**2+KY**2+KZ**2)
    K = np.fft.fftshift(K)#numpy convention, highest frequencies at the centre

    Wr = np.random.randn(M, M, M) # random number from Gaussian for both 
    Wi = np.random.randn(M, M, M) # real and imaginary components

    Wr = Wr + np.flip(Wr) #f(-k)=f*(k)
    Wi = Wi - np.flip(Wi)

    W = Wr+1j*Wi

    F = W*np.sqrt(k_func(K)) # power spectra follows power law, so sqrt here.

    F_shift = np.fft.ifftshift(F)

    F_shift[0,0,0] = 0 # 0 mean

    signal = np.fft.ifftn(F_shift)
    
    return signal.real
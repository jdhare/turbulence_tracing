import numpy as np

def spectrum_3D_scalar(data, dx, k_bin_width):
    """Calculates and returns the 3D spectrum for a 3D gaussian field of scalars, assuming isotropy of the turbulence
        Example:
            d=np.random.randn(101,91,111)
            dx=1
            k_bins_weighted,spect3D=spectrum_3D_scalar(d, dx, k_bin_width=0.01)

            fig,ax=plt.subplots()
            ax.scatter(k_bins_weighted,spect3D)
    Arguments:
        data {(Mx,My,Mz) array of floats} -- 3D Gaussian field of scalars
        dx {float} -- grid spacing, assumed the same for all
        k_bin_width {float} -- width of bins in reciprocal space

    Returns:
        k_bins_weighted {array of floats} -- location of bin centres
        spect3D {array of floats} -- spectral power within bin
    """

    #fourier transform data, shift to have zero freq at centre, find power
    f=np.fft.fftshift(np.fft.fftn(data))
    fsqr=np.real(f*np.conj(f))

    #calculate k vectors in each dimension
    [Mx,My,Mz] = d.shape

    kx = np.fft.fftshift(np.fft.fftfreq(Mx, dx))
    ky = np.fft.fftshift(np.fft.fftfreq(My, dx))
    kz = np.fft.fftshift(np.fft.fftfreq(Mz, dx))

    #calculate magnitude of k at each grid point
    [KX,KY,KZ]=np.meshgrid(kx,ky,kz, indexing='ij')
    K=np.sqrt(KX**2+KY**2+KZ**2)

    #determine 1D spectrum of k, measured from origin
    #sort array in ascending k, and sort power by the same factor

    K_flat=K.flatten()
    fsqr_flat=fsqr.flatten()

    K_sort = K_flat[K_flat.argsort()]
    fsqr_sort = fsqr_flat[K_flat.argsort()]
    
    k_bin_num = int(np.floor(K_sort.max()/k_bin_width)+1)

    k_bins = k_bin_width*np.arange(0,k_bin_num+1)
    k_bins_weighted = (0.5*(k_bins[:-1]**3+k_bins[1:]**3))**(1/3)
    
    spect3D=np.zeros_like(k_bins_weighted)

    for i in range(1,k_bin_num):
        upper=K_sort<i*k_bin_width # find only values below upper bound: BOOL
        lower=K_sort>=(i-1)*k_bin_width #find only values above upper bound: BOOL
        f_filtered=fsqr_sort[upper*lower] # use super numpy array filtering to select only those which match both!
        spect3D[i-1] = f_filtered.mean() #and take their mean.
        
    return k_bins_weighted, spect3D
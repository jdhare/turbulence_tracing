import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
from scipy.interpolate import RectBivariateSpline
from ipywidgets import FloatProgress
from turboGen import gaussian3D_FFT, gaussian3Dcos, gaussian2D_FFT, gaussian1D_FFT

def power_spectrum(k,a):
    """Simple function for power laws

    Args:
        k (float array): wave number
        a (float): power law

    Returns:
        float array: k^-a (note minus sign is assumed!)
    """
    return k**-a

def k41_3D(k):
    """Helper function to generate a kolmogorov spectrum in 3D

    Args:
        k (float array): wave number       

    Returns:
        function: function of k
    """
    return power_spectrum(k, 11/3)

def generate_collimated_beam(N, X):
    """Simple helper function to generate a collimate beam

    Args:
        N (float): number of rays
        X (float): size of beam, will generate rays in -X/2 to X/2 in x and y.

    Returns:
        4xN float array: N rays, represented by x, theta, y, phi
    """
    rr0=np.random.rand(4,int(N))
    rr0[0,:]-=0.5
    rr0[2,:]-=0.5

    #x, θ, y, ϕ
    scales=np.diag(np.array([X,0,X,0]))
    r0=np.matmul(scales, rr0)
    
    return r0

L = sym.symbols('L', real=True) # used for sympy

def transform(matrix, rays):
    '''
    Simple wrapper for matrix multiplication
    '''
    return np.matmul(matrix,rays)

def distance(d):
    '''4x4 symbolic matrix for travelling a distance d
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    d = sym.Matrix([[1, d],
                    [0, 1]])
    L=sym.zeros(4,4)
    L[:2,:2]=d
    L[2:,2:]=d
    return L

d1=distance(L) #symbolic matrix
Z1=sym.lambdify([L], d1, "numpy") #a function to return a numpy matrix

def gradient_interpolator(ne, x, y):
    """Deceptively simple. First we take the gradient of ne, use a second order centered differences approach
    Then we create a bivariate spline to handle interpolation of this gradient field

    Args:
        ne (NxM float array): electron density
        x (M float array): x coordinates
        y (N float array): y coordinates

    Returns:
        RectBivariateSpline tuple: Two functions which take coordinates and return values of the gradient.
    """
    grad_ney, grad_nex=np.gradient(ne, y, x)
    gx=RectBivariateSpline(y,x,grad_nex)
    gy=RectBivariateSpline(y,x,grad_ney)
    
    return gx, gy

def deflect_rays(rays, grad_nex,grad_ney, dz, n_cr=1.21e21):
    """Deflects rays at a slice based on the gradient of the electron density

    Args:
        rays (4xN float): array representing N rays
        grad_nex (RectBivariateSpline): function which take coordinates and return values of the gradient
        grad_ney ([type]): function which take coordinates and return values of the gradient
        dz (float): distance in z which each slice covers. Use a consistent system
        n_cr (float): critical density. Change this based on wavelength of laser and units system. Default is 1 um laser in cm^-3

    Returns:
        4xN float array: rays after deflection
    """
    n_cr=1.21e21 #cm^-3, assumes ne has the same units
    xs=rays[0,:]
    ys=rays[2,:]
    dangle=np.zeros_like(rays)
    dangle[1,:]=grad_nex(ys, xs, grid=False)*dz/(2*n_cr)
    dangle[3,:]=grad_ney(ys, xs, grid=False)*dz/(2*n_cr)
    return rays+dangle

def histogram(rays, bin_scale=10, pix_x=1000, pix_y=1000, Lx=10,Ly=10):
    """Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

    Args:
        rays (4xN float): array representing N rays
        bin_scale (int, optional): bin size, same in x and y. Defaults to 10.
        pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
        pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        Lx (int, optional): x detector size in consistent units. Defaults to 10.
        Ly (int, optional): y detector size in consistent units. Defaults to 10.

    Returns:
        MxN array, M array, N array: binned histogram and bin edges.
    """
    x=rays[0,:]
    y=rays[2,:]

    x=x[~np.isnan(x)]
    y=y[~np.isnan(y)]

    H, xedges, yedges = np.histogram2d(x, y, 
                                       bins=[pix_x//bin_scale, pix_y//bin_scale],
                                      range=[[-Lx/2, Lx/2],[-Ly/2,Ly/2]])
    H=H.T
    return H, xedges, yedges

def plot_histogram(H, xedges, yedges, ax, clim=None, cmap=None):
    """[summary]

    Args:
        H (MxN float): histogram
        xedges (N float): x bin edges
        yedges (M float): y bin edges
        ax (matplotlib axis): axis to plot to
        clim (tuple, optional): Limits for imshow Defaults to None.
        cmap (str, optional): matplotlib colourmap. Defaults to None.
    """
    ax.imshow(H, interpolation='nearest', origin='low', clim=clim, cmap=cmap,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect=1)

class GridTracer:
    """
    Provides the functions for examining and tracing rays through a grid of electron density.
    Inherit from this and implement __init__ for different grid configurations.
    """
    def plot_ne_slices(self):
        """Plot 9 slices from the density grid, for inspection

        Returns:
            fig, ax: matplotlib figure and axis.
        """
        fig,ax=plt.subplots(3,3, figsize=(8,8), sharex=True, sharey=True)
        ax=ax.flatten()

        sc=self.scale/2

        for i, a in enumerate(ax):
            r=(2*self.N+1)*i//9
            d=self.ne_grid[r,:,:]
            a.imshow(d, cmap='bwr', extent=[-sc,sc,-sc, sc])
            a.set_title("z="+str(round(self.z[r])))
            
        return fig, ax
    
    def solve(self, r0):
        """Trace rays through the turbulent grid

        Args:
            r0 (4xN float): array of N rays, in their initial configuration
        """
        f = FloatProgress(min=0, max=self.ne_grid.shape[0], description='Progress:')
        display(f)
        
        self.r0 = r0 # keep the original
        dz = self.z[1]-self.z[0]
        DZ = Z1(dz) # matrix to push rays by dz

        rt = r0.copy() # iterate to save memory, starting at r0

        for i, ne_slice in enumerate(self.ne_grid):
            f.value = i

            gx, gy = gradient_interpolator(ne_slice, self.x, self.y)
            rr1 = deflect_rays(rt, gx, gy, dz=dz)
            rt = transform(DZ, rr1)
            
        self.rt = rt
        
    def plot_rays(self, clim=None):
        H0, xedges0, yedges0 = histogram(self.r0, bin_scale=10, pix_x=1000, pix_y=1000, Lx=10,Ly=10)
        Hf, xedgesf, yedgesf = histogram(self.rt, bin_scale=10, pix_x=1000, pix_y=1000, Lx=10,Ly=10)
        
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4))
        
        plot_histogram(H0, xedges0,yedges0, ax1, clim=clim)
        plot_histogram(Hf, xedgesf,yedgesf, ax2, clim=clim)


class TurbulentGrid(GridTracer):
    """Trace rays through a turbulent electron density defined on a grid
    """
    def __init__(self, N, spectrum, n_e0, dn_e, scale):
        """generate a turbulent grid.

        You can use cm^-3 for density and mm for scales.
        This code assumes x,y, and z have the same grid spacing.
        If this is so, the scale difference drops out because \nabla n_e/n_cr dz has units of
        [mm^-1] [cm^-3]/[cm^-3] [mm] = dimless (radians).
        If you are using this class as a guide on how to implement another GridTracer, be careful!
        Safest would be to use cm for everything, but they are a bit large for our porpoises.

        Args:
            N (int): half size of cube, will be 2*N+1
            spectrum (function of k): a spectrum, such as k**-11/3
            n_e0 (float): mean electron density
            dn_e (float): standard deviation of electron density
            scale (float): length of a box side. 
        """
        self.N=N
        self.scale=scale
        self.x = np.linspace(-scale,scale,2*N+1)
        self.y = self.x
        self.z = self.x
        
        s3 = gaussian3D_FFT(N, spectrum)
        self.ne_grid = n_e0 + dn_e*s3/s3.std()
    

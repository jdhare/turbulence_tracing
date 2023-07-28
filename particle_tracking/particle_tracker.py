"""PARTICLE TRACKER
BASED ON: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.61.895

SOLVES: 
$ \frac{d\vec{v}}{dt} = -\nabla \left( \frac{c^2}{2} \frac{n_e}{n_c} \right) $

$ \frac{d\vec{x}}{dt} = \vec{v} $

CODE BY: Aidan CRILLY
REFACTORING: Jack HARE

EXAMPLES:
#############################
#NULL TEST: no deflection
import particle_tracker as pt

## Create the coordiantes for a cube with 201x201x201 cells, 
## and attach a [-5,5] mm coordinate system to them.

N_V = 100
M_V = 2*N_V+1
extent = 5.0e-3
x = np.linspace(-extent,extent,M_V)
y = np.linspace(-extent,extent,M_V)
z = np.linspace(-extent,extent,M_V)

null = pt.ElectronCube(x,y,z)
null.test_null()
null.calc_dndr()

### solve
null.solve()
s0 = null.s0
rf = null.rf

### Plot
fig, axs = plt.subplots(2, 2, figsize=(8,6), dpi = 200)
nbins = 201

ax1 = axs[0,0]
_,_,_,im1 = ax1.hist2d(s0[0]*1e3, s0[1]*1e3, bins=(nbins, nbins), cmap='gray');
plt.colorbar(im1,ax=ax1)
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
ax1.set_aspect('equal')

ax2 = axs[0,1]
_,_,_,im2 = ax2.hist2d(s0[3]*1e3, s0[4]*1e3, bins=(nbins, nbins), cmap='gray');
plt.colorbar(im2,ax=ax2)
ax2.set_xlabel(r"$\theta$ (mrad)")
ax2.set_ylabel(r"$\phi$ (mrad)")
ax2.set_aspect('equal')

ax3 = axs[1,0]
_,_,_,im3 = ax3.hist2d(rf[0]*1e3, rf[2]*1e3, bins=(nbins, nbins), cmap='gray');
plt.colorbar(im3,ax=ax3)
ax3.set_xlabel("x (mm)")
ax3.set_ylabel("y (mm)")
ax3.set_aspect('equal')

ax4 = axs[1,1]
_,_,_,im4 = ax4.hist2d(rf[1]*1e3, rf[3]*1e3, bins=(nbins, nbins), cmap='gray');
plt.colorbar(im4,ax=ax4)
ax4.set_xlabel(r"$\theta$ (mrad)")
ax4.set_ylabel(r"$\phi$ (mrad)")
ax4.set_aspect('equal')


fig.tight_layout()

###########################
#SLAB TEST: Deflect rays in -ve x-direction
import particle_tracker as pt

N_V = 100
M_V = 2*N_V+1
extent = 5.0e-3
x = np.linspace(-extent,extent,M_V)
y = np.linspace(-extent,extent,M_V)
z = np.linspace(-extent,extent,M_V)

slab = pt.ElectronCube(x,y,z)
slab.test_slab(s=8, n_e0=1e25)
slab.calc_dndr()

## Initialise rays and solve
slab.init_beam(Np = 100000, beam_size=2e-3, divergence = 10e-3)
slab.solve()
s0 = slab.s0
rf = slab.rf

## Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), dpi = 200)
nbins = 201

_,_,_,im1 = ax1.hist2d(rf[0]*1e3, rf[2]*1e3, bins=(nbins, nbins), range = ((-3,3), (-3,3)), cmap=plt.cm.gray);
plt.colorbar(im1,ax=ax1)
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
ax1.set_aspect('equal')

_,_,_,im2 = ax2.hist2d(rf[1]*1e3, rf[3]*1e3, bins=(nbins, nbins), range = ((-100,100), (-100,100)), cmap=plt.cm.gray);
plt.colorbar(im2,ax=ax2)
ax2.set_xlabel(r"$\theta$ (mrad)")
ax2.set_ylabel(r"$\phi$ (mrad)")
ax1.set_aspect('equal')

fig.tight_layout()
"""

import numpy as np
from scipy.integrate import odeint,solve_ivp
from scipy.interpolate import RegularGridInterpolator
from time import time
import scipy.constants as sc
import pickle
from datetime import datetime

c = sc.c # honestly, this could be 3e8 *shrugs*

class ElectronCube:
    """A class to hold and generate electron density cubes
    """
    
    def __init__(self, x, y, z, probing_direction = 'z'):
        """
        Example:
            N_V = 100
            M_V = 2*N_V+1
            ne_extent = 5.0e-3
            ne_x = np.linspace(-ne_extent,ne_extent,M_V)
            ne_y = np.linspace(-ne_extent,ne_extent,M_V)
            ne_z = np.linspace(-ne_extent,ne_extent,M_V)

        Args:
            x (float array): x coordinates, m
            y (float array): y coordinates, m
            z (float array): z coordinates, m
            extent (float): physical size, m
        """
        self.z,self.y,self.x = z, y, x
        self.extent_x = x.max()
        self.extent_y = y.max()
        self.extent_z = z.max()
        self.probing_direction = probing_direction
        
    def test_null(self):
        """
        Null test, an empty cube
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = np.zeros_like(self.XX)
        
    def test_slab(self, s=1, n_e0=2e23):
        """A slab with a linear gradient in x:
        n_e =  n_e0 * (1 + s*x/extent)

        Will cause a ray deflection in x

        Args:
            s (int, optional): scale factor. Defaults to 1.
            n_e0 ([type], optional): mean density. Defaults to 2e23 m^-3.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = n_e0*(1.0+s*self.XX/self.extent_x)
        
    def test_linear_cos(self,s1=0.1,s2=0.1,n_e0=2e23,Ly=1):
        """Linearly growing sinusoidal perturbation

        Args:
            s1 (float, optional): scale of linear growth. Defaults to 0.1.
            s2 (float, optional): amplitude of sinusoidal perturbation. Defaults to 0.1.
            n_e0 ([type], optional): mean electron density. Defaults to 2e23 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = n_e0*(1.0+s1*self.XX/self.extent_x)*(1+s2*np.cos(2*np.pi*self.YY/Ly))
        
    def test_exponential_cos(self,n_e0=1e24,Ly=1e-3, s=2e-3):
        """Exponentially growing sinusoidal perturbation

        Args:
            n_e0 ([type], optional): mean electron density. Defaults to 1e24 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1e-3 m.
            s ([type], optional): scale of exponential growth. Defaults to 2e-3 m.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        self.ne = n_e0*10**(self.XX/s)*(1+np.cos(2*np.pi*self.YY/Ly))

    def test_lens(self,n_e0=1e24,LR=1e-3):
        """Normal distribution with axis along z

        Args:
            n_e0 ([type], optional): peak electron density. Defaults to 1e24 m^-3.
            LR (int, optional): Spatial scale of gaussian. Defaults to 1e-3 m.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        RR = np.sqrt(self.XX**2 + self.YY**2)
        self.ne = n_e0*np.exp(-RR**2/LR**2)

    def test_liner(self,n_e0=1e24,LR=1e-3):
        """Normal distribution with axis along y

        Args:
            n_e0 ([type], optional): peak electron density. Defaults to 1e24 m^-3.
            LR (int, optional): Spatial scale of gaussian. Defaults to 1e-3 m.
        """
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x,self.y,self.z, indexing='ij')
        RR = np.sqrt(self.XX**2 + self.ZZ**2)
        self.ne = n_e0*np.exp(-RR**2/LR**2)

    def external_ne(self, ne):
        """Load externally generated grid

        Args:
            ne ([type]): MxMxM grid of density in m^-3
        """
        self.ne = ne

    def calc_dndr(self, lwl=1053e-9, ne_max = 1):
        """Generate interpolators for derivatives.

        Args:
            lwl (float, optional): laser wavelength. Defaults to 1053e-9 m.
        """

        self.omega = 2*np.pi*(c/lwl)
        nc = 3.14207787e-4*self.omega**2

        ne_nc = self.ne/nc #normalise to critical density
        ne_nc[ne_nc>ne_max] = ne_max # don't allow densities above ne_max
        self.ne_nc = ne_nc
        
        #More compact notation is possible here, but we are explicit
        self.dndx = -0.5*c**2*np.gradient(self.ne_nc,self.x,axis=0)
        self.dndy = -0.5*c**2*np.gradient(self.ne_nc,self.y,axis=1)
        self.dndz = -0.5*c**2*np.gradient(self.ne_nc,self.z,axis=2)
        
        self.dndx_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndx, bounds_error = False, fill_value = 0.0)
        self.dndy_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndy, bounds_error = False, fill_value = 0.0)
        self.dndz_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndz, bounds_error = False, fill_value = 0.0)
   
    def dndr(self,x):
        """returns the gradient at the locations x

        Args:
            x (3xN float): N [x,y,z] locations

        Returns:
            3 x N float: N [dx,dy,dz] electron density gradients
        """
        grad = np.zeros_like(x)
        grad[0,:] = self.dndx_interp(x.T)
        grad[1,:] = self.dndy_interp(x.T)
        grad[2,:] = self.dndz_interp(x.T)
        return grad

    def init_beam(self, Np, beam_size, divergence):
        """[summary]

        Args:
            Np (int): Number of photons
            beam_size (float): beam radius, m
            divergence (float): beam divergence, radians
            ne_extent (float): size of electron density cube, m. Used to back propagate the rays to the start
            probing_direction (str): direction of probing. I suggest 'z', the best tested

        Returns:
            s0, 6 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s
        """
        s0 = np.zeros((6,Np))
        # position, uniformly within a circle
        t  = 2*np.pi*np.random.rand(Np) #polar angle of position
        u  = np.random.rand(Np)+np.random.rand(Np) # radial coordinate of position
        u[u > 1] = 2-u[u > 1]
        # angle
        ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
        χ = divergence*np.random.randn(Np) #polar angle of velocity

        if(self.probing_direction == 'x'):
            self.extent = self.extent_x
            # Initial velocity
            s0[3,:] = c * np.cos(χ)
            s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = self.extent
            s0[1,:] = beam_size*u*np.cos(t)
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'y'):
            self.extent = self.extent_y
            # Initial velocity
            s0[4,:] = c * np.cos(χ)
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
            # Initial position
            s0[0,:] = beam_size*u*np.cos(t)
            s0[1,:] = -self.extent
            s0[2,:] = beam_size*u*np.sin(t)
        elif(self.probing_direction == 'z'):
            self.extent = self.extent_z
            # Initial velocity
            s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
            s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
            s0[5,:] = c * np.cos(χ)
            # Initial position
            s0[0,:] = beam_size*u*np.cos(t)
            s0[1,:] = beam_size*u*np.sin(t)
            s0[2,:] = -self.extent
        self.s0 = s0

    def solve(self, method = 'RK45'):
        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume

        t  = np.linspace(0.0,np.sqrt(8.0)*self.extent/c,2)

        s0 = self.s0.flatten() #odeint insists

        start = time()
        dsdt_ODE = lambda t, y: dsdt(t, y, self)
        sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t, method = method)
        finish = time()
        print("Ray trace completed in:\t",finish-start,"s")

        Np = s0.size//6
        self.sf = sol.y[:,-1].reshape(6,Np)
        # Fix amplitudes
        self.rf = self.ray_at_exit()
        return self.rf

    def ray_at_exit(self):
        """Takes the output from the 6D solver and returns 4D rays for ray-transfer matrix techniques.
        Effectively finds how far the ray is from the end of the volume, returns it to the end of the volume.

        Args:
            ode_sol (6xN float): N rays in (x,y,z,vx,vy,vz) format, m and m/s and amplitude, phase and polarisation
            ne_extent (float): edge length of cube, m
            probing_direction (str): x, y or z.

        Returns:
            [type]: [description]
        """
        ode_sol = self.sf
        Np = ode_sol.shape[1] # number of photons
        ray_p = np.zeros((4,Np))

        x, y, z, vx, vy, vz = ode_sol[0], ode_sol[1], ode_sol[2], ode_sol[3], ode_sol[4], ode_sol[5]

        # Resolve distances and angles
        # YZ plane
        if(self.probing_direction == 'x'):
            t_bp = (x-self.extent)/vx
            # Positions on plane
            ray_p[0] = y-vy*t_bp
            ray_p[2] = z-vz*t_bp
            # Angles to plane
            ray_p[1] = np.arctan(vy/vx)
            ray_p[3] = np.arctan(vz/vx)
        # XZ plane
        elif(self.probing_direction == 'y'):
            t_bp = (y-self.extent)/vy
            # Positions on plane
            ray_p[0] = x-vx*t_bp
            ray_p[2] = z-vz*t_bp
            # Angles to plane
            ray_p[1] = np.arctan(vx/vy)
            ray_p[3] = np.arctan(vz/vy)
        # XY plane
        elif(self.probing_direction == 'z'):
            t_bp = (z-self.extent)/vz
            # Positions on plane
            ray_p[0] = x-vx*t_bp
            ray_p[2] = y-vy*t_bp
            # Angles to plane
            ray_p[1] = np.arctan(vx/vz)
            ray_p[3] = np.arctan(vy/vz)

        return ray_p

    def save_output_rays(self, fn = None):
        """
        Saves the output rays as a binary numpy format for minimal size.
        Auto-names the file using the current date and time.
        """
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

        if fn is None:
            fn = '{} rays.npy'.format(dt_string)
        else:
            fn = '{}.npy'.format(fn)
        with open(fn,'wb') as f:
            np.save(f, self.rf)

    
def dsdt(t, s, ElectronCube):
    """Returns an array with the gradients and velocity per ray for ode_int. Cannot be a method of ElectronCube due to expected call signature for the ODE solver

    Args:
        t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        s (6N float array): flattened 6xN array of rays used by ode_int
        ElectronCube (ElectronCube): an ElectronCube object which can calculate gradients

    Returns:
        6N float array: flattened array for ode_int
    """
    Np     = s.size//6
    s      = s.reshape(6,Np)
    sprime = np.zeros_like(s)
    # Velocity and position
    v = s[3:,:]
    x = s[:3,:]

    sprime[3:6,:] = ElectronCube.dndr(x)
    sprime[:3,:]  = v

    return sprime.flatten()

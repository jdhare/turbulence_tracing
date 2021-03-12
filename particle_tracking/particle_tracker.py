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

N_V = 100
M_V = 2*N_V+1
ne_extent = 5.0e-3
ne_x = np.linspace(-ne_extent,ne_extent,M_V)
ne_y = np.linspace(-ne_extent,ne_extent,M_V)
ne_z = np.linspace(-ne_extent,ne_extent,M_V)

null = pt.ElectronCube(ne_x,ne_y,ne_z,ne_extent)
null.test_null()
null.calc_dndr()

### Initialise rays
s0 = pt.init_beam(Np = 100000, beam_size=5e-3, divergence = 0.5e-3, ne_extent = ne_extent)
### solve
null.solve(s0)
rf = null.rf

### Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
nbins = 201

_,_,_,im1 = ax1.hist2d(rf[0]*1e3, rf[2]*1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
plt.colorbar(im1,ax=ax1)
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
_,_,_,im2 = ax2.hist2d(rf[1]*1e3, rf[3]*1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
plt.colorbar(im2,ax=ax2)
ax2.set_xlabel(r"$\theta$ (mrad)")
ax2.set_ylabel(r"$\phi$ (mrad)")

fig.tight_layout()

###########################
#SLAB TEST: Deflect rays in -ve x-direction
import particle_tracker as pt

N_V = 100
M_V = 2*N_V+1
ne_extent = 6.0e-3
ne_x = np.linspace(-ne_extent,ne_extent,M_V)
ne_y = np.linspace(-ne_extent,ne_extent,M_V)
ne_z = np.linspace(-ne_extent,ne_extent,M_V)

slab = pt.ElectronCube(ne_x,ne_y,ne_z,ne_extent)
slab.test_slab(s=10, n_e0=1e25)
slab.calc_dndr()

## Initialise rays and solve
s0 = pt.init_beam(Np = 100000, beam_size=5e-3, divergence = 0, ne_extent = ne_extent)
slab.solve(s0)
rf = slab.rf

## Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
nbins = 201

_,_,_,im1 = ax1.hist2d(rf[0]*1e3, rf[2]*1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
plt.colorbar(im1,ax=ax1)
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
_,_,_,im2 = ax2.hist2d(rf[1]*1e3, rf[3]*1e3, bins=(nbins, nbins), cmap=plt.cm.jet);
plt.colorbar(im2,ax=ax2)
ax2.set_xlabel(r"$\theta$ (mrad)")
ax2.set_ylabel(r"$\phi$ (mrad)")

fig.tight_layout()
"""

import numpy as np
from scipy.integrate import odeint,solve_ivp
from scipy.interpolate import RegularGridInterpolator
from time import time
import scipy.constants as sc

c = sc.c # honestly, this could be 3e8 *shrugs*

class ElectronCube:
    """A class to hold and generate electron density cubes
    """
    
    def __init__(self, x, y, z, extent, B_on = False, inv_brems = False, phaseshift = False, probing_direction = 'z'):
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
        self.dx = x[1]-x[0]
        self.XX, self.YY, self.ZZ = np.meshgrid(x,y,z, indexing='ij')
        self.extent = extent
        self.probing_direction = probing_direction
        # Logical switches
        self.B_on       = B_on
        self.inv_brems  = inv_brems
        self.phaseshift = phaseshift
        
    def test_null(self):
        """
        Null test, an empty cube
        """
        self.ne = np.zeros_like(self.XX)
        
    def test_slab(self, s=1, n_e0=2e23):
        """A slab with a linear gradient in x:
        n_e =  n_e0 * (1 + s*x/extent)

        Will cause a ray deflection in x

        Args:
            s (int, optional): scale factor. Defaults to 1.
            n_e0 ([type], optional): mean density. Defaults to 2e23 m^-3.
        """
        self.ne = n_e0*(1.0+s*self.XX/self.extent)
        
    def test_linear_cos(self,s1=0.1,s2=0.1,n_e0=2e23,Ly=1):
        """Linearly growing sinusoidal perturbation

        Args:
            s1 (float, optional): scale of linear growth. Defaults to 0.1.
            s2 (float, optional): amplitude of sinusoidal perturbation. Defaults to 0.1.
            n_e0 ([type], optional): mean electron density. Defaults to 2e23 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1.
        """
        self.ne = n_e0*(1.0+s1*self.XX/self.extent)*(1+s2*np.cos(2*np.pi*self.YY/Ly))
        
    def test_exponential_cos(self,n_e0=1e24,Ly=1e-3, s=2e-3):
        """Exponentially growing sinusoidal perturbation

        Args:
            n_e0 ([type], optional): mean electron density. Defaults to 2e23 m^-3.
            Ly (int, optional): spatial scale of sinusoidal perturbation. Defaults to 1e-3 m.
            s ([type], optional): scale of exponential growth. Defaults to 2e-3 m.
        """
        self.ne = n_e0*10**(self.XX/s)*(1+np.cos(2*np.pi*self.YY/Ly))
        
    def external_ne(self, ne):
        """Load externally generated grid

        Args:
            ne ([type]): MxMxM grid of density in m^-3
        """
        self.ne = ne

    def external_B(self, B):
        """Load externally generated grid

        Args:
            B ([type]): MxMxMx3 grid of B field in T
        """
        self.B = B

    def external_Te(self, Te, Te_min = 1.0):
        """Load externally generated grid

        Args:
            Te ([type]): MxMxM grid of electron temperature in eV
        """
        self.Te = np.maximum(Te_min,Te)

    def external_Z(self, Z):
        """Load externally generated grid

        Args:
            Z ([type]): MxMxM grid of ionisation
        """
        self.Z = Z
        
    def test_B(self, Bmax=1.0):
        """A Bz field with a linear gradient in x:
        Bz =  Bmax*x/extent

        Args:
            Bmax ([type], optional): maximum B field, default 1.0 T
        """
        self.B          = np.zeros(np.append(np.array(self.XX.shape),3))
        self.B[:,:,:,2] = Bmax*self.XX/self.extent

    def calc_dndr(self, lwl=1053e-9):
        """Generate interpolators for derivatives.

        Args:
            lwl (float, optional): laser wavelength. Defaults to 1053e-9 m.
        """

        self.omega = 2*np.pi*(c/lwl)
        nc = 3.14207787e-4*self.omega**2

        # Find Faraday rotation constant http://farside.ph.utexas.edu/teaching/em/lectures/node101.html
        if (self.B_on):
            self.VerdetConst = 2.62e-13*lwl**2 # radians per Tesla per m^2

        self.ne_nc = self.ne/nc #normalise to critical density
        
        #More compact notation is possible here, but we are explicit
        self.dndx = -0.5*c**2*np.gradient(self.ne_nc,self.x,axis=0)
        self.dndy = -0.5*c**2*np.gradient(self.ne_nc,self.y,axis=1)
        self.dndz = -0.5*c**2*np.gradient(self.ne_nc,self.z,axis=2)
        
        self.dndx_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndx, bounds_error = False, fill_value = 0.0)
        self.dndy_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndy, bounds_error = False, fill_value = 0.0)
        self.dndz_interp = RegularGridInterpolator((self.x, self.y, self.z), self.dndz, bounds_error = False, fill_value = 0.0)

    # NRL formulary inverse brems - cheers Jack Halliday for coding in Python
    # Converted to rate coefficient by multiplying by group velocity in plasma
    def kappa(self):
        # Useful subroutines
        def omega_pe(ne):
            '''Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28'''
            return 5.64e4*np.sqrt(ne)
        def v_the(Te):
            '''Calculate electron thermal speed. Provide Te in eV. Retrurns result in m/s'''
            return 4.19e5*np.sqrt(Te)
        def V(ne, Te, Z, omega):
            o_pe  = omega_pe(ne)
            o_max = np.copy(o_pe)
            o_max[o_pe < omega] = omega
            L_classical = Z*sc.e/Te
            L_quantum = 2.760428269727312e-10/np.sqrt(Te) # sc.hbar/np.sqrt(sc.m_e*sc.e*Te)
            L_max = np.maximum(L_classical, L_quantum)
            return o_max*L_max
        def coloumbLog(ne, Te, Z, omega):
            return np.maximum(2.0,np.log(v_the(Te)/V(ne, Te, Z, omega)))
        ne_cc = self.ne*1e-6
        o_pe  = omega_pe(ne_cc)
        CL    = coloumbLog(ne_cc, self.Te, self.Z, self.omega)
        return 3.1e-5*self.Z*c*np.power(ne_cc/self.omega,2)*CL*np.power(self.Te, -1.5) # 1/s

    # Plasma refractive index
    def n_refrac(self):
        def omega_pe(ne):
            '''Calculate electron plasma freq. Output units are rad/sec. From nrl pp 28'''
            return 5.64e4*np.sqrt(ne)
        ne_cc = self.ne*1e-6
        o_pe  = omega_pe(ne_cc)
        o_pe[o_pe > self.omega] = self.omega
        return np.sqrt(1.0-(o_pe/self.omega)**2)

    def set_up_interps(self):
        # Electron density
        self.ne_interp = RegularGridInterpolator((self.x, self.y, self.z), self.ne, bounds_error = False, fill_value = 0.0)
        # Magnetic field
        if(self.B_on):
            self.Bx_interp = RegularGridInterpolator((self.x, self.y, self.z), self.B[:,:,:,0], bounds_error = False, fill_value = 0.0)
            self.By_interp = RegularGridInterpolator((self.x, self.y, self.z), self.B[:,:,:,1], bounds_error = False, fill_value = 0.0)
            self.Bz_interp = RegularGridInterpolator((self.x, self.y, self.z), self.B[:,:,:,2], bounds_error = False, fill_value = 0.0)
        # Inverse Bremsstrahlung
        if(self.inv_brems):
            self.kappa_interp = RegularGridInterpolator((self.x, self.y, self.z), self.kappa(), bounds_error = False, fill_value = 0.0)
        # Phase shift
        if(self.phaseshift):
            self.refractive_index_interp = RegularGridInterpolator((self.x, self.y, self.z), self.n_refrac(), bounds_error = False, fill_value = 1.0)

    def plot_midline_gradients(self,ax,probing_direction):
        """I actually don't know what this does. Presumably plots the gradients half way through the box? Cool.

        Args:
            ax ([type]): [description]
            probing_direction ([type]): [description]
        """
        N_V = self.x.shape[0]//2
        if(probing_direction == 'x'):
            ax.plot(self.y,self.dndx[:,N_V,N_V])
            ax.plot(self.y,self.dndy[:,N_V,N_V])
            ax.plot(self.y,self.dndz[:,N_V,N_V])
        elif(probing_direction == 'y'):
            ax.plot(self.y,self.dndx[N_V,:,N_V])
            ax.plot(self.y,self.dndy[N_V,:,N_V])
            ax.plot(self.y,self.dndz[N_V,:,N_V])
        elif(probing_direction == 'z'):
            ax.plot(self.y,self.dndx[N_V,N_V,:])
            ax.plot(self.y,self.dndy[N_V,N_V,:])
            ax.plot(self.y,self.dndz[N_V,N_V,:])
        else: # Default to y
            ax.plot(self.y,self.dndx[N_V,:,N_V])
            ax.plot(self.y,self.dndy[N_V,:,N_V])
            ax.plot(self.y,self.dndz[N_V,:,N_V])
    
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

    # Attenuation due to inverse bremsstrahlung
    def atten(self,x):
        if(self.inv_brems):
            return -self.kappa_interp(x.T)
        else:
            return 0.0

    # Phase shift introduced by refractive index
    def phase(self,x):
        if(self.phaseshift):
            return self.omega*self.refractive_index_interp(x.T)
        else:
            return 0.0

    def get_ne(self,x):
        return self.ne_interp(x.T)

    def get_B(self,x):
        B = np.array([self.Bx_interp(x.T),self.By_interp(x.T),self.Bz_interp(x.T)])
        return B

    def neB(self,x,v):
        """returns the VerdetConst ne B.v

        Args:
            x (3xN float): N [x,y,z] locations
            v (3xN float): N [vx,vy,vz] velocities

        Returns:
            N float: N values of ne B.v
        """
        if(self.B_on):
            ne_N = self.get_ne(x)
            Bv_N = np.sum(self.get_B(x)*v,axis=0)
            pol  = self.VerdetConst*ne_N*Bv_N
        else:
            pol = 0.0

        return pol

    def solve(self, s0, method = 'RK45'):
        # Need to make sure all rays have left volume
        # Conservative estimate of diagonal across volume
        # Then can backproject to surface of volume

        t  = np.linspace(0.0,np.sqrt(8.0)*self.extent/c,2)

        s0 = s0.flatten() #odeint insists

        start = time()
        dsdt_ODE = lambda t, y: dsdt(t, y, self)
        sol = solve_ivp(dsdt_ODE, [0,t[-1]], s0, t_eval=t, method = method)
        finish = time()
        print("Ray trace completed in:\t",finish-start,"s")

        Np = s0.size//9
        self.sf = sol.y[:,-1].reshape(9,Np)
        # Fix amplitudes
        self.sf[6,self.sf[6,:] < 0.0] = 0.0
        self.rf,self.Jf = ray_to_Jonesvector(self.sf, self.extent, probing_direction = self.probing_direction)
        return self.rf

    def clear_memory(self):
        """
        Clears variables not needed by solve method, saving memory

        Can also use after calling solve to clear ray positions - important when running large number of rays

        """
        self.dndx = None
        self.dndx = None
        self.dndx = None
        self.ne = None
        self.ne_nc = None
        self.sf = None
        self.rf = None
    
# ODEs of photon paths
def dsdt(t, s, ElectronCube):
    """Returns an array with the gradients and velocity per ray for ode_int

    Args:
        t (float array): I think this is a dummy variable for ode_int - our problem is time invarient
        s (9N float array): flattened 9xN array of rays used by ode_int
        ElectronCube (ElectronCube): an ElectronCube object which can calculate gradients

    Returns:
        9N float array: flattened array for ode_int
    """
    Np     = s.size//9
    s      = s.reshape(9,Np)
    sprime = np.zeros_like(s)
    # Velocity and position
    v = s[3:6,:]
    x = s[:3,:]
    # Amplitude, phase and polarisation
    a = s[6,:]
    p = s[7,:]
    r = s[8,:]

    sprime[3:6,:] = ElectronCube.dndr(x)
    sprime[:3,:]  = v
    sprime[6,:]   = ElectronCube.atten(x)*a
    sprime[6,a < 1e-5] = 0.0
    sprime[7,:]   = ElectronCube.phase(x)
    sprime[8,:]   = ElectronCube.neB(x,v)

    return sprime.flatten()

def init_beam(Np, beam_size, divergence, ne_extent, probing_direction = 'z', coherent = False):
    """[summary]

    Args:
        Np (int): Number of photons
        beam_size (float): beam radius, m
        divergence (float): beam divergence, radians
        ne_extent (float): size of electron density cube, m. Used to back propagate the rays to the start
        probing_direction (str): direction of probing. I suggest 'z', the best tested

    Returns:
        s0, 9 x N float: N rays with (x, y, z, vx, vy, vz) in m, m/s and amplitude, phase and polarisation (a, p, r) 
    """
    s0 = np.zeros((9,Np))
    # position, uniformly within a circle
    t  = 2*np.pi*np.random.rand(Np) #polar angle of position
    u  = np.random.rand(Np)+np.random.rand(Np) # radial coordinate of position
    u[u > 1] = 2-u[u > 1]
    # angle
    ϕ = np.pi*np.random.rand(Np) #azimuthal angle of velocity
    χ = divergence*np.random.randn(Np) #polar angle of velocity

    if(probing_direction == 'x'):
        # Initial velocity
        s0[3,:] = c * np.cos(χ)
        s0[4,:] = c * np.sin(χ) * np.cos(ϕ)
        s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
        # Initial position
        s0[0,:] = -ne_extent
        s0[1,:] = beam_size*u*np.cos(t)
        s0[2,:] = beam_size*u*np.sin(t)
    elif(probing_direction == 'y'):
        # Initial velocity
        s0[4,:] = c * np.cos(χ)
        s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
        s0[5,:] = c * np.sin(χ) * np.sin(ϕ)
        # Initial position
        s0[0,:] = beam_size*u*np.cos(t)
        s0[1,:] = -ne_extent
        s0[2,:] = beam_size*u*np.sin(t)
    elif(probing_direction == 'z'):
        # Initial velocity
        s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
        s0[4,:] = c * np.sin(χ) * np.sin(ϕ)
        s0[5,:] = c * np.cos(χ)
        # Initial position
        s0[0,:] = beam_size*u*np.cos(t)
        s0[1,:] = beam_size*u*np.sin(t)
        s0[2,:] = -ne_extent
    else: # Default to y
        print("Default to y")
        # Initial velocity
        s0[4,:] = c * np.cos(χ)
        s0[3,:] = c * np.sin(χ) * np.cos(ϕ)
        s0[5,:] = c * np.sin(χ) * np.sin(ϕ)        
        # Initial position
        s0[0,:] = beam_size*u*np.cos(t)
        s0[1,:] = -ne_extent
        s0[2,:] = beam_size*u*np.sin(t)

    # Initialise amplitude, phase and polarisation

    if coherent is False:
        #then it's incoherent, random phase
        print('Incoherent')
        phase = 2*np.pi*np.random.rand(Np)
    if coherent is True:
        print('Coherent')
        phase = np.zeros(Np)

    s0[6,:] = 1.0
    s0[7,:] = phase
    s0[8,:] = 0.0
    return s0

# Need to backproject to ne volume, then find angles
def ray_to_Jonesvector(ode_sol, ne_extent, probing_direction = 'z'):
    """Takes the output from the 9D solver and returns 6D rays for ray-transfer matrix techniques.
    Effectively finds how far the ray is from the end of the volume, returns it to the end of the volume.

    Args:
        ode_sol (6xN float): N rays in (x,y,z,vx,vy,vz) format, m and m/s and amplitude, phase and polarisation
        ne_extent (float): edge length of cube, m
        probing_direction (str): x, y or z.

    Returns:
        [type]: [description]
    """
    Np = ode_sol.shape[1] # number of photons
    ray_p = np.zeros((4,Np))
    ray_J = np.zeros((2,Np),dtype=np.complex)

    x, y, z, vx, vy, vz = ode_sol[0], ode_sol[1], ode_sol[2], ode_sol[3], ode_sol[4], ode_sol[5]

    # Resolve distances and angles
    # YZ plane
    if(probing_direction == 'x'):
        t_bp = (x-ne_extent)/vx
        # Positions on plane
        ray_p[0] = y-vy*t_bp
        ray_p[2] = z-vz*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vy/vx)
        ray_p[3] = np.arctan(vz/vx)
    # XZ plane
    elif(probing_direction == 'y'):
        t_bp = (y-ne_extent)/vy
        # Positions on plane
        ray_p[0] = x-vx*t_bp
        ray_p[2] = z-vz*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vx/vy)
        ray_p[3] = np.arctan(vz/vy)
    # XY plane
    elif(probing_direction == 'z'):
        t_bp = (z-ne_extent)/vz
        # Positions on plane
        ray_p[0] = x-vx*t_bp
        ray_p[2] = y-vy*t_bp
        # Angles to plane
        ray_p[1] = np.arctan(vx/vz)
        ray_p[3] = np.arctan(vy/vz)

    # Resolve Jones vectors
    amp,phase,pol = ode_sol[6], ode_sol[7], ode_sol[8]
    # Assume initially polarised along y
    E_x_init = np.zeros(Np)
    E_y_init = np.ones(Np)
    # Perform rotation for polarisation, multiplication for amplitude, and complex rotation for phase
    ray_J[0] = amp*(np.cos(phase)+1.0j*np.sin(phase))*(np.cos(pol)*E_x_init-np.sin(pol)*E_y_init)
    ray_J[1] = amp*(np.cos(phase)+1.0j*np.sin(phase))*(np.sin(pol)*E_x_init+np.cos(pol)*E_y_init)

    # ray_p [x,phi,y,theta], ray_J [E_x,E_y]

    return ray_p,ray_J



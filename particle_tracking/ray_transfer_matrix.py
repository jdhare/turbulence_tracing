import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

'''
Example:

###INITIALISE RAYS###
#Rays are a 4 vector of x, theta, y, phi
#here we initialise 10*7 randomly distributed rays
rr0=np.random.rand(4,1000*1000*10)
rr0[0,:]-=0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[2,:]-=0.5

#x, θ, y, ϕ
scales=np.diag(np.array([10,0,10,0])) #set angles to 0, collimated beam. x, y in [-5,5]
rr0=np.matmul(scales, rr0)
r0=circular_aperture(10, rr0) #cut out a circle

###CREATE A SHOCK PAIR FOR TESTING###
def α(x, n_e0, w, x0, Dx, l=10):
    dn_e = n_e0*(np.tanh((x+Dx+x0)/w)**2-np.tanh((x-Dx+x0)/w)**2)
    n_c=1e21
    a = 0.5* l/n_c * dn_e
    return a

def ne(x,n_e0, w, Dx, x0):
    return n_e0*(np.tanh((x+Dx+x0)/w)-np.tanh((x-Dx+x0)/w))

def ne_ramp(y, ne_0, scale):
    return ne_0*10**(y/scale)

# Parameters for shock pair
w=0.1
Dx=1
x0=0
ne0=1e18
s=5

x=np.linspace(-5,5,1000)
y=np.linspace(-5,5,1000)

a=α(x, n_e0=ne0, w=w, Dx=Dx, x0=x0)
n=ne(x, n_e0=ne0, w=w, Dx=Dx, x0=x0)
ne0s=ne_ramp(y, ne_0=ne0, scale=s)

nn=np.array([ne(x, n_e0=n0, w=w, Dx=Dx, x0=x0) for n0 in ne0s])
nn=np.rot90(nn)

###PLOT SHOCKS###
fig, (ax1,ax2) = plt.subplots(1,2, figsize=(6.67/2, 2))

ax1.imshow(nn, clim=[1e16,1e19], cmap='inferno')
ax1.axis('off')
ax2.plot(x, n/5e18, label=r'$n_e$')
ax2.plot(x, a*57, label=r'$\alpha$')

ax2.set_xlim([-5,5])
ax2.set_xticks([])
ax2.set_yticks([])
ax2.legend(borderpad=0.5, handlelength=1, handletextpad=0.2, labelspacing=0.2)
fig.subplots_adjust(left=0, bottom=0.14, right=0.98, top=0.89, wspace=0.1, hspace=None)

###DEFLECT RAYS###
r0[3,:]=α(r0[2,:],n_e0=ne_ramp(r0[0,:], ne0, s), w=w, Dx=Dx, x0=x0)

###SOLVE FOR RAYS###
b=BurdiscopeRays(r0)
sh=ShadowgraphyRays(r0)
sc=SchlierenRays(r0)

sh.solve(displacement=10)
sh.histogram(bin_scale=10)
sc.solve()
sc.histogram(bin_scale=10)
b.solve()
b.histogram(bin_scale=10)

###PLOT DATA###
fig, axs = plt.subplots(1,3,figsize=(6.67, 1.8))

cm='gray'
clim=[0,100]

sh.plot(axs[1], clim=clim, cmap=cm)
#axs[0].imshow(nn.T, extent=[-5,5,-5,5])
sc.plot(axs[0], clim=clim, cmap=cm)
b.plot(axs[2], clim=clim, cmap=cm)

for ax in axs:
    ax.axis('off')
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=None)
'''

def transform(matrix, rays):
    '''
    Simple wrapper for matrix multiplication
    '''
    return np.matmul(matrix,rays)

def circular_aperture(R, rays):
    '''
    Filters rays to find those inside a radius R
    '''
    filt = rays[0,:]**2+rays[2,:]**2 > R**2
    rays[:,filt]=None
    return rays

def rect_aperture(Lx, Ly, rays):
    '''
    Filters rays inside a rectangular aperture, total size 2*Lx x 2*Ly
    '''
    filt1 = (rays[0,:]**2 > Lx**2)
    filt2 = (rays[2,:]**2 > Ly**2)
    filt=filt1*filt2
    rays[:,filt]=None
    return rays

def knife_edge(axis, rays):
    '''
    Filters rays using a knife edge.
    Default is a knife edge in y, can also do a knife edge in x.
    '''
    if axis is 'y':
        a=2
    else:
        a=0
    filt = rays[a,:] < 1e-1
    rays[:,filt]=None
    return rays

def lens(f1,f2):
    '''4x4 symbolic matrix for a thin lens, focal lengths f1 and f2 in orthogonal axes
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    l1= sym.Matrix([[1,    0],
                    [-1/f1, 1]])
    l2= sym.Matrix([[1,    0],
                    [-1/f2, 1]])
    L=sym.zeros(4,4)
    L[:2,:2]=l1
    L[2:,2:]=l2
    return L

def sym_lens(f):
    '''
    helper function to create an axisymmetryic lens
    '''
    return lens(f,f)

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


def ray(x, θ, y, ϕ):
    '''
    4x1 matrix representing a ray. Spatial units must be consistent, angular units in radians
    '''
    return sym.Matrix([x,
                       θ,
                       y,
                       ϕ])  

def d2r(d):
    # helper function, degrees to radians
    return d*np.pi/180

class BurdiscopeOptics:
    """
    Class to hold the Burdiscope optics
    """
    x, y, θ, ϕ, L = sym.symbols('x, y, θ, ϕ, L', real=True)
    #our two lenses. f1 is spherical, f2 is composite spherical/cylindrical
    f1=sym_lens(L/2)
    f2=lens(L/3, L/2)
    #our three distances
    d1=distance(L)
    d2=distance(3*L/2)
    d3=d1
    #ray-vector at selected planes
    X0=ray(x, θ, y, ϕ)
    X1=f1*d1*X0 #ray directly after f1
    X2=f2*d2*X1 #ray directly after second f1
    X3=d3*X2    #ray at detector
    #lambdify allows for numerical evaluation of symbolic expressions
    #these are the matrices which transfer rays between planes
    L1=sym.lambdify([L], f1*d1, "numpy")
    L2=sym.lambdify([L], f2*d2, "numpy")
    X3=sym.lambdify([L], d3, "numpy")
    
class ShadowgraphyOptics:
    """
    Class to hold the Shadwography optics
    """
    x, y, θ, ϕ, L = sym.symbols('x, y, θ, ϕ, L', real=True)
    #lenses
    f1=sym_lens(L/2)
    f2=sym_lens(L/3)
    #distances
    d1=distance(L)
    d2=distance(3*L/2)
    d3=d1
    #ray-vector at selected planes
    X0=ray(x, θ, y, ϕ)
    X1=f1*d1*X0 #ray directly after f1
    X2=d1*X1 #ray directly after second f1
    #lambdify allows for numerical evaluation of symbolic expressions
    #these are the matrices which transfer rays between planes
    L1=sym.lambdify([L], f1*d1, "numpy")
    L2=sym.lambdify([L], f2*d2, "numpy")
    X3=sym.lambdify([L], d1, "numpy")
    
class SchlierenOptics:
    x, y, θ, ϕ, L = sym.symbols('x, y, θ, ϕ, L', real=True)
    #lenses
    f1=sym_lens(L/2)
    f2=sym_lens(L/3)
    #distances
    d1=distance(L)
    d2=distance(L/2)
    #ray-vector at selected planes
    X0=ray(x, θ, y, ϕ)
    X1=f1*d1*X0 #ray directly after f1
    X2=d2*X1 #ray at Fourier Plane
    X3=f1*d1*X2 #ray at second lens
    X4=d1*X3 # ray at detector
    #lambdify allows for numerical evaluation of symbolic expressions
    #these are the matrices which transfer rays between planes
    L1=sym.lambdify([L], f1*d1, "numpy")
    X2=sym.lambdify([L], d2, "numpy") #fourier plane
    L2=sym.lambdify([L], f2*d1, "numpy") #second lens
    X3=sym.lambdify([L], d1, "numpy")

class Rays:
    """
    Inheritable class for ray diagnostics.
    """
    def __init__(self, r0, L=400, R=25, Lx=18, Ly=13.5):
        """Initialise ray diagnostic.

        Args:
            r0 (4xN float array): N rays, [x, theta, y, phi]
            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
        """        
        self.r0, self.L, self.R, self.Lx, self.Ly = r0, L, R, Lx, Ly
    def histogram(self, bin_scale=10, pix_x=3448, pix_y=2574, clear_mem=False):
        """Bin data into a histogram. Defaults are for a KAF-8300.
        Outputs are H, the histogram, and xedges and yedges, the bin edges.

        Args:
            bin_scale (int, optional): bin size, same in x and y. Defaults to 10.
            pix_x (int, optional): number of x pixels in detector plane. Defaults to 3448.
            pix_y (int, optional): number of y pixels in detector plane. Defaults to 2574.
        """        
        x=self.rf[0,:]
        y=self.rf[2,:]

        x=x[~np.isnan(x)]
        y=y[~np.isnan(y)]

        self.H, self.xedges, self.yedges = np.histogram2d(x, y, 
                                           bins=[pix_x//bin_scale, pix_y//bin_scale], 
                                           range=[[-self.Lx/2, self.Lx/2],[-self.Ly/2,self.Ly/2]])
        self.H = self.H.T

        # Optional - clear ray attributes to save memory
        if(clear_mem):
            self.clear_rays()

    def plot(self, ax, clim=None, cmap=None):
        ax.imshow(self.H, interpolation='nearest', origin='low', clim=clim, cmap=cmap,
                extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])

    def clear_rays(self):
        '''
        Clears the r0 and rf variables to save memory
        '''
        self.r0 = None
        self.rf = None

class BurdiscopeRays(Rays):
    '''
    Simple class to keep all the ray properties together
    '''              
    def solve(self):
        O=BurdiscopeOptics
        
        rr0=transform(O.X3(0), self.r0) # small displacement, currently does nothing

        rr1=transform(O.L1(self.L), rr0) # first lens
        r1=circular_aperture(self.R, rr1) # first lens cutoff

        rr2=transform(O.L2(self.L), r1) # second lens
        r2=circular_aperture(self.R, rr2) # second lens cutoff

        rr3=transform(O.X3(self.L), r2) #detector
        #3=rect_aperture(self.Lx/2,self.Ly/2,rr3) # detector cutoff
        self.rf=rr3
        
class ShadowgraphyRays(Rays):
    '''
    Simple class to keep all the ray properties together
    '''              
    def solve(self, displacement=10):
        O=ShadowgraphyOptics
        
        rr0=transform(O.X3(displacement), self.r0) #small displacement
        
        rr1=transform(O.L1(self.L), rr0) #lens 1
        r1=circular_aperture(self.R, rr1) # cut off

        rr2=transform(O.L2(self.L), r1) #lens 2
        r2=circular_aperture(self.R, rr2) # cut off

        rr3=transform(O.X3(self.L), r2) #detector
        #r3=rect_aperture(self.Lx/2,self.Ly/2,rr3) #cut off
        self.rf=rr3
        
class SchlierenRays(Rays):
    '''
    Simple class to keep all the ray properties together
    '''              
    def solve(self):
        O=SchlierenOptics
        
        rr0=transform(O.X3(0), self.r0) #small displacement

        rr1=transform(O.L1(self.L), rr0) #first lens
        r1=circular_aperture(self.R, rr1) #cut off

        rrk=transform(O.X2(self.L), r1) #fourier plane
        rk=knife_edge('y', rrk) #knife edge cuts off y.

        rr2=transform(O.L2(self.L), rk) #second lens
        r2=circular_aperture(self.R, rr2) #cut off

        rr3=transform(O.X3(self.L), r2) #detector
        #r3=rect_aperture(self.Lx/2,self.Ly/2,rr3) #cut off
        self.rf=rr3

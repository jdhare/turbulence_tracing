import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
Example:

###INITIALISE RAYS###
#Rays are a 4 vector of x, theta, y, phi,
#here we initialise 10*7 randomly distributed rays
rr0=np.random.rand(6,int(1e7))
rr0[0,:]-=0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[2,:]-=0.5

rr0[4,:]-=0.5 #rand generates [0,1], so we recentre [-0.5,0.5]
rr0[5,:]-=0.5

#x, θ, y, ϕ
scales=np.diag(np.array([10,0,10,0,1,1j])) #set angles to 0, collimated beam. x, y in [-5,5]. Circularly polarised beam, E_x = iE_y
rr0=np.matmul(scales, rr0)
r0=circular_aperture(5, rr0) #cut out a circle

### Shadowgraphy, no polarisation
## object_length: determines where the focal plane is. If you object is 10 mm long, object length = 5 will
## make the focal plane in the middle of the object. Yes, it's a bad variable name.
s = Shadowgraphy(rr0, L = 400, R = 25, object_length=5)
s.solve()
s.histogram(bin_scale = 25)
fig, axs = plt.subplots(figsize=(6.67, 6))

cm='gray'
clim=[0,100]

s.plot(axs, clim=clim, cmap=cm)

'''
def m_to_mm(r):
    rr = np.ndarray.copy(r)
    rr[0::2,:]*=1e3
    return rr

def lens(r, f1,f2):
    '''4x4 matrix for a thin lens, focal lengths f1 and f2 in orthogonal axes
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    l1= np.array([[1,    0],
                [-1/f1, 1]])
    l2= np.array([[1,    0],
                [-1/f2, 1]])
    L=np.zeros((4,4))
    L[:2,:2]=l1
    L[2:,2:]=l2

    return np.matmul(L, r)

def sym_lens(r, f):
    '''
    helper function to create an axisymmetryic lens
    '''
    return lens(r, f,f)

def distance(r, d):
    '''4x4 matrix  matrix for travelling a distance d
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    d = np.array([[1, d],
                    [0, 1]])
    L=np.zeros((4,4))
    L[:2,:2]=d
    L[2:,2:]=d
    return np.matmul(L, r)

def circular_aperture(r, R):
    '''
    Rejects rays outside radius R
    '''
    filt = r[0,:]**2+r[2,:]**2 > R**2
    r[:,filt]=None
    return r

def circular_stop(r, R):
    '''
    Rjects rays inside a radius R
    '''
    filt = r[0,:]**2+r[2,:]**2 < R**2
    r[:,filt]=None
    return r

def annular_stop(r, R1, R2):
    '''
    Rejects rays which fall between R1 and R2
    '''
    filt1 = (r[0,:]**2+r[2,:]**2 > R1**2)
    filt2 = (r[0,:]**2+r[2,:]**2 < R2**2)
    filt = (filt1 & filt2)

    return filt

def angular_filter(r, Rs):
    '''
    Filters rays to find those inside a radius R
    '''

    f = np.zeros((r.shape[1]), dtype = bool)

    for i in range(0, len(Rs)//2):
        R1 = Rs[2*i]
        R2 = Rs[2*i+1]
        f += annular_stop(r, R1, R2)
    r[:,f]=None
    return r

def plot_afr(Rs):
    fig, ax = plt.subplots(figsize = (4,4), dpi = 200)
    for i in range(0, len(Rs)//2):
        R1 = Rs[2*i]
        R2 = Rs[2*i+1]
        dR = R2-R1
        if dR >= R2:
            an = mpl.patches.Circle((0,0), R2)
        else:
            an = mpl.patches.Annulus((0,0), R2, dR)
        ax.add_patch(an)
    ax.set_xlim([-Rs.max(), Rs.max()])
    ax.set_ylim([-Rs.max(), Rs.max()])
    return fig, ax

def rect_aperture(r, Lx, Ly):
    '''
    Rejects rays outside a rectangular aperture, total size 2*Lx x 2*Ly
    '''
    filt1 = (r[0,:]**2 > Lx**2)
    filt2 = (r[2,:]**2 > Ly**2)
    filt=filt1*filt2
    r[:,filt]=None
    return r

def knife_edge(r, offset, axis, direction):
    '''
    Filters rays using a knife edge.
    Default is a knife edge in y, can also do a knife edge in x.
    '''
    if axis == 'y':
        a=2
    if axis == 'x':
        a=0
    if direction > 0:
        filt = r[a,:] > offset
    if direction < 0:
        filt = r[a,:] < offset
    if direction == 0:
        print('Direction must be <0 or >0')
    r[:,filt]=None
    return r

class Rays:
    """
    Inheritable class for ray diagnostics.
    """
    def __init__(self, r0, focal_plane = 0, L=400, R=25, Lx=18, Ly=13.5):
        """Initialise ray diagnostic.

        Args:
            r0 (4xN float array): N rays, [x, theta, y, phi]

            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
        """        
        self.focal_plane, self.L, self.R, self.Lx, self.Ly = focal_plane, L, R, Lx, Ly
        self.r0 = m_to_mm(r0)
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
        ax.imshow(self.H, interpolation='nearest', origin='lower', clim=clim, cmap=cmap,
                extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]])

    def clear_rays(self):
        '''
        Clears the r0 and rf variables to save memory
        '''
        self.r0 = None
        self.rf = None
        
class Shadowgraphy(Rays):
    """
    Example shadowgraphy diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    """
    def solve(self):
        ## 2 lens telescope, M = 1
        r1=distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2=circular_aperture(r1, self.R) # cut off
        r3=sym_lens(r2, self.L) #lens 1

        r4=distance(r3, self.L*2) #displace rays to lens 2.

        r5=circular_aperture(r4, self.R) # cut off
        r6=sym_lens(r5, self.L) #lens 2

        r7=distance(r6, self.L) #displace rays to detector

        self.rf = r7
        
class Schlieren_DF(Rays):
    """
    Example dark field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point afte rthe first lens which rejects rays which hit the focal planes at distance less than R [mm] from the optical axis.
    """
    def solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1=distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2=circular_aperture(r1, self.R) # cut off
        r3=sym_lens(r2, self.L) #lens 1

        r4=distance(r3, self.L) #displace rays to stop
        r5=circular_stop(r4, R = R) # stop

        r6=distance(r5, self.L) #displace rays to lens 2
        r7=circular_aperture(r6, self.R) # cut off
        r8=sym_lens(r7, self.L) #lens 2

        r9=distance(r8, self.L) #displace rays to detector
        self.rf = r9
        
class Schlieren_LF(Rays):
    """
    Example light field schlieren diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a circular stop placed at the focal point afte rthe first lens which accepts only rays which hit the focal planes at distance less than R [mm] from the optical axis.
    """
    def solve(self, R = 1):
        ## 2 lens telescope, M = 1
        r1=distance(self.r0, self.L - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2=circular_aperture(r1, self.R) # cut off
        r3=sym_lens(r2, self.L) #lens 1

        r4=distance(r3, self.L) #displace rays to stop
        r5=circular_aperture(r4, R = R) # stop

        r6=distance(r5, self.L) #displace rays to lens 2
        r7=circular_aperture(r6, self.R) # cut off
        r8=sym_lens(r7, self.L) #lens 2

        r9=distance(r8, self.L) #displace rays to detector
        self.rf = r9

class AFR(Rays):
    """
    Example angular fringe refractometry diagnostic. Inherits from Rays, has custom solve method.
    Implements a two lens telescope with M = 1. Both lenses have a f = L/2 focal length, where L is a length scale specified when the class is initialized.
    Each optic has a radius R, which is used to reject rays outside the numerical aperture of the optical system.
    There is a series of annuli placed at the focal point after the first lens. 
    The radius of these annuli are specified as a list, Rs. 
    The opaque regions of the stop are between odd and even elements of the list - the regions between even and odd elemetns are considered transparent.
    The list can be of arbitrary length with unevenly spaced elements.
    """
    def solve(self, Rs):
        ## 2 lens telescope, M = 1
        r1=distance(self.r0, self.L/2 - self.focal_plane) #displace rays to lens. Accounts for object with depth
        r2=circular_aperture(r1, self.R) # cut off
        r3=sym_lens(r2, self.L/2) #lens 1

        r4=distance(r3, self.L/4) #displace rays to stop
        r5=angular_filter(r4, Rs = Rs) # stop

        r6=distance(r5, self.L/4) #displace rays to lens 2
        r7=circular_aperture(r6, self.R) # cut off
        r8=sym_lens(r7, self.L/2) #lens 2

        r9=distance(r8, self.L/2) #displace rays to detector
        self.rf = r9
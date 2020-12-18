import numpy as np
import matplotlib.pyplot as plt

'''
Example:

###INITIALISE RAYS###
#Rays are a 6 vector of x, theta, y, phi, E_x, E_y, where E_x and E_y can be complex, and the rest are scalars.
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
s = Shadowgraphy(rr0, L = 400, R = 25, object_length=5)
s.solve()
s.histogram(bin_scale = 25)
fig, axs = plt.subplots(figsize=(6.67, 6))

cm='gray'
clim=[0,100]

s.plot(axs, clim=clim, cmap=cm)


### Faraday, with a polarisation β, in degrees, which puts the axis of extinction at beta degrees to the y direction.
### that is, beta = 0 extinguishes E_y, beta = 90 extinguishes E_x
### of course, in this example we have E_x = i E_y, so all the polariser will do is reduce the intensity.
f = Faraday(rr0, L = 400, R = 25, object_length=5)
f.solve(β = 80)
f.histogram(bin_scale = 25)
fig, axs = plt.subplots(figsize=(6.67, 6))

cm='gray'
clim=[0,100]

f.plot(axs, clim=clim, cmap=cm)

'''

I = np.array([[1, 0],
              [0, 1]])

def lens(f1,f2):
    '''6x6 matrix for a thin lens, focal lengths f1 and f2 in orthogonal axes
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    l1= np.array([[1,    0],
                [-1/f1, 1]])
    l2= np.array([[1,    0],
                [-1/f2, 1]])
    L=np.zeros((6,6))
    L[:2,:2]=l1
    L[2:4,2:4]=l2
    L[4:,4:]=I

    return L

def sym_lens(f):
    '''
    helper function to create an axisymmetryic lens
    '''
    return lens(f,f)

def distance(d):
    '''6x6 matrix  matrix for travelling a distance d
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    '''
    d = np.array([[1, d],
                    [0, 1]])
    L=np.zeros((6,6))
    L[:2,:2]=d
    L[2:4,2:4]=d
    L[4:,4:]=I
    return L

def polariser(β):
    '''6x6  matrix for a polariser with the axis of extinction at β radians to the vertical
    See: https://en.wikipedia.org/wiki/Ray_transfer_matrix_analysis
    and https://en.wikipedia.org/wiki/Jones_calculus
    '''
    L=np.zeros((6,6))
    L[:2,:2]=I
    L[2:4,2:4]=I
    L[4:,4:]=np.array([[np.cos(β)**2, np.cos(β)*np.sin(β)],
                        [np.cos(β)*np.sin(β), np.sin(β)**2]])
    return L

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

def knife_edge(axis, rays, edge = 1e-1):
    '''
    Filters rays using a knife edge.
    Default is a knife edge in y, can also do a knife edge in x.
    '''
    if axis is 'y':
        a=2
    else:
        a=0
    filt = rays[a,:] < edge
    rays[:,filt]=None
    return rays

    class Rays:
    """
    Inheritable class for ray diagnostics.
    """
    def __init__(self, r0, L=400, R=25, Lx=18, Ly=13.5, object_length = 0):
        """Initialise ray diagnostic.

        Args:
            r0 (6xN float array): N rays, [x, theta, y, phi, Ex, Ey]
            L (int, optional): Length scale L. First lens is at L. Defaults to 400.
            R (int, optional): Radius of lenses. Defaults to 25.
            Lx (int, optional): Detector size in x. Defaults to 18.
            Ly (float, optional): Detector size in y. Defaults to 13.5.
            Object_length (float, optional): Length of object (diagnostic focused at object center). Defaults to 0.
        """        
        self.r0, self.L, self.R, self.Lx, self.Ly, self.object_length = r0, L, R, Lx, Ly, object_length

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
        
        nonans = ~np.isnan(x)
        
        x=x[nonans]
        y=y[nonans]
        
        #treat the imaginary and real parts of E_x and E_y all separately.
        E_x_real = np.real(self.rf[4,:])
        E_x_imag = np.imag(self.rf[4,:])
        E_y_real = np.real(self.rf[5,:])
        E_y_imag = np.imag(self.rf[5,:])
        
        E_x_real = E_x_real[nonans]
        E_x_imag = E_x_imag[nonans]
        E_y_real = E_y_real[nonans]
        E_y_imag = E_y_imag[nonans]

        ## create four separate histograms for the real and imaginary parts of E_x and E-y
        self.H_Ex_real, self.xedges, self.yedges = np.histogram2d(x, y, 
                                                                   bins=[pix_x//bin_scale, pix_y//bin_scale], 
                                                                   range=[[-self.Lx/2, self.Lx/2],[-self.Ly/2,self.Ly/2]],
                                                                   normed = False, weights = E_x_real)
        
        self.H_Ex_imag, self.xedges, self.yedges = np.histogram2d(x, y, 
                                                                   bins=[pix_x//bin_scale, pix_y//bin_scale], 
                                                                   range=[[-self.Lx/2, self.Lx/2],[-self.Ly/2,self.Ly/2]],
                                                                   normed = False, weights = E_x_imag)
            
        self.H_Ey_real, self.xedges, self.yedges = np.histogram2d(x, y, 
                                                                   bins=[pix_x//bin_scale, pix_y//bin_scale], 
                                                                   range=[[-self.Lx/2, self.Lx/2],[-self.Ly/2,self.Ly/2]],
                                                                   normed = False, weights = E_y_real)
                
        self.H_Ey_imag, self.xedges, self.yedges = np.histogram2d(x, y, 
                                                                   bins=[pix_x//bin_scale, pix_y//bin_scale], 
                                                                   range=[[-self.Lx/2, self.Lx/2],[-self.Ly/2,self.Ly/2]],
                                                                   normed = False, weights = E_y_imag)
        
        # Recontruct the complex valued E_x and E_y components
        self.H_Ex = self.H_Ex_real+1j*self.H_Ex_imag
        self.H_Ey = self.H_Ey_real+1j*self.H_Ey_imag
        
        # Find the intensity using complex conjugates. Take the real value to remove very small (numerical error) imaginary components
        self.H = np.real(self.H_Ex*np.conj(self.H_Ex) + self.H_Ey*np.conj(self.H_Ey))

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
        
class Shadowgraphy(Rays):
    def solve(self):
        rl1=np.matmul(distance(self.L - self.object_length), self.r0) #displace rays to lens. Accounts for object with depth
        rc1=circular_aperture(self.R, rl1) # cut off
        r2=np.matmul(sym_lens(self.L/2), rc1) #lens 1

        rl2=np.matmul(distance(3*self.L/2), r2) #displace rays to lens 2.
        rc2=circular_aperture(self.R, rl2) # cut off
        r3=np.matmul(sym_lens(self.L/3), rc2) #lens 2

        rd3=np.matmul(distance(self.L), r3) #displace rays to detector

        self.rf = rd3
        
class Faraday(Rays):
    def solve(self, β = 3.0):
        rl1=np.matmul(distance(self.L - self.object_length), self.r0) #displace rays to lens. Accounts for object with depth
        rc1=circular_aperture(self.R, rl1) # cut off
        r2=np.matmul(sym_lens(self.L/2), rc1) #lens 1

        rl2=np.matmul(distance(3*self.L/2), r2) #displace rays to lens 2.
        rc2=circular_aperture(self.R, rl2) # cut off
        r3=np.matmul(sym_lens(self.L/3), rc2) #lens 2

        rp3=np.matmul(distance(self.L), r3) #displace rays to polariser
        
        β = β * np.pi/180
        rd3=np.matmul(polariser(β), rp3) #pass polariser rays to detector

        self.rf = rd3
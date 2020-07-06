"""
Example code for running on the RCD jupyter notebook cluster.

I recommend creating a new conda environment with:

numpy
scipy
matplotlib
multiprocessing

And also clone the repo into your home folder (or wherever) so you can load the code from the repo.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint,solve_ivp
from scipy.interpolate import RegularGridInterpolator
from time import time
from multiprocessing import Pool

%cd ~/turbulence_tracing/particle_tracking/
import particle_tracker as pt
import ray_transfer_matrix as rtm

##Sinusoidal test

N_V = 100
M_V = 2*N_V+1
ne_extent = 5.0e-3
ne_x = np.linspace(-ne_extent,ne_extent,M_V)
ne_y = np.linspace(-ne_extent,ne_extent,M_V)
ne_z = np.linspace(-ne_extent,ne_extent,M_V)

sin = pt.ElectronCube(ne_x,ne_y,ne_z,ne_extent)
s = 4e-3 #4 mm exponential growth rate
n_e0 = 2e17*1e6 #2e17 cm^-3 amplitude density
Ly= 1e-3 # 1 mm perturbation wavelength
sin.test_exponential_cos(n_e0=n_e0,Ly=Ly, s=s)
sin.calc_dndr()

num_processors = 8
Np=int(1e6)
#Create a pool of processors
p=Pool(processes = num_processors)

beam_size = 5e-3 # 5 mm
divergence = 0.05e-3 #0.05 mrad, realistic

## Create num_processors bundles of Np rays for solving separately
ss = [pt.init_beam(Np = Np, beam_size=5e-3, divergence = 0.05e-3, ne_extent = ne_extent) for i in range(num_processors)]
output = p.map(sin.solve, ss) # output of solve is the rays in (x, theta, y, phi) format

## Combine output
output=np.array(output) # output was a list
o=np.zeros((4,Np*num_processors)) #create a new array to put the results into
o[0,:]=output[:,0,:].flatten() #x
o[1,:]=output[:,1,:].flatten() #theta
o[2,:]=output[:,2,:].flatten() #y
o[3,:]=output[:,3,:].flatten() #phi

rf = o #I use rf as notation for the final rays everywhere else, so...
rf[0:4:2,:] *= 1e3 #convert to mm, a nicer unit

## Ray transfer matrix
b=rtm.BurdiscopeRays(rf)
sh=rtm.ShadowgraphyRays(rf)
sc=rtm.SchlierenRays(rf)

sh.solve(displacement=0)
sh.histogram(bin_scale=10)
sc.solve()
sc.histogram(bin_scale=10)
b.solve()
b.histogram(bin_scale=10)

## Plot results
fig, axs = plt.subplots(1,3,figsize=(6.67, 1.7))

cm='gray'
clim=[0,3000]

sc.plot(axs[0], clim=clim, cmap=cm)
sh.plot(axs[1], clim=clim, cmap=cm)
b.plot(axs[2], clim=[0,20000], cmap=cm)

for ax in axs:
    ax.axis('off')
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=None)
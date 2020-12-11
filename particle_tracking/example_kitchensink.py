import numpy as np
import particle_tracker as pt
import matplotlib.pyplot as plt

# Kitchen sink test
Np  = 50000
N_V = 100
M_V = 2*N_V+1
ne_extent = 5.0e-3
ne_x = np.linspace(-ne_extent,ne_extent,M_V)
ne_y = np.linspace(-ne_extent,ne_extent,M_V)
ne_z = np.linspace(-ne_extent,ne_extent,M_V)

test = pt.ElectronCube(ne_x,ne_y,ne_z,ne_extent,B_on=True,inv_brems=True,phaseshift=True)

xx,yy,zz = np.meshgrid(ne_x,ne_y,ne_z,indexing='ij')

Te = np.ones_like(xx)
B_shape = list(xx.shape)
B_shape.append(3)
B  = np.zeros(B_shape)
B[:,:,:,2] = xx/ne_extent

test.test_linear_cos(n_e0=1e24,Ly=2.0e-3)
test.external_Te(Te)
test.external_Z(1.0)
test.external_B(B)
test.calc_dndr()
test.set_up_interps()
test.clear_memory()

## Beam properties
beam_size = 5e-3 # 5 mm
divergence = 0.05e-3 #0.05 mrad, realistic

## Initialise laser beam
ss = pt.init_beam(Np = Np, beam_size=beam_size, divergence = divergence, ne_extent = ne_extent)
## Propogate rays through ne_cube
rf = test.solve(ss) # output of solve is the rays in (x, theta, y, phi) format
Jf = test.Jf
# Convert to mm, a nicer unit
rf[0:4:2,:] *= 1e3 

rx = rf[0,:]
ry = rf[2,:]
amp = np.sqrt(np.abs(Jf[0,:]**2+Jf[1,:]**2))
aEy = np.arctan(np.real(Jf[0,:]/Jf[1,:]))
REy = np.cos(np.angle(Jf[1,:]))

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

im1 = ax1.tripcolor(rx,ry,amp)
ax1.set_aspect('equal')
ax1.set_title(r"$\sqrt{E_x^2+E_y^2}$")
fig.colorbar(im1,ax=ax1,orientation='horizontal')

im2 = ax2.tripcolor(rx,ry,aEy,cmap='coolwarm',vmin=-np.amax(aEy),vmax=np.amax(aEy))
ax2.set_aspect('equal')
ax2.set_title(r"atan$\left(\frac{E_x}{E_y}\right)$")
fig.colorbar(im2,ax=ax2,orientation='horizontal')

im3 = ax3.tripcolor(rx,ry,REy,cmap='Greys')
ax3.set_aspect('equal')
ax3.set_title(r"cos(arg$\left(E_y\right)$)")
fig.colorbar(im3,ax=ax3,orientation='horizontal')

plt.show()
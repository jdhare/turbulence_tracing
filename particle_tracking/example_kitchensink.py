import numpy as np
import particle_tracker as pt
import matplotlib.pyplot as plt
import vtk
from vtk.util import numpy_support as vtk_np

def pvti_readin(filename):
	'''
	Reads in data from pvti with filename, use this to read in electron number density data

	'''

	reader = vtk.vtkXMLPImageDataReader()
	reader.SetFileName(filename)
	reader.Update()

	data = reader.GetOutput()
	dim = data.GetDimensions()
	spacing = np.array(data.GetSpacing())

	v = vtk_np.vtk_to_numpy(data.GetCellData().GetArray(0))
	n_comp = data.GetCellData().GetArray(0).GetNumberOfComponents()
	
	vec = [int(i-1) for i in dim]

	if(n_comp > 1):
		vec.append(n_comp)

	if(n_comp > 2):
		img = v.reshape(vec,order="F")[0:dim[0]-1,0:dim[1]-1,0:dim[2]-1,:]
	else:
		img = v.reshape(vec,order="F")[0:dim[0]-1,0:dim[1]-1,0:dim[2]-1]

	dim = img.shape

	return img,dim,spacing

# Kitchen sink test
Np  = int(1e5)

## Load in data
print("Loading data...")
vti_file = "./data/x08_rnec-400.pvti"
rnec,dim,spacing = pvti_readin(vti_file)
vti_file = "./data/x08_Te-300.pvti"
Te,dim,spacing = pvti_readin(vti_file)
vti_file = "./data/x08_Bvec-400.pvti"
Bvec,dim,spacing = pvti_readin(vti_file)
# Probing direction is along y
M_V2 = dim[2]//2
ne_extent2 = 2*spacing[2]*((M_V2-1)/2)
ne_z = np.linspace(-ne_extent2,ne_extent2,M_V2)
M_V1 = 80
ne_extent1 = 2*spacing[0]*((M_V1-1)/2)
ne_x = np.linspace(-ne_extent1,ne_extent1,M_V1)
M_V = dim[1]//2
ne_extent = 2*spacing[1]*((M_V-1)/2)
ne_y = np.linspace(-ne_extent,ne_extent,M_V)

rnec = rnec[-2*M_V1::2,::2,::2]
Te   = Te[-2*M_V1::2,::2,::2]
Bvec = Bvec[-2*M_V1::2,::2,::2,:]

print("Data loaded...")

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.set_title(r"$n_e$")
im1 = ax1.imshow(rnec[:,rnec.shape[1]//2,:])
fig.colorbar(im1,ax=ax1,orientation='horizontal')
np.savetxt('ne_slice.dat',rnec[:,rnec.shape[1]//2,:])

ax2.set_title(r"$B_z$")
im2 = ax2.imshow(Bvec[:,rnec.shape[1]//2,:,1],cmap='jet',vmin=-5)
fig.colorbar(im2,ax=ax2,orientation='horizontal')
np.savetxt('Bz_slice.dat',Bvec[:,Bvec.shape[1]//2,:,1])

ax3.set_title(r"$n_eL$")
neL = np.sum(2*spacing[1]*rnec[:,:,:],axis=1)
im3 = ax3.imshow(neL)
fig.colorbar(im3,ax=ax3,orientation='horizontal')
np.savetxt('neL.dat',neL)

ax4.set_title(r"$n_eB_zL/n_eL$")
neBL = np.sum(2*spacing[1]*rnec[:,:,:]*Bvec[:,:,:,1],axis=1)
im4 = ax4.imshow(neBL/neL,cmap='jet')
fig.colorbar(im4,ax=ax4,orientation='horizontal')
np.savetxt('neBL.dat',neBL)

plt.show()

from sys import exit
exit()


test = pt.ElectronCube(ne_x,ne_y,ne_z,ne_extent,B_on=True,inv_brems=True,phaseshift=True, probing_direction = 'y')

test.external_ne(rnec)
test.external_Te(Te)
test.external_Z(1.0)
test.external_B(Bvec)
test.calc_dndr()
test.set_up_interps()
test.clear_memory()
rnec = None
Bvec = None

## Beam properties
beam_size  = 10e-3 # 10 mm
divergence = 0.05e-3 #0.05 mrad, realistic

## Initialise laser beam
ss = pt.init_beam(Np = Np, beam_size=beam_size, divergence = divergence, ne_extent = ne_extent, probing_direction = 'y')
## Propogate rays through ne_cube
rf = test.solve(ss) # output of solve is the rays in (x, theta, y, phi) format
Jf = test.Jf
# Convert to mm, a nicer unit
rf[0:4:2,:] *= 1e3 

rx = rf[0,:]
ry = rf[2,:]

amp = np.sqrt(np.abs(Jf[0,:]**2+Jf[1,:]**2))
aEy = np.arctan(np.real(Jf[0,:]/Jf[1,:]))
REy = np.cos(np.angle(Jf[1,:]))**2

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

bin_edges = 1e3*beam_size*np.linspace(-1.0,1.0,100)
amp_hist,xedges,yedges = np.histogram2d(rx,ry,bins=bin_edges,weights=amp)

im1 = ax1.imshow(amp_hist,extent=[bin_edges[0],bin_edges[-1],bin_edges[0],bin_edges[-1]])
ax1.set_xlim([bin_edges[0],bin_edges[-1]])
ax1.set_ylim([bin_edges[0],bin_edges[-1]])
ax1.set_aspect('equal')
ax1.set_title(r"$\sqrt{E_x^2+E_y^2}$")
fig.colorbar(im1,ax=ax1,orientation='horizontal')

aEy_hist,xedges,yedges = np.histogram2d(rx,ry,bins=bin_edges,weights=aEy)
# Normalise
aEy_hist /= amp_hist

im2 = ax2.imshow(aEy_hist,cmap='coolwarm',extent=[bin_edges[0],bin_edges[-1],bin_edges[0],bin_edges[-1]])
ax2.set_aspect('equal')
ax2.set_title(r"atan$\left(\frac{E_x}{E_y}\right)$")
fig.colorbar(im2,ax=ax2,orientation='horizontal')

REy_hist,xedges,yedges = np.histogram2d(rx,ry,bins=bin_edges,weights=REy)
# Normalise
REy_hist /= amp_hist

im3 = ax3.imshow(REy_hist,cmap='Greys',extent=[bin_edges[0],bin_edges[-1],bin_edges[0],bin_edges[-1]])
ax3.set_aspect('equal')
ax3.set_title(r"cos$^2$(arg$\left(E_y\right)$)")
fig.colorbar(im3,ax=ax3,orientation='horizontal')

fig.savefig("DannyExpRayTrace.png")

plt.show()
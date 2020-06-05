"""
Author: Stefano Merlini
Created: 01/06/2020

"""


# EXAMPLE 
# General example on how to run the GaussianFieldGenerator
# ----------------------------------------------------------
# README - REQUIREMENT
# The file example.py required the following dependencies:
#  - Numpy
#  - Matplotlib
#  - turbogen.py = This file contains the methods to compute the turbulence field.
#  - cmpspec.py = This file contains the method to compute the power spectrum given a specific field


#  ____  _  _   __   _  _  ____  __    ____ 
# (  __)( \/ ) / _\ ( \/ )(  _ \(  )  (  __)
#  ) _)  )  ( /    \/ \/ \ ) __// (_/\ ) _) 
# (____)(_/\_)\_/\_/\_)(_/(__)  \____/(____)
# this example generates a 3D Gaussian Field density distribution with a classic kolmogorov spectrum


# import library

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm

# import dependencies

import cmpspec
import turboGen as tg

#  ____  ____  ____  ___  ____  ____  _  _  _  _  
# / ___)(  _ \(  __)/ __)(_  _)(  _ \/ )( \( \/ ) 
# \___ \ ) __/ ) _)( (__   )(   )   /) \/ (/ \/ \ 
# (____/(__)  (____)\___) (__) (__\_)\____/\_)(_/ 
# this is the standard kolmogorov spectrum -5/3
# Here, I define the given power spectrum

class k41:
	def evaluate(self, k):
		a = -5/3
		espec = pow(k,a)
		return espec



#  ____  ____  __ _  ____  __  ____  _  _    ____  __     __  __  ____ 
# (    \(  __)(  ( \/ ___)(  )(_  _)( \/ )  (_  _)/  \   (  )/  \(  _ \
#  ) D ( ) _) /    /\___ \ )(   )(   )  /     )( (  O )   )((  O ))   /
# (____/(____)\_)__)(____/(__) (__) (__/     (__) \__/   (__)\__/(__\_)
# This function convert the 3D electron density field in to 3D Index of Refraction (IOR) Field
# Required parameters:
# - laser wavelength
# - electron density


def computeIOR(n_e):
	lambda_laser = 1.06 # [um] laser wavelength 
	n_cr = (1.12 * 1e21)/lambda_laser**2 # Calculate the critical density
	IOR = np.sqrt((1 - n_e/n_cr)) # calculate the index of refraction
	return IOR


#  ____      ____    ____  __  ____  __    ____ 
# ( __ \ ___(    \  (  __)(  )(  __)(  )  (    \
#  (__ ((___)) D (   ) _)  )(  ) _) / (_/\ ) D (
# (____/    (____/  (__)  (__)(____)\____/(____/
# First case. let's assume 3-D
# GRID RESOLUTION nx, ny, nz
# Changing the grid resolution will change the maximum limit that the code resolves the spectrum.
# As you increase the number of cells (e.g. 128,256,...) the code is able to resolve higher wavenumbers but it takes LONGER TIME!
nx = 64
ny = 64
nz = 64
# DOMAIN DEFINITION 
lx = 1
ly = 1
lz = 1
# NUMBER OF MODES
# Increasing the number of modes, the accuracy should increases but it takes LONGER TIME!
nmodes = 100
# SPECIFY THE SPECTRUM THAT WE WANT
# right now only kolmogorov -5/3
inputspec = 'k41'
# PATH folder
pathfolder = './Output'
filename3 = inputspec + '_' + str(nx) + '_' + str(ny) + '_' + str(nz) + '_' + str(nmodes) + '_modes'
# CALL CLASS SPECTRUM
whichspect = k41().evaluate
# Defining the smallest wavenumber represented by this spectrum
wn1 = min(2.0*np.pi/lx, min (2.0*np.pi/ly, 2.0*np.pi/lz))
# Summary of the user input
print("SUMMARY OF THE USER INPUTs:")
print("---------------------------")
print("Type of generator: 3D")
print("Spectrum: ", inputspec)
print("Domain size: ", lx, ly, lz)
print("Grid Resolution", nx, ny, nz)
print("Fourier accuracy (modes): ", nmodes)
#
# STARTING...
# Smallest step size
dx = lx/nx
dy = ly/ny
dz = lz/nz
t0 = time.time() # initial time
# --------------------------------------------------
# Run the function TurboGenerator
# --------------------------------------------------
r_xyz = tg.gaussian3Dcos(lx, ly, lz, nx, ny, nz, nmodes, wn1, whichspect)
t1 = time.time() # final time
computing_time = t1 - t0
## Rescaling to a seamless electron density field
r_xyz = (r_xyz - np.min(r_xyz))*1e19/(np.max(r_xyz)-np.min(r_xyz))
#
print("It took me ", computing_time, "to generate the 3D turbulence.")
# COMPUTE THE POWER SPECTRUM OF THE 3-D FIELD
# verify that the generated density field fit the spectrum
knyquist3D, wavenumbers3D, psdspec3D = cmpspec.compute3Dspectrum(r_xyz, lx, ly, lz, False)
# save the generated spectrum to a text file for later post processing
np.savetxt(pathfolder + '/3D_psdspec_' + filename3 + '.txt', np.transpose([wavenumbers3D, psdspec3D]))
#
#
# EXPORT generated density field
np.savez(pathfolder + '/3D_Density_Field_' + filename3 + '.npz', r_xyz)
#
print('mean field value: ', np.mean(r_xyz))
print('max field value: ', np.max(r_xyz))
print('min field value: ', np.min(r_xyz))
# Convert the 3D electron density field into the 3D Index of Refraction field
IOR = computeIOR(r_xyz)
# EXPORT the IOR field
np.savez(pathfolder + '/3D_IOR_' + filename3 + '.npz', IOR)
#
#  ____  __     __  ____    ____  ____  ____  _  _  __   ____  ____  
# (  _ \(  )   /  \(_  _)  (  _ \(  __)/ ___)/ )( \(  ) (_  _)/ ___) 
#  ) __// (_/\(  O ) )(     )   / ) _) \___ \) \/ (/ (_/\ )(  \___ \ 
# (__)  \____/ \__/ (__)   (__\_)(____)(____/\____/\____/(__) (____/ 
# PLOT THE 1D, 2D, 3D FIELD IN REAL DOMAIN AND RELATIVE POWER SPECTRUM
# ---------------------------------------------------------------------

# Plot 3D-FIELD
plt.rc("font", size=10, family='serif')
fig = plt.figure(figsize=(3.5, 2.8), dpi=300, constrained_layout=True)
X, Y = np.meshgrid(np.arange(0,lx,dx),np.arange(0,ly,dy))
cp = plt.contourf(X, Y, r_xyz[:,:,10], cmap = matplotlib.cm.get_cmap('plasma'))
cb = plt.colorbar(cp)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Meter [m]')
plt.ylabel('Meter [m]')
cb.set_label(r'$ \rho(x,y,z) $', rotation=270)
plt.grid()
fig.savefig(pathfolder + '/3D_field_slice_' + str(10) + filename3 + '.pdf')
plt.show()


# PLOT 3-D FIELD SPECTRUM
# Range of wavenumbers from minimum wavenumber wn1 up to 2000
plt.rc("font", size=10, family='serif')
fig = plt.figure(figsize=(3.5, 2.8), dpi=300, constrained_layout=True)
wnn = np.arange(wn1, 1000)
l1, = plt.loglog(wnn, 1e18 **2 * whichspect(wnn), 'k-', label='input')
l2, = plt.loglog(wavenumbers3D[1:6], psdspec3D[1:6], 'bo--', markersize=3, markerfacecolor='w', markevery=1, label='computed')
plt.loglog(wavenumbers3D[5:], psdspec3D[5:], 'bo--', markersize=3, markerfacecolor='w', markevery=4)
# plt.axis([3, 2000, 1e-7, 1e-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=knyquist3D, linestyle='--', color='black')
plt.xlabel('$\kappa$ [1/m]')
plt.ylabel('$PSD(\kappa)$')
plt.grid()
plt.legend()
fig.savefig(pathfolder + '/3D_psdspec_' + filename3 + '.pdf')
plt.show()
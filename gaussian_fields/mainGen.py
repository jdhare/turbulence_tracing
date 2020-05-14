# Main program - Version 1
# This is an example of how to use the library turboGen.py
# and cmpspec.py
# GENERATING 1D-2D-3D GAUSSIAN STOCHASTIC FIELD WITH A GIVEN POWER SPECTRUM AS INPUT

"""
Author: Stefano Merlini
Created: 14/05/2020
"""


#  ____  _  _   __   _  _  ____  __    ____ 
# (  __)( \/ ) / _\ ( \/ )(  _ \(  )  (  __)
#  ) _)  )  ( /    \/ \/ \ ) __// (_/\ ) _) 
# (____)(_/\_)\_/\_/\_)(_/(__)  \____/(____)


# import library

import numpy as np
import turboGen as tg
import time
import matplotlib.pyplot as plt
import cmpspec
import matplotlib.cm
from mpl_toolkits.mplot3d import Axes3D


#  ____  ____  ____  ___  ____  ____  _  _  _  _  
# / ___)(  _ \(  __)/ __)(_  _)(  _ \/ )( \( \/ ) 
# \___ \ ) __/ ) _)( (__   )(   )   /) \/ (/ \/ \ 
# (____/(__)  (____)\___) (__) (__\_)\____/\_)(_/ 
# this is the standard kolmogorov spectrum -5/3
#
class k41:
    def evaluate(self, k):
        espec = pow(k,-5.0/3.0)
        return espec


#   __      ____    ____  __  ____  __    ____ 
#  /  \ ___(    \  (  __)(  )(  __)(  )  (    \
# (_/ /(___)) D (   ) _)  )(  ) _) / (_/\ ) D (
#  (__)    (____/  (__)  (__)(____)\____/(____/
# First case. let's assume 1-D
# GRID RESOLUTION nx
nx = 64
# DOMAIN DEFINITION 
lx = 1 
# NUMBER OF MODES
nmodes = 100
# SPECIFY THE SPECTRUM THAT WE WANT
# right now only kolmogorov -5/3
inputspec = 'k41'
# PATH folder
pathfolder = './Output'
filename1 = inputspec + '_' + str(nx) + '_' + str(nmodes) + '_modes'
# CALL CLASS SPECTRUM
whichspect = k41().evaluate
# Defining the smallest wavenumber represented by this spectrum
wn1 = 2.0*np.pi/lx
# Summary of the user input
print("SUMMARY OF THE USER INPUTs:")
print("---------------------------")
print("Type of generator: 1D")
print("Spectrum: ", inputspec)
print("Domain size: ", lx)
print("Grid Resolution", nx)
print("Fourier accuracy (modes): ", nmodes)
#
# STARTING...
# Smallest step size
dx = lx/nx
t0 = time.time() # initial time
# --------------------------------------------------
# Run the function TurboGenerator
# --------------------------------------------------
r_x = tg.gaussian1Dcos(lx, nx, nmodes, wn1, whichspect)
#
t1 = time.time() # final time
computing_time = t1 - t0
#
print("It took me ", computing_time, "to generate the 1D turbulence.")
# COMPUTE THE POWER SPECTRUM OF THE 1-D FIELD
# verify that the generated velocities fit the spectrum
knyquist1D, wavenumbers1D, tkespec1D = cmpspec.compute1Dspectrum(r_x, lx, False)
# save the generated spectrum to a text file for later post processing
np.savetxt(pathfolder + '/1D_tkespec_' + filename1 + '.txt', np.transpose([wavenumbers1D, tkespec1D]))






#  ____      ____    ____  __  ____  __    ____ 
# (___ \ ___(    \  (  __)(  )(  __)(  )  (    \
#  / __/(___)) D (   ) _)  )(  ) _) / (_/\ ) D (
# (____)    (____/  (__)  (__)(____)\____/(____/
# First case. let's assume 2-D
# GRID RESOLUTION nx, ny
nx = 64
ny = 64
# DOMAIN DEFINITION 
lx = 1 
ly = 1
# NUMBER OF MODES
nmodes = 100
# SPECIFY THE SPECTRUM THAT WE WANT
# right now only kolmogorov -5/3
inputspec = 'k41'
# PATH folder
pathfolder = './Output'
filename2 = inputspec + '_' + str(nx) + '_' + str(ny) + '_' + str(nmodes) + '_modes'
# CALL CLASS SPECTRUM
whichspect = k41().evaluate
# Defining the smallest wavenumber represented by this spectrum
wn1 = min(2.0*np.pi/lx, 2.0*np.pi/ly)
# Summary of the user input
print("SUMMARY OF THE USER INPUTs:")
print("---------------------------")
print("Type of generator: 2D")
print("Spectrum: ", inputspec)
print("Domain size: ", lx, ly)
print("Grid Resolution", nx, ny)
print("Fourier accuracy (modes): ", nmodes)
#
# STARTING...
# Smallest step size
dx = lx/nx
dy = ly/ny
t0 = time.time() # initial time
# --------------------------------------------------
# Run the function TurboGenerator
# --------------------------------------------------
r_xy = tg.gaussian2Dcos(lx, ly, nx, ny, nmodes, wn1, whichspect)
t1 = time.time() # final time
computing_time = t1 - t0
print("It took me ", computing_time, "to generate the 2D turbulence.")
# COMPUTE THE POWER SPECTRUM OF THE 2-D FIELD
# verify that the generated velocities fit the spectrum
knyquist2D, wavenumbers2D, tkespec2D = cmpspec.compute2Dspectrum(r_xy, lx, ly, False)
# save the generated spectrum to a text file for later post processing
np.savetxt(pathfolder + '/2D_tkespec_' + filename2 + '.txt', np.transpose([wavenumbers2D, tkespec2D]))








#  ____      ____    ____  __  ____  __    ____ 
# ( __ \ ___(    \  (  __)(  )(  __)(  )  (    \
#  (__ ((___)) D (   ) _)  )(  ) _) / (_/\ ) D (
# (____/    (____/  (__)  (__)(____)\____/(____/
# First case. let's assume 3-D
# GRID RESOLUTION nx, ny, nz
nx = 64
ny = 64
nz = 64
# DOMAIN DEFINITION 
lx = 1 
ly = 1
lz = 1
# NUMBER OF MODES
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
wn1 = min(2.0*np.pi/lx, 2.0*np.pi/ly)
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
print("It took me ", computing_time, "to generate the 3D turbulence.")
# COMPUTE THE POWER SPECTRUM OF THE 2-D FIELD
# verify that the generated velocities fit the spectrum
knyquist3D, wavenumbers3D, tkespec3D = cmpspec.compute3Dspectrum(r_xyz, lx, ly, lz, False)
# save the generated spectrum to a text file for later post processing
np.savetxt(pathfolder + '/3D_tkespec_' + filename3 + '.txt', np.transpose([wavenumbers3D, tkespec3D]))







#  ____  __     __  ____    ____  ____  ____  _  _  __   ____  ____  
# (  _ \(  )   /  \(_  _)  (  _ \(  __)/ ___)/ )( \(  ) (_  _)/ ___) 
#  ) __// (_/\(  O ) )(     )   / ) _) \___ \) \/ (/ (_/\ )(  \___ \ 
# (__)  \____/ \__/ (__)   (__\_)(____)(____/\____/\____/(__) (____/ 
# PLOT THE 1D, 2D, 3D FIELD IN REAL DOMAIN AND RELATIVE POWER SPECTRUM
# ---------------------------------------------------------------------

# Plot 1D-FIELD
plt.rc("font", size=10, family='serif')
fig = plt.figure(figsize=(3.5, 2.8), dpi=200, constrained_layout=True)
X = np.arange(0,lx,dx)
plt.plot(X,r_x, 'k-', label='computed')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Meter [m]')
plt.ylabel(r'$ \rho(x) $')
plt.legend()
plt.grid()
fig.savefig(pathfolder + '/1D_field_' + filename1 + '.pdf')

# Plot 2D-FIELD
plt.rc("font", size=10, family='serif')
fig = plt.figure(figsize=(3.5, 2.8), dpi=200, constrained_layout=True)
X, Y = np.meshgrid(np.arange(0,lx,dx),np.arange(0,ly,dy))
cp = plt.contourf(X, Y, r_xy, cmap = matplotlib.cm.get_cmap('plasma'))
cb = plt.colorbar(cp)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Meter [m]')
plt.ylabel('Meter [m]')
cb.set_label(r'$ \rho(x,y) $', rotation=270)
plt.grid()
fig.savefig(pathfolder + '/2D_field_' + filename2 + '.pdf')
plt.show()

# Plot 3D-FIELD
plt.rc("font", size=10, family='serif')
fig = plt.figure(figsize=(3.5, 2.8), dpi=200, constrained_layout=True)
# X, Y, Z = np.meshgrid(np.arange(0,lx,dx),np.arange(0,ly,dy),np.arange(0,lz,dz))
X, Y = np.meshgrid(np.arange(0,lx,dx),np.arange(0,ly,dy))
cp = plt.contourf(X, Y, r_xyz[:,:,1], cmap = matplotlib.cm.get_cmap('plasma'))
cb = plt.colorbar(cp)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Meter [m]')
plt.ylabel('Meter [m]')
cb.set_label(r'$ \rho(x,y) $', rotation=270)
plt.grid()
fig.savefig(pathfolder + '/3D_field_slice_' + filename3 + '.pdf')
plt.show()

# --------------------------------------------------------------
# PLOT NUMERICAL AND THEORICAL POWER SPECTRUM
# Plot in log-log 
# --------------------------------------------------------------

# PLOT 1-D FIELD SPECTRUM
# Range of wavenumbers from minimum wavenumber wn1 up to 2000
plt.rc("font", size=10, family='serif')
fig = plt.figure(figsize=(3.5, 2.8), dpi=200, constrained_layout=True)
wnn = np.arange(wn1, 2000)
l1, = plt.loglog(wnn, whichspect(wnn), 'k-', label='input')
l2, = plt.loglog(wavenumbers1D[1:6], tkespec1D[1:6], 'bo--', markersize=3, markerfacecolor='w', markevery=1, label='computed')
plt.loglog(wavenumbers1D[5:], tkespec1D[5:], 'bo--', markersize=3, markerfacecolor='w', markevery=4)
plt.axis([3, 10000, 1e-7, 1e-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=knyquist1D, linestyle='--', color='black')
plt.xlabel('$\kappa$ [1/m]')
plt.ylabel('$E(\kappa)$ [m$^3$/s$^2$]')
plt.grid()
plt.legend()
fig.savefig(pathfolder + '/1D_tkespec_' + filename1 + '.pdf')
plt.show()


# PLOT 2-D FIELD SPECTRUM
# Range of wavenumbers from minimum wavenumber wn1 up to 2000
plt.rc("font", size=10, family='serif')
fig = plt.figure(figsize=(3.5, 2.8), dpi=200, constrained_layout=True)
wnn = np.arange(wn1, 2000)
l1, = plt.loglog(wnn, whichspect(wnn), 'k-', label='input')
l2, = plt.loglog(wavenumbers2D[1:6], tkespec2D[1:6], 'bo--', markersize=3, markerfacecolor='w', markevery=1, label='computed')
plt.loglog(wavenumbers2D[5:], tkespec2D[5:], 'bo--', markersize=3, markerfacecolor='w', markevery=4)
plt.axis([3, 10000, 1e-7, 1e-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=knyquist2D, linestyle='--', color='black')
plt.xlabel('$\kappa$ [1/m]')
plt.ylabel('$E(\kappa)$ [m$^3$/s$^2$]')
plt.grid()
plt.legend()
fig.savefig(pathfolder + '/2D_tkespec_' + filename2 + '.pdf')
plt.show()


# PLOT 3-D FIELD SPECTRUM
# Range of wavenumbers from minimum wavenumber wn1 up to 2000
plt.rc("font", size=10, family='serif')
fig = plt.figure(figsize=(3.5, 2.8), dpi=200, constrained_layout=True)
wnn = np.arange(wn1, 2000)
l1, = plt.loglog(wnn, whichspect(wnn), 'k-', label='input')
l2, = plt.loglog(wavenumbers3D[1:6], tkespec3D[1:6], 'bo--', markersize=3, markerfacecolor='w', markevery=1, label='computed')
plt.loglog(wavenumbers3D[5:], tkespec3D[5:], 'bo--', markersize=3, markerfacecolor='w', markevery=4)
plt.axis([3, 10000, 1e-7, 1e-1])
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axvline(x=knyquist3D, linestyle='--', color='black')
plt.xlabel('$\kappa$ [1/m]')
plt.ylabel('$E(\kappa)$ [m$^3$/s$^2$]')
plt.grid()
plt.legend()
fig.savefig(pathfolder + '/3D_tkespec_' + filename3 + '.pdf')
plt.show()






#  ____      ____    ____  __     __  ____    ____  __  ____  __    ____ 
# ( __ \ ___(    \  (  _ \(  )   /  \(_  _)  (  __)(  )(  __)(  )  (    \
#  (__ ((___)) D (   ) __// (_/\(  O ) )(     ) _)  )(  ) _) / (_/\ ) D (
# (____/    (____/  (__)  \____/ \__/ (__)   (__)  (__)(____)\____/(____/


# plt.rc("font", size=10, family='serif')
# fig = plt.figure(figsize=(3.5, 2.8), dpi=200, constrained_layout=True)
# ax = fig.gca(projection='3d')

# X, Y = np.meshgrid(np.arange(0,lx,dx),np.arange(0,ly,dy))

# cset = [[],[],[]]
# # this is the example that worked for you:

# Z = r_xyz[0,:,:]

# cset[0] = ax.contourf(Z, X, Y, zdir = 'x', offset = , cmap = matplotlib.cm.get_cmap('plasma'))
# # cset[0] = ax.contourf(X, Y, Z, zdir = 'y', offset = , levels=np.linspace(np.min(Z),np.max(Z),30), cmap = matplotlib.cm.get_cmap('plasma'))

# # now, for the x-constant face, assign the contour to the x-plot-variable:
# # cset[1] = ax.contourf(X, Y, r_xyz[:,:,31], levels=np.linspace(np.min(r_xyz[:,:,31]),np.max(r_xyz[:,:,31]),30), cmap = matplotlib.cm.get_cmap('plasma'))

# # likewise, for the y-constant face, assign the contour to the y-plot-variable:
# # cset[2] = ax.contourf(X, Y, r_xyz[:,:,63] , levels=np.linspace(np.min(r_xyz[:,:,63]),np.max(r_xyz[:,:,63]),30), cmap = matplotlib.cm.get_cmap('plasma'))

# # # setting 3D-axis-limits:    
# # ax.set_xlim3d(0,nx)
# # ax.set_ylim3d(0,ny)
# # ax.set_zlim3d(0,nz)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.grid()
# fig.savefig(pathfolder + '/3D_field_' + filename3 + '.pdf')
# plt.show()



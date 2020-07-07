"""
Example code for running on hpc using mpiexec

On top of usual requirements to run the particle tracking code it requires:

pickle
mpi4py

MPI allows multiple nodes to be used while multiprocessing does not
I also have more experience with MPI so know how to keep the memory usage low

Example PBS queue submission script to use 48 cores on 1 node with 2e7 rays per processor,
The resulting synthetic diagnostics are saved to the directory output

#!/bin/sh
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=48:mpiprocs=48:mem=128gb
#PBS -j oe
cd [HOME]/turbulence_tracing/particle_tracking

module load anaconda3/personal

mpiexec python example_MPI.py 2e7 ./output/

Using this set up took 2 hours of computing time for a total of 9.8e8 rays

There are a couple of variables which can be changed depending on your computing set up:

Np_ray_split - the number of rays per "bundle" considered, if you request number of rays per processor Np > Np_ray_split,
then the task is split such that bundles of Np_ray_split are considered until Np rays have been calculated,
this variable can be changed to fit into your memory budget. For 48 processors, 5e5 rays uses around 60 GB

bin_scale - this is the ratio of the computational to experimental pixel count, this can be reduced when more rays are considered

Outputs are pickled to keep their object structure, however information on the rays is not saved

"""

import numpy as np
from time import time
from mpi4py import MPI
import pickle
import sys
import gc
import particle_tracker as pt
import ray_transfer_matrix as rtm

def full_system_solve(Np,beam_size,divergence,ne_extent,ne_cube):
	'''
	Main function called by all processors, considers Np rays traversing the electron density volume ne_cube

	beam_size and divergence set the initial properties of the laser beam, ne_extent contains the size of the electron density volume

	'''

	## Initialise laser beam
	ss = pt.init_beam(Np = Np, beam_size=beam_size, divergence = divergence, ne_extent = ne_extent)
	## Propogate rays through ne_cube
	rf = ne_cube.solve(ss) # output of solve is the rays in (x, theta, y, phi) format
	# Save memory by deleting initial ray positions
	del ss
	ne_cube.clear_memory()
	# Convert to mm, a nicer unit
	rf[0:4:2,:] *= 1e3 

	## Ray transfer matrix
	b=rtm.BurdiscopeRays(rf)
	b.solve()
	b.histogram(bin_scale=1,clear_mem=True)

	sh=rtm.ShadowgraphyRays(rf)
	sh.solve(displacement=0)
	sh.histogram(bin_scale=1,clear_mem=True)

	sc=rtm.SchlierenRays(rf)
	sc.solve()
	sc.histogram(bin_scale=1,clear_mem=True)

	return sc,sh,b

## Initialise the MPI
comm = MPI.COMM_WORLD
## Each processor is allocated a number (rank)
rank = comm.Get_rank()
## Number of rays at which the computation is split to save memory
Np_ray_split = int(5e5)
## Number of processors being used
num_processors = comm.size
## Takes number of rays per processors as command line argument
Np=int(float(sys.argv[1]))
## Takes output directory as command line argument
output_dir = sys.argv[2]
if(rank == 0):
	print("Number of processors: %s"%num_processors)
	print("Rays per processors: %s"%Np)

## Sinusoidal test
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
sin.clear_memory()

## Beam properties
beam_size = 5e-3 # 5 mm
divergence = 0.05e-3 #0.05 mrad, realistic

# May trip memory limit, so split up calculation
if(Np > Np_ray_split):
	number_of_splits = Np//Np_ray_split
	remaining_rays   = Np%Np_ray_split
	if(rank == 0):
		print("Splitting to %d ray bundles"%number_of_splits)
		# Solve subsystem once to initialise the diagnostics
		print("Solve for remainder: %d"%remaining_rays)
	# Remaining_rays could be zero, this doesn't matter
	sc,sh,b = full_system_solve(remaining_rays,beam_size,divergence,ne_extent,sin)
	# Iterate over remaining ray bundles
	for i in range(number_of_splits):
		if(rank == 0):
			print("%d of %d"%(i+1,number_of_splits))
		# Solve subsystem
		sc_split,sh_split,b_split = full_system_solve(Np_ray_split,beam_size,divergence,ne_extent,sin)
		# Force garbage collection - this may be unnecessary but better safe than sorry
		gc.collect()
		# Add in results from splitting
		sc.H += sc_split.H
		sh.H += sh_split.H
		b.H  += b_split.H
else:
	print("Solving whole system...")
	sc,sh,b = full_system_solve(Np,beam_size,divergence,ne_extent,sin)

## Now each processor has calculated Schlieren, Shadowgraphy and Burdiscope results
## Must sum pixel arrays and give to root processor

# Collect and sum all results and store on only root processor
sc.H = comm.reduce(sc.H,root=0,op=MPI.SUM)
sh.H = comm.reduce(sh.H,root=0,op=MPI.SUM)
b.H  = comm.reduce(b.H,root=0,op=MPI.SUM)

# Perform file saves on root processor only
if(rank == 0):

	# Save diagnostics as a pickle
	# For Schlieren
	filehandler = open(output_dir+"Schlieren.pkl","wb")
	pickle.dump(sc,filehandler)
	filehandler.close()
	# For Shadowgraphy
	filehandler = open(output_dir+"Shadowgraphy.pkl","wb")
	pickle.dump(sh,filehandler)
	filehandler.close()
	# For Burdiscope
	filehandler = open(output_dir+"Burdiscope.pkl","wb")
	pickle.dump(b,filehandler)
	filehandler.close()

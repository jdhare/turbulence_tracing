"""
Example code for plotting data produced by hpc runs

"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import ray_transfer_matrix as rtm

sc = pickle.load( open( "./Schlieren.pkl", "rb" ) )
sh = pickle.load( open( "./Shadowgraphy.pkl", "rb" ) )
b  = pickle.load( open( "./Burdiscope.pkl", "rb" ) )

print("Number of rays at Burdiscope: %d"%(int(np.sum(b.H))))

## Plot results
fig, axs = plt.subplots(1,3,figsize=(6.67, 1.7),dpi=200)

cm='gray'
clim=[0,300]

sc.plot(axs[0],clim=clim, cmap=cm)
sh.plot(axs[1],clim=clim, cmap=cm)
b.plot(axs[2],clim=[0,2000], cmap=cm)

for ax in axs:
    ax.axis('off')
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=None)

fig.savefig("mp_plot.pdf")
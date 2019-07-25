#Ava Nederlander Code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import sys
import corner
import csv
import pylab


#ndim=6
#old_df = pd.read_csv("chain_20steps_new8params.csv", skiprows=range(1,1121))
df = pd.read_csv("chain_20steps_new8params.csv")
#df = pd.read_csv("chain_20steps_new8params.csv")
#x = [-np.inf]

#df = old_df[~old_df.lnprob.isin(x)]
#ndim = len(df.columns) - 1
#nsamples =
nsamples, ndim = df.shape
#Here: see if you can get ndim and nsamples from .csv file directly
ndim = ndim - 2
print(ndim)
print(nsamples)
max_lnprob = df['lnprob'].max()
max_r_in = df.r_in[df.lnprob.idxmax()]
max_delta_r = df.delta_r[df.lnprob.idxmax()]
max_m_disk = df.m_disk[df.lnprob.idxmax()]
max_f_star = df.f_star[df.lnprob.idxmax()]
max_position_angle = df.position_angle[df.lnprob.idxmax()]
max_inclination = df.inclination[df.lnprob.idxmax()]
max_xoffs = df.xoffs[df.lnprob.idxmax()]
max_yoffs = df.yoffs[df.lnprob.idxmax()]

#sampler = emcee.EnsembleSampler(16, ndim, max_lnprob)
#samples = sampler.chain[:,:,:].reshape((-1, ndim))
samples = np.zeros([nsamples,ndim])
samples[:,0] = df['r_in']
samples[:,1] = df['delta_r']
samples[:,2] = df['m_disk']
samples[:,3] = df['f_star']
samples[:,4] = df['position_angle']
samples[:,5] = df['inclination']
samples[:,6] = df['xoffs']
samples[:,7] = df['yoffs']
#fig = corner.corner(samples, labels=["$r_in$", "$delta_r$"],truths=[max_r_in, max_delta_r])
fig = corner.corner(samples, labels=["$r_{in}$", "$\Delta_r$", "$m_{disk}$", "$f_{star}$", "$position angle$", "$inclination$", "$\Delta_x$", "$\Delta_y$"],truths=[max_r_in, max_delta_r, max_m_disk, max_f_star, max_position_angle, max_inclination, max_xoffs, max_yoffs])

fig.savefig("cornerplot_20steps.png")

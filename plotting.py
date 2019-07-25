import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import emcee
import sys
import corner
import csv
import pylab
import seaborn as sns
#from astropy.io import fits
#from debris_disk import *
#from raytrace import *
#from single_model import *
#from ava_disk import *
#from multiprocessing import Pool


nwalkers=16
ndim=8
df = pd.read_csv("chain_20steps_new8params.csv")

#cmap_light = sns.diverging_palette(220, 20, center='dark', n=nwalkers)
#colors = ['red', 'blue', 'green', 'purple', 'yellow', 'black']
fig, ax = plt.subplots()
'''for i in range(nwalkers):
    #c = colors[i]
    ax.plot(df['r_in'][i::nwalkers], df['delta_r'][i::nwalkers], linestyle='-', marker='.', alpha=0.5)
plt.show(block=False)'''

w1m = df['r_in'][0::nwalkers]
w2m = df['delta_r'][1::nwalkers]
fig, (ax0,ax1,ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(ndim+1)
x = np.arange(0,len(w1m))
#print(np.shape(x),np.shape(w1m))
#print(np.shape(x),np.shape(w2m))
for i in range(0, nwalkers):
    ax0.plot(x, df['r_in'][i::nwalkers])
    ax1.plot(x, df['delta_r'][i::nwalkers])
    ax2.plot(x, df['m_disk'][i::nwalkers])
    ax3.plot(x, df['f_star'][i::nwalkers])
    ax4.plot(x, df['position_angle'][i::nwalkers])
    ax5.plot(x, df['inclination'][i::nwalkers])
    ax6.plot(x, df['xoffs'][i::nwalkers])
    ax7.plot(x, df['yoffs'][i::nwalkers])
    ax8.plot(x, df['lnprob'][i::nwalkers])
fig.suptitle('r_in, delta_r, m_disk, f_star, position_angle, inclination, xoffs, yoffs, lnprob')
plt.show(block=False)

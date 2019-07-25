import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import sys
import corner
import csv
import pylab
import seaborn as sns
from astropy.io import fits
from debris_disk import *
from raytrace import *
from single_model import *

#Parameters = ['r_in', 'delta_r', 'm_disk', 'f_star', 'position_angle', 'inclination', 'lnprob']

df = pd.read_csv("chain_20steps_new8params.csv")
max_lnprob = df['lnprob'].max()
max_r_in = df.r_in[df.lnprob.idxmax()]
max_delta_r = df.delta_r[df.lnprob.idxmax()]
max_m_disk = df.m_disk[df.lnprob.idxmax()]
max_f_star = df.f_star[df.lnprob.idxmax()]
max_position_angle = df.position_angle[df.lnprob.idxmax()]
max_inclination = df.inclination[df.lnprob.idxmax()]
max_xoffs = df.xoffs[df.lnprob.idxmax()]
max_yoffs = df.yoffs[df.lnprob.idxmax()]

print("Best-Fit Parameters:")
print("lnprob:", max_lnprob)
print("r_in:", max_r_in)
print("delta_r:", max_delta_r)
print("m_disk:", max_m_disk)
print("f_star:", max_f_star)
print("position_angle:", max_position_angle)
print("inclination:", max_inclination)
print("x__offset:", max_xoffs)
print("y__offsets:", max_yoffs)

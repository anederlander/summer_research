#Ava Nederlander MCMC code
#Summer 2019
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
#from ava_disk import *
from multiprocessing import Pool
#rin=1.
#rout=100.
#x_data_1=(np.genfromtxt('hoggs_data.txt',usecols=(1,))), y_data_1=(np.genfromtxt('hoggs_data.txt',usecols=(2,))), sigma_data_1=(np.genfromtxt('hoggs_data.txt',usecols=(4,)))
import time
start=time.time()

def chiSq(modelfiles_path, datafiles_path):
    """Calculate the goodness of fit between data and model."""
    data_uvf = fits.open(datafiles_path + '.uvfits')
    data_vis = data_uvf[0].data['data'].squeeze()

    model = fits.open(modelfiles_path + '.vis.fits')
    model_vis = model[0].data['data'].squeeze()

    data_real = (data_vis[:, :, 0, 0] + data_vis[:, :, 1, 0])/2.
    data_imag = (data_vis[:, :, 0, 1] + data_vis[:, :, 1, 1])/2.

    model_real = model_vis[::2, :, 0]
    model_imag = model_vis[::2, :, 1]

    wt = data_vis[:, :, 0, 2]

    raw_chi = np.sum(wt * (data_real - model_real)**2 +
                     wt * (data_imag - model_imag)**2)

    raw_chi = raw_chi
    print("Raw Chi2: ", raw_chi)
    return raw_chi * -0.5

def lnprob(p1):
    r_in, delta_r, log_m_disk, f_star, position_angle, inclination, xoffs, yoffs = p1
    priors_r_in = [0, 10000]
    priors_delta_r = [0, 10000]
    priors_m_disk = [-10, -2]
    priors_f_star = [0, 100000000]
    priors_position_angle = [0, 180]
    priors_inclination = [0, 90]
    priors_xoffs = [-5, 5]
    priors_yoffs = [-5, 5]

    if r_in < priors_r_in[0] or r_in > priors_r_in[1]:
        return -np.inf
        print("rin out of bounds")

    if delta_r < priors_delta_r[0] or delta_r > priors_delta_r[1]:
        return -np.inf
        print("delta_r out of bounds")

    if log_m_disk < priors_m_disk[0] or log_m_disk > priors_m_disk[1]:
        return -np.inf
        print("mdisk out of bounds")

    if f_star < priors_f_star[0] or f_star > priors_f_star[1]:
        return -np.inf
        print("fstar out of bounds")

    if position_angle < priors_position_angle[0] or position_angle > priors_position_angle[1]:
        return -np.inf
        print("position_angle out of bounds")

    if inclination < priors_inclination[0] or inclination > priors_inclination[1]:
        return -np.inf
        print("inclination out of bounds")

    if xoffs < priors_xoffs[0] or xoffs > priors_xoffs[1]:
        return -np.inf
        print("x offset out of bounds")

    if yoffs < priors_yoffs[0] or yoffs > priors_yoffs[1]:
        return -np.inf
        print("y offset out of bounds")

    m_disk = 10**log_m_disk
    r_out = r_in + delta_r

    x = Disk(params=[-0.5,m_disk,1.,r_in,r_out,150.,inclination,2.3,1e-4,0.01,33.9,[.79,1000],[10.,1000], -1, 500, 500, 2.83, 0.1])

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_0',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq0 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_0')
    print("*********0 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_1',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq1 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_1')
    print("*********1 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_0',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq2 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_0')
    print("*********2 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_1',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq3 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_1')
    print("*********3 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_2',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq4 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_2')
    print("*********4 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_3',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq5 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_high_res_avg_noflag_3')
    print("*********5 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw0',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq6 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw0')
    print("*********6 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw1',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq7 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw1')
    print("*********7 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw2',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq8 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_0_spw2')
    print("*********8 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw3',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq9 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw3')
    print("*********9 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw4',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq10 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw4')
    print("*********10 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw5',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq11 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/com_low_res_avg_noflag_1_spw5')
    print("*********11 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw0',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq12 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw0')
    print("*********12 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw1',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq13 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw1')
    print("*********13 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw2',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq14 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_0_spw2')
    print("*********14 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw3',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq15 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw3')
    print("*********15 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw4',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq16 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw4')
    print("*********16 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw5',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq17 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_1_spw5')
    print("*********17 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw6',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq18 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw6')
    print("*********18 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw7',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq19 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw7')
    print("*********19 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw8',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq20 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_2_spw8')
    print("*********20 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw9',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq21 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw9')
    print("*********21 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw10',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq22 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw10')
    print("*********22 through***********")

    total_model(x,imres=0.05,distance=40.8,nchans=5,freq0=222.516,xnpix=256,vsys=-12.4510196,PA=position_angle,offs=[xoffs,yoffs],modfile='raytrace_test3',isgas=False, x_offset=-0.480, y_offset=0.062, stellar_flux=f_star)
    make_model_vis(datfile='/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw11',modfile='raytrace_test3',isgas=True,freq0=345.79599)
    chiSq23 = chiSq('raytrace_test3.model', '/Volumes/disks/ava/hd206893/ext_low_res_avg_noflag_3_spw11')
    #make_model_vis(datfile='/Volumes/disks/ava/hd206893/com_high_res_avg_noflag_0',modfile='raytrace_test2',isgas=True,freq0=345.79599)
    print("*********23 through***********")
    print("HERE -- 100% through")
    return chiSq0+chiSq1+chiSq2+chiSq3+chiSq4+chiSq5+chiSq6+chiSq7+chiSq8+chiSq9+chiSq10+chiSq11+chiSq12+chiSq13+chiSq14+chiSq15+chiSq16+chiSq17+chiSq18+chiSq19+chiSq20+chiSq21+chiSq22+chiSq23


def emcee(nsteps=500, ndim=8, nwalkers=16, walker_1=50.0, walker_2=30.0, walker_3=-6, walker_4=1.65e-5, walker_5=60, walker_6=40, walker_7=.2, walker_8=.8, sigma_1=50.0, sigma_2=50.0, sigma_3=2, sigma_4=8e-6, sigma_5=20, sigma_6=10, sigma_7=.5, sigma_8=.5, restart=True):
    '''Perform MCMC affine invariants
    :param nsteps:              The number of iterations
    :param ndim:                number of dimensions
    :param nwalkers:            number of walkers
    :param walker_1:            the first parameter for the 1st dimension - r_in
    :param walker_2:            the first parameter for the 2nd dimension - delta_r
    :param walker_3:            the first parameter for the 3rd dimension - log_m_disk
    :param walker_4:            the first parameter for the 4th dimension - f_star
    :param walker_5:            the first parameter for the 5th dimension - position_angle
    :param walker_6:            the first parameter for the 6th dimension - inclination
    :param walker_7:            the first parameter for the 7th dimension - xoffs for disk
    :param walker_8:            the first parameter for the 8th dimension - yoffs for disk
    :param sigma_1:             sigma for walker_1
    :param sigma_2:             sigma for walker_2
    :param sigma_3:             sigma for walker_3
    :param sigma_4:             sigma for walker_4
    :param sigma_5:             sigma for walker_5
    :param sigma_6:             sigma for walker_6
    :param sigma_7:             sigma for walker_7
    :param sigma_8:             sigma for walker_8
    '''
    #r_out = r_in + delta_r
    '''walker_1_array = [walker_1]
    walker_2_array = [walker_2]
    walker_3_array = [walker_3]
    walker_4_array = [walker_4]
    walker_5_array = [walker_5]
    walker_6_array = [walker_6]
    p0 = [walker_1, walker_2, walker_3, walker_4, walker_5, walker_6]'''
    #chi_array = [np.sum(((y_data_1) - (walker_1_array*x_data_1+walker_2_array))**2/sigma_data_1**2)]
    if restart == False:
        p0 = np.random.normal(loc=(walker_1, walker_2, walker_3, walker_4, walker_5, walker_6, walker_7, walker_8), size=(nwalkers, ndim), scale=(sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6, sigma_7, sigma_8))
    else:
        #read from csv file
        dg = pd.read_csv("chain_25steps_new8params.csv")
        p0 = np.zeros([nwalkers,ndim])
        for i in range(nwalkers):
            p0[i,0] = dg['r_in'].iloc[-(nwalkers-i+1)]
            p0[i,1] = dg['delta_r'].iloc[-(nwalkers-i+1)]-p0[i,0] #future versions should be delta_r
            p0[i,2] = dg['m_disk'].iloc[-(nwalkers-i+1)]
            p0[i,3] = dg['f_star'].iloc[-(nwalkers-i+1)]
            p0[i,4] = dg['position_angle'].iloc[-(nwalkers-i+1)]
            p0[i,5] = dg['inclination'].iloc[-(nwalkers-i+1)]
            p0[i,6] = dg['xoffs'].iloc[-(nwalkers-i+1)]
            p0[i,7] = dg['yoffs'].iloc[-(nwalkers-i+1)]
        #p0 = (loc=(walker_1, walker_2, walker_3, walker_4, walker_5, walker_6), size=(nwalkers, ndim))
    import emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob) #threads=10, a=4.0
    run = sampler.sample(p0, iterations=nsteps, storechain=True)
    steps = []
    for i, result in enumerate(run):
        pos, lnprobs, blob = result

        new_step = [np.append(pos[k], lnprobs[k]) for k in range(nwalkers)]
        steps += new_step
        #print(pos)
        print(lnprobs)
        sys.stdout.write("Completed step {} of {}  \r".format(i, nsteps) )
        sys.stdout.flush()

    #steps = steps[5000:]
    df = pd.DataFrame(steps)
    df.columns = ['r_in', 'delta_r', 'm_disk', 'f_star', 'position_angle', 'inclination', 'xoffs', 'yoffs', 'lnprob']
    df.to_csv('chain_475steps_new8params.csv')

    '''max_lnprob = df['lnprob'].max()
    max_m = df.x[df.lnprob.idxmax()]
    max_b = df.y[df.lnprob.idxmax()]'''

    print(np.shape(sampler.chain))
    '''samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))
    fig = corner.corner(samples, labels=["$m$", "$b$"],truths=[max_m, max_b])
    fig.savefig("triangle1.png")'''

    '''print(max_lnprob)
    print(max_m)
    print(max_b)'''
    print("Finished MCMC.")
    print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    #plt.close()

    cmap_light = sns.diverging_palette(220, 20, center='dark', n=nwalkers)
    #colors = ['red', 'blue', 'green', 'purple', 'yellow', 'black']
    fig, ax = plt.subplots()
    for i in range(nwalkers):
        #c = colors[i]
        ax.plot(df['r_in'][i::nwalkers], df['delta_r'][i::nwalkers], linestyle='-', marker='.', alpha=0.5)
    plt.show(block=False)

    w1m = df['r_in'][0::nwalkers]
    w2m = df['delta_r'][1::nwalkers]
    fig, (ax0,ax1,ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(ndim+1)
    x = np.arange(0,len(w1m))
    print(np.shape(x),np.shape(w1m))
    print(np.shape(x),np.shape(w2m))
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

    print(np.shape(x),np.shape(w1m))


emcee()

print ('Elapsed time (hrs): ',(time.time()-start)/3600.)
'''
max_lnprob = df['lnprob'].max()
max_m = df.m[df.lnprob.idxmax()]
max_b = df.b[df.lnprob.idxmax()]
'''

#finish

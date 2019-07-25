#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:30:00 2019

@author: anederlander
"""
##importing relevant stuff
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import math
from matplotlib import *

plt.clf()
plt.rcParams.update({'font.size': 25})

##opening the .fits file to see what it contains
#hdulist = fits.open('/Volumes/disks/ava/hd206893/2best_fit_imaging.fits')
#hdulist = fits.open('/Volumes/disks/ava/hd206893/2best_fit_residual.fits')
hdulist = fits.open('/Volumes/disks/ava/hd206893/data_image2.fits')
hdulist.info()

hdulist1 = fits.open('/Volumes/disks/ava/hd206893/data_image2.fits')
hdulist1.info()
##defining so I can look at the header
hdu = hdulist1[0]
print(hdu.header)
##taking off the un-needed dimensions in the nparray
image_data = np.squeeze(hdulist[0].data)*1e6
image_data1 = np.squeeze(hdulist1[0].data)*1e6
##closing because we got everything we need
hdulist.close()
hdulist1.close()
#checking max and min
print(np.amax(image_data))
print(np.min(image_data))
##defining stuff
crval1=hdu.header['CRVAL1']*3600
crval2=hdu.header['CRVAL2']*3600
crdelt1=hdu.header['CDELT1']*3600
crdelt2=hdu.header['CDELT2']*3600
pix1=hdu.header['NAXIS1']
pix2=hdu.header['NAXIS2']
pixs1=pix1*crdelt1
pixs2=pix2*crdelt2
##plotting stuff and annotations
fig, ax = plt.subplots(figsize=(15,11))
plt.imshow(image_data, vmin=np.amin(image_data1), vmax=np.amax(image_data1),extent=[-(.5*pixs1),(.5*pixs1),-(.5*pixs2),(.5*pixs2)],cmap='afmhot',origin ='lower')
plt.xlabel(r'$\Delta \alpha$'   '["]',fontsize=25)
plt.ylabel(r'$\Delta \delta$'  '["]',fontsize=25)
#ax.annotate('Model',xy=(5.5, 5.5), xycoords='data',xytext=(-10, 10), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=22)
ax.annotate('HD 206893',xy=(5.5, 5.5), xycoords='data',xytext=(-10, 10), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=22)
ax.annotate('ALMA $\lambda$'"=1348" r'$\mu$''m',xy=(5.5, 4.9), xycoords='data',xytext=(-10, 10), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=22)
#ax.annotate('ALMA',xy=(1,1), xycoords='data',xytext=(-20, 20), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=25)
ax.annotate('100 au',xy=(-3.7, -5), xycoords='data',xytext=(-30, 30), textcoords='offset pixels',horizontalalignment='left',verticalalignment='top',color='white',fontsize=25)
##beamcode
xoff=5
yoff=-5
t=np.arange(0.0,2.0 *math.pi,0.05)
x=hdu.header['bmaj']*1800.0*np.cos(t)*np.cos((hdu.header['bpa']+90.0)*math.pi/180.0)-1800.0*hdu.header['bmin']*np.sin(t)*np.sin((hdu.header['bpa']+90.0)*math.pi/180.0)+xoff
y=1800.0*hdu.header['bmaj']*np.cos(t)*np.sin((hdu.header['bpa']+90.0)*math.pi/180.0)+1800.0*hdu.header['bmin']*np.sin(t)*np.cos((hdu.header['bpa']+90.0)*math.pi/180.0)+yoff
plt.fill(x, y, fill=False,hatch='//////',color='white')
#scalebar length 2.45" located at -7
plt.plot([-3, -5.45], [-5.5,-5.5], color="white")
##colorbar
cbar=plt.colorbar(pad=0)
cbar.set_label( r'$\mu$Jy / beam',fontsize=25)

#plt.savefig('hd206893_model.pdf')
plt.savefig('model77.pdf')
plt.show()
#100best_fit_residual_taper

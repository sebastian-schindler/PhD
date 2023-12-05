from tools_matplotlib import *
from tools_highdim import *

import numpy as np

import os.path as pth

def no_nan(array, *args):
	"""
	Remove NaNs from an array.
	For several arrays (of same shape): All positions that have a NaN in any of the arrays (i.e. the union of Nan positions) are removed from all arrays. A list of the arrays without NaNs is returned.
	"""
	
	mask = ~ np.isnan(array)
	
	if len(args) == 0:
		return array[mask]
	
	for i, arg in enumerate(args):
		if arg.shape != mask.shape:
			raise Exception("All arrays must have the same shape")
		try:
			mask &= ~ np.isnan(arg)
		except TypeError:
			print("Ignoring array no. %d in mask creation due to incompatible types" % (i+2))
	
	return [array[mask]] + [arg[mask] for arg in args]


def cache_file(url):
	"""Cache a remote file locally and return local path to the cache copy."""
	
	local = url.replace("ftp://", "").replace("http://", "").replace("https://", "")

	if not pth.exists(local):
		# trigger caching of file
		try:
			np.loadtxt(url)
		except ValueError:
			pass
	
	return local


import astropy.coordinates as coord
import astropy.units as u
def plt_skyplot(ra, dec, galactic=False, galaxy=False, figsize=[16,8], **kwargs):
    """Create skyplot in equatorial or galactic coordinates from arrays of RA and Dec values in degrees."""
    
    def trafo_equ(ra):
        ra = ra + 180*u.deg
        ra = ra.wrap_at(180*u.deg)
        ra = -ra
        return ra
    def trafo_gal(l):
        l = l.wrap_at(180*u.deg)
        l = -l
        return l
    
    plt.figure(figsize=figsize)
    plt.subplot(projection='mollweide')
    plt.grid()
    
    def set_xticks(left, right):
        
        xticks = np.linspace(-180, 180, 13) # 30 deg steps incl. 0 and bounds
        xticks = np.deg2rad(xticks)
        xticks = xticks[1:-1] # exclude bounds
        
        xticklabels = []
        for number in np.linspace(left, right, 13):
            xticklabels.append(u"%d \N{DEGREE SIGN}" % number)
        xticklabels = xticklabels[1:-1]
        
        plt.xticks(xticks, xticklabels)
    
    coords = coord.SkyCoord(ra*u.deg, dec*u.deg)
    if galactic:
        coords = coords.transform_to('galactic')
        coord_x = trafo_gal(coords.l)
        coord_y = coords.b
        set_xticks(180, -180)
        plt.text(0.9, 0.1, "Galactic", transform=plt.figure(1).axes[0].transAxes)
 
    else:
        coord_x = trafo_equ(coords.ra)
        coord_y = coords.dec
        set_xticks(360, 0)
        plt.text(0.9, 0.1, "Equatorial", transform=plt.figure(1).axes[0].transAxes)

    plt.scatter(coord_x.rad, coord_y.rad, **kwargs)
    
    if galaxy:
        
        plane = coord.SkyCoord(frame='galactic', l=np.linspace(0, 360, 100)*u.deg, b=0*u.deg)
        plane = plane.transform_to('icrs')
        plt.plot(trafo_equ(plane.ra).rad, plane.dec.rad, 'black', linewidth=.5)

        center = coord.SkyCoord(frame='galactic', l=0*u.deg, b=0*u.deg)
        center = center.transform_to('icrs')
        plt.plot(trafo_equ(center.ra).rad, center.dec.rad, '*k', markersize=5)


import pickle as pkl

def pickle(obj, filename):
	pkl.dump(obj, open(filename, 'wb'))

def unpickle(filename):
	pkl.load(open(filename, 'rb'))


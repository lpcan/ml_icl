"""
Script to download cutouts of cluster catalogue using unagi
"""

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii
import astropy.units as u
import numpy as np
import os
from unagi import hsc
from unagi.task import hsc_cutout

# Parameters
script_dir = os.path.dirname(__file__)

cat_path = os.path.join(script_dir, "../raw/camira_s20a_dud.tbl") #"../raw/camira_s20a_wide.tbl"
exclude = None #"../raw/camira_s20a_dud.tbl"
half_size = 1 * u.arcmin
output_dir = os.path.join(script_dir, '../../cutouts_550kpc/') #"../raw/cutouts/"
rerun = "pdr2_dud"

# HSC username: locan@local
# HSC password: 

# Create a list of coordinates of the cluster locations
tbl = ascii.read(cat_path, names = ['ID', 
                                    'Name', 
                                    'RA [deg]', 
                                    'Dec [deg]', 
                                    'z_cl', 
                                    'Richness', 
                                    'BCG redshift'])

zs = tbl['z_cl']
ras = tbl['RA [deg]'][zs <= 0.5]
decs = tbl['Dec [deg]'][zs <= 0.5]
names = tbl['Name'][zs <= 0.5]
zs = zs[zs <= 0.5]

coords = SkyCoord(ras, decs, unit=u.deg)
print(f"{len(coords)} clusters to download")

# Check for overlap with exclude catalogue and remove
if exclude is not None:
    tbl = ascii.read(exclude, names = ['ID', 
                                       'Name', 
                                       'RA [deg]', 
                                       'Dec [deg]', 
                                       'z_cl', 
                                       'Richness', 
                                       'BCG redshift'])
    
    ex_zs = tbl['z_cl']
    ex_ras = tbl['RA [deg]'][ex_zs <= 0.5]
    ex_decs = tbl['Dec [deg]'][ex_zs <= 0.5]

    ex_coords = SkyCoord(ex_ras, ex_decs, unit=u.deg)

    _, idx, _, _ = coords.search_around_sky(ex_coords, 25*u.arcsec)
    mask = np.ones(len(coords), dtype=bool)
    mask[idx] = 0

    coords = coords[mask]
    names = names[mask]
    zs = zs[mask]
    print(f"{len(idx)} matches made, now {len(coords)} clusters to download") 

# Download the cutouts and put them in the output directory
pdr2 = hsc.Hsc(dr='pdr2', rerun=rerun)

f = open(output_dir+"failed_downloads.txt", "a+")
f.write(f"Clusters from {cat_path}\n")

cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)

for i,c in enumerate(coords):
    print(f"Downloading cluster {i}")

    ################ REMOVE THIS LATER #####################
    half_size = cosmo.arcsec_per_kpc_proper(zs[i]) * 550 * u.kpc
    ########################################################

    try:
        hsc_cutout(c, cutout_size=half_size, filters='r', archive=pdr2, 
                   use_saved=False, output_dir=output_dir, verbose=False, 
                   save_output=True, variance=False, mask=True)
    except Exception as e:
        print(f"{e} for cluster {i}")
        f.write(f"Cluster {i} ({names[i]})\n")

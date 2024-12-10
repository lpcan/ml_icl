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
SCRIPT_DIR = os.path.dirname(__file__)

CAT_PATH = SCRIPT_DIR + '/../processed/camira_final.tbl'
EXCLUDE = None # do not redownload clusters that are already in `exclude`
HALF_SIZE = 1 * u.arcmin 
OUTPUT_DIR = 'TODO:OUTPUT_DIR'
RERUN = 'pdr2_dud'

# Create a list of coordinates of the cluster locations
tbl = ascii.read(CAT_PATH, names = ['ID', 
                                    'Name', 
                                    'RA [deg]', 
                                    'Dec [deg]', 
                                    'z_cl', 
                                    'Richness', 
                                    'BCG redshift'])
tbl = tbl[:125]
zs = tbl['z_cl']
ras = tbl['RA [deg]'][zs <= 0.5] # remove clusters that are z > 0.5
decs = tbl['Dec [deg]'][zs <= 0.5]
names = tbl['Name'][zs <= 0.5]
zs = zs[zs <= 0.5]

coords = SkyCoord(ras, decs, unit=u.deg)
print(f"{len(coords)} clusters to download")

# Check for overlap with exclude catalogue and remove
if EXCLUDE is not None:
    tbl = ascii.read(EXCLUDE, names = ['ID', 
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
pdr2 = hsc.Hsc(dr='pdr2', rerun=RERUN)

f = open(OUTPUT_DIR+"failed_downloads.txt", "a+")
f.write(f"Clusters from {CAT_PATH}\n")

cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)

for i,c in enumerate(coords):
    print(f"Downloading cluster {i}")

    try:
        hsc_cutout(c, cutout_size=HALF_SIZE, filters='r', archive=pdr2, 
                   use_saved=False, output_dir=HALF_SIZE, verbose=False, 
                   save_output=True, variance=False, mask=True)
    except Exception as e:
        print(f"{e} for cluster {i}")
        f.write(f"Cluster {i} ({names[i]})\n")

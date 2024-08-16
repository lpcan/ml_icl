"""
Script to do multiple bulk downloads of clusters using unagi
"""

from astropy.io import ascii, fits
from astropy.table import Table, join, vstack
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde
import shutil
import skimage
from unagi import hsc
# from unagi.task import hsc_bulk_cutout
from task import hsc_bulk_cutout

from astropy.cosmology import FlatLambdaCDM

# Parameters
script_dir = os.path.dirname(__file__)
tbl_path = '/srv/scratch/z5214005/lrgs_sampled_1405.tbl'
dwnld_half_size = 163 * u.arcsec
output_dir = '/srv/scratch/z5214005/tmp/'
rerun = 'pdr2_wide'

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# HSC username: locan@local
# HSC password: ########################################
                
def compute_sample_weights(table=None):
    """
    Create a table that samples from the LRG table to match the BCG mass distribution
    """
    if table is None:
        f = fits.open('/srv/scratch/z5214005/lrg_s18a_wide_sm.fits')
        tbl = f[1].data[f[1].data['z'] <= 0.5]
        table = Table(tbl)

    lrg_masses = np.log10(table['ms']) 
    lrg_kernel = gaussian_kde(lrg_masses) # Kernel density estimate for LRGs

    # Get the masses of the BCGs
    cluster_tbl = ascii.read('/srv/scratch/z5214005/camira_final.tbl')
    cluster_tbl = cluster_tbl[cluster_tbl['z_cl'] <= 0.5]

    members = ascii.read('/srv/scratch/z5214005/camira_s20a_wide_member.dat', 
                    names=['RA_cl', 'Dec_cl', 'Richness', 'z_cl', 'RA', 'Dec', 'M', 'w'])
    merged1 = join(members, cluster_tbl, keys_left=['RA', 'Dec'], keys_right=['RA [deg]', 'Dec [deg]'])

    members = ascii.read('/srv/scratch/z5214005/camira_s20a_dud_member.dat', 
                    names=['RA_cl', 'Dec_cl', 'Richness', 'z_cl', 'RA', 'Dec', 'M', 'w'])
    merged2 = join(members, cluster_tbl, keys_left=['RA', 'Dec'], keys_right=['RA [deg]', 'Dec [deg]'])
    merged = vstack([merged1, merged2])
    
    bcg_masses = merged['M']
    bcg_kernel = gaussian_kde(bcg_masses) # Kernel density estimate for BCGs

    # Compute the probability of sampling from each mass bin in the two samples
    step = 0.001 # Larger step means faster computation but less accurate kernel
    pts = np.arange(lrg_masses.min(), lrg_masses.max(), step=step)
    probs_lrg = lrg_kernel(pts)
    probs_bcg = bcg_kernel(pts) 

    # Compute the required sample weights
    weights = probs_bcg / probs_lrg

    idxs = ((lrg_masses - lrg_masses.min()) // step).astype(int) # What weight does each entry in the lrg table correspond to? 
    weights = weights[idxs]

    return weights / np.sum(weights) # Needs to be a vector of probabilities

def create_table(size=50_000):
    """
    Create a list of `size` coordinates to download by weighted sampling of LRG
    table
    """
    
    f = fits.open(os.path.join(script_dir, '/srv/scratch/z5214005/lrg_s18a_wide_sm.fits'))
    tbl = f[1].data[f[1].data['z'] <= 0.5]
    tbl = tbl[tbl['z'] >= 0.1]
    tbl = Table(tbl)
    tbl['old_ids'] = np.arange(len(tbl)).astype(str)

    weights = compute_sample_weights(tbl)

    # Select 50,000 galaxies at random
    rand_is = sorted(np.random.choice(a=len(tbl), size=size, p=weights)) 
    tbl = tbl[rand_is]

    # Update labels so that we don't have duplicate object ids
    new_ids = np.arange(len(tbl)).astype(str)
    new_ids = np.char.add('-', new_ids)
    tbl['new_ids'] = np.char.add(tbl['old_ids'], new_ids)
    tbl['object_id'] = np.arange(len(tbl))

    return tbl

def download_bulk(tbl, overwrite=False):
    """
    Perform multiple bulk downloads of all clusters in `tbl`, crop to 600x600kpc,
    resize to 224x224 pixels, and save into a HDF file.
    """
    if overwrite:
        input('Are you sure you want to overwrite? If not, exit now!')
        resized_cutouts = h5py.File('/srv/scratch/z5214005/lrg_cutouts_300kpc_resized.hdf', 'w') # File to put all the processed cutouts into
    else:
        resized_cutouts = h5py.File('/srv/scratch/z5214005/lrg_cutouts_300kpc_resized.hdf', 'a')

    # Remove any rows that have already been downloaded
    to_remove = []
    for i in range(len(tbl)):
        key = tbl['new_ids'][i]
        if key in resized_cutouts:
            to_remove.append(i)
    tbl.remove_rows(to_remove)
    print(f'Skipping {len(to_remove)} rows that have already been downloaded.')

    # Download cutouts 2000 galaxies at a time 
    pdr2 = hsc.Hsc(dr='pdr2', rerun=rerun)

    # Only download the unique galaxies to minimise the amount of downloads
    _, unique_ids, repeats = np.unique(tbl['old_ids'], return_index=True, return_counts=True)
    to_download = tbl[unique_ids]
    to_download['repeats'] = repeats
    to_download['unique_ids'] = unique_ids
    inds = np.arange(len(to_download) + 1000, step=1000)

    for i in range(len(inds) - 1):
        # Download the cutouts
        sub_tbl = to_download[inds[i]:inds[i+1]]

        filename = hsc_bulk_cutout(sub_tbl, cutout_size=dwnld_half_size, filters='r', 
                                   archive=pdr2, overwrite=True, tmp_dir=output_dir, 
                                   nproc=8, mask=True, output_dir=output_dir)

        # Open the HDF file, resize the cutouts to 224x224 and save into a new file
        print('Resizing batch...')
        cutouts = h5py.File(filename)
        
        for j, key in enumerate(sub_tbl['object_id']):
            if str(key) not in cutouts:
                print(f'Key {key} missing from download')
                continue
            cutout = np.array(cutouts[str(key)]['HSC-R']['HDU0']['DATA'])
            mask = np.array(cutouts[str(key)]['HSC-R']['HDU1']['DATA']).astype(int)

            BAD = 1
            NO_DATA = (1 << 8)
            BRIGHT_OBJECT = (1 << 9)
            mask = (mask & (BAD | BRIGHT_OBJECT | NO_DATA)).astype(bool)

            # Crop the cutout and mask to the correct physical size
            half_size = int(((cosmo.arcsec_per_kpc_proper(sub_tbl[j]['z']) * 300).value) / 0.168)
            centre = cutout.shape[0] // 2, cutout.shape[1] // 2
            mask = mask[centre[0]-half_size:centre[0]+half_size,centre[1]-half_size:centre[1]+half_size]
            cutout = cutout[centre[0]-half_size:centre[0]+half_size,centre[1]-half_size:centre[1]+half_size]

            resized = skimage.transform.resize(cutout, (224,224))
            resized_mask = skimage.transform.resize(mask, (224, 224))
            # If necessary, save multiple copies of this galaxy
            for repeat in range(sub_tbl['repeats'][j]):
                tbl_id = sub_tbl['unique_ids'][j]
                new_key = tbl['new_ids'][tbl_id+repeat]
                if new_key in resized_cutouts:
                    # This key already exists in the file, do not overwrite
                    print(f'Clash with key {new_key}')
                    continue
                else:
                    resized_cutouts[f'{new_key}/HDU0/DATA'] = resized.copy()
                    resized_cutouts[f'{new_key}/HDU1/DATA'] = resized_mask.copy()
        print('Batch resized to 224x224 cutouts')
        cutouts.close()

if __name__ == '__main__':
    if tbl_path is None:
        # No table path provided, create and save a new table
        tbl = create_table()
        tbl.write('/srv/scratch/z5214005/lrgs_sampled_1405.tbl', format='ascii', overwrite=False)
    else:
        tbl = ascii.read(tbl_path)

    download_bulk(tbl, overwrite=False)

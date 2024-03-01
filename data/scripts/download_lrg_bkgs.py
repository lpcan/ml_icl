from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii
import astropy.units as u
from astropy.coordinates import SkyCoord
import h5py
import numpy as np
from unagi import hsc
from unagi.task import hsc_cutout
from photutils.background import Background2D
from scipy.interpolate import bisplrep, bisplev
from scipy.ndimage import zoom
import skimage
import sys
import time

tbl_path = '/srv/scratch/z5214005/lrgs_sampled.tbl' # Path relative to script directory
output_dir = '/srv/scratch/mltidal/tmp/'
rerun = 'pdr2_wide'

cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)

start_time = time.time()

def download_all(tbl, overwrite=False):
    if overwrite:
        resized_cutouts = h5py.File('/srv/scratch/mltidal/lrg_bkg+mask_resized.hdf', 'w') # File to put all the processed cutouts into
    else:
        resized_cutouts = h5py.File('/srv/scratch/mltidal/lrg_bkg+mask_resized.hdf', 'a')

    # Remove any rows that have already been downloaded
    to_remove = []
    for i in range(len(tbl)):
        key = tbl['new_ids'][i]
        if key in resized_cutouts:
            to_remove.append(i)
    tbl.remove_rows(to_remove)
    print(f'Skipping {len(to_remove)} rows that have already been downloaded.')

    pdr2 = hsc.Hsc(dr='pdr2', rerun=rerun)

    # Only download the unique galaxies
    _, unique_ids, repeats = np.unique(tbl['old_ids'], return_index=True, return_counts=True)
    to_download = tbl[unique_ids]
    to_download['repeats'] = repeats
    to_download['unique_ids'] = unique_ids 

    for idx, row in enumerate(to_download):
        print(row) 
        half_size = cosmo.arcsec_per_kpc_proper(row['z']) * 500 * u.kpc
        c = SkyCoord(row['ra'], row['dec'], unit=u.deg)
        cutout = hsc_cutout(c, cutout_size=half_size, filters='r', archive=pdr2, 
                   use_saved=False, output_dir=output_dir, verbose=False, 
                   save_output=False, variance=False, mask=True)

        mask = cutout[2].data.astype(int) 
        cutout = cutout[1].data

        BAD = 1
        NO_DATA = (1 << 8)
        BRIGHT_OBJECT = (1 << 9)
        bad_mask = (mask & (BAD | BRIGHT_OBJECT | NO_DATA)).astype(bool)

        # Background estimate
        box_size = cutout.shape[0] // 14
        bkg_initial = Background2D(cutout, box_size=box_size, mask=bad_mask)
        mesh = bkg_initial.background_mesh

        # Get the size of the circle in box coordinates
        radius_px = (cosmo.arcsec_per_kpc_proper(to_download[idx]['z']) * 300).value * 1/0.168
        radius_box = int(np.ceil(radius_px / box_size))
        Y, X = np.ogrid[:mesh.shape[0], :mesh.shape[1]]
        y_cen, x_cen = (mesh.shape[0] // 2, mesh.shape[1] // 2)
        dist_from_centre = np.sqrt((X - x_cen)**2 + (Y - y_cen)**2)
        circle = dist_from_centre <= radius_box

        # Get values from the mesh corresponding to these coordinates
        vals = mesh[~circle]

        # Do the interpolation
        box_circle = np.argwhere(~circle)
        tck = bisplrep(*box_circle.T, vals)
        znew = bisplev(np.arange(14), np.arange(14), tck)
        bkg = zoom(znew, np.array(cutout.shape) / np.array([14, 14]), mode='reflect')
        
        # Crop and resize everything
        centre = cutout.shape[0] // 2, cutout.shape[1] // 2
        mask = bad_mask[centre[0]-358:centre[0]+358,centre[1]-358:centre[1]+358]
        mask = skimage.transform.resize(bad_mask, (224,224))
        bkg = bkg[centre[0]-358:centre[0]+358,centre[1]-358:centre[1]+358]
        bkg = skimage.transform.resize(bkg, (224,224))

        mask = mask.astype('<f4')
        bkg = bkg.astype('<f4')

        # If necessary, save multiple copies of this galaxy
        for repeat in range(to_download['repeats'][idx]):
            tbl_id = to_download['unique_ids'][idx]
            new_key = tbl['new_ids'][tbl_id+repeat]
            if new_key in resized_cutouts:
                # This key already exists in the file, do not overwrite
                print(f'Clash with key {new_key}')
                continue
            else:
                resized_cutouts[f'{new_key}/BKG'] = bkg.copy()
                resized_cutouts[f'{new_key}/MASK'] = mask.copy()
        
        # If we are almost out of time, make sure to close the file properly
        if (time.time() - start_time) > (11 * 60 * 60):
            print('Out of time')
            resized_cutouts.close()
            sys.exit(0) 
    
def finish_partial(to_download, resized_cutouts, tbl):
    import glob
    from astropy.io import fits
    filenames = glob.glob('/srv/scratch/mltidal/tmp/arch-240215-001029/*.fits')
    for filename in filenames:
        print(filename)
        f = fits.open(filename)
        idx = int(filename.split('/')[-1].split('-')[0]) - 2
        
        cutout = f[1].data 
        mask = f[2].data

        # Crop to 500kpc
        px_half_size = np.min((int((cosmo.arcsec_per_kpc_proper(to_download[idx]['z']) * 500).value * 1/0.168), cutout.shape[0] // 2, cutout.shape[1] // 2))
        centre = cutout.shape[0] // 2, cutout.shape[1] // 2
        cutout = cutout[centre[0] - px_half_size : centre[0] + px_half_size, centre[1] - px_half_size : centre[1] + px_half_size]
        mask = mask[centre[0] - px_half_size : centre[0] + px_half_size, centre[1] - px_half_size : centre[1] + px_half_size].astype(int)

        BAD = 1
        NO_DATA = (1 << 8)
        BRIGHT_OBJECT = (1 << 9)
        bad_mask = (mask & (BAD | BRIGHT_OBJECT | NO_DATA)).astype(bool)

        # Background estimate
        box_size = cutout.shape[0] // 14
        bkg_initial = Background2D(cutout, box_size=box_size, mask=bad_mask)
        mesh = bkg_initial.background_mesh

        # Get the size of the circle in box coordinates
        radius_px = (cosmo.arcsec_per_kpc_proper(to_download[idx]['z']) * 300).value * 1/0.168
        radius_box = int(np.ceil(radius_px / box_size))
        Y, X = np.ogrid[:mesh.shape[0], :mesh.shape[1]]
        y_cen, x_cen = (mesh.shape[0] // 2, mesh.shape[1] // 2)
        dist_from_centre = np.sqrt((X - x_cen)**2 + (Y - y_cen)**2)
        circle = dist_from_centre <= radius_box

        # Get values from the mesh corresponding to these coordinates
        vals = mesh[~circle]
        print(cutout.shape, px_half_size, radius_px)
        # Do the interpolation
        box_circle = np.argwhere(~circle)
        tck = bisplrep(*box_circle.T, vals)
        znew = bisplev(np.arange(14), np.arange(14), tck)
        bkg = zoom(znew, np.array(cutout.shape) / np.array([14, 14]), mode='reflect')
        
        # Crop and resize everything
        centre = cutout.shape[0] // 2, cutout.shape[1] // 2
        mask = mask[centre[0]-358:centre[0]+358,centre[1]-358:centre[1]+358]
        mask = skimage.transform.resize(mask, (224,224))
        bkg = bkg[centre[0]-358:centre[0]+358,centre[1]-358:centre[1]+358]
        bkg = skimage.transform.resize(bkg, (224,224))

        # If necessary, save multiple copies of this galaxy
        for repeat in range(to_download['repeats'][idx]):
            tbl_id = to_download['unique_ids'][idx]
            new_key = tbl['new_ids'][tbl_id+repeat]
            if new_key in resized_cutouts:
                # This key already exists in the file, do not overwrite
                print(f'Clash with key {new_key}')
                continue
            else:
                resized_cutouts[f'{new_key}/BKG'] = bkg.copy()
                resized_cutouts[f'{new_key}/MASK'] = mask.copy()

if __name__=='__main__':
    tbl = ascii.read(tbl_path)
    download_all(tbl, overwrite=False)



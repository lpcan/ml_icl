from measure_sb_cut.scripts import measure_icl
from data.scripts.display_cutouts import stretch

from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats, mad_std
import h5py
import numpy as np
from photutils.background import Background2D
from photutils.segmentation import detect_sources
import scipy
from scipy.interpolate import bisplrep, bisplev
from scipy.ndimage import zoom
from skimage.morphology import binary_closing

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import tqdm
import sys

import skimage

def k_corr(z):
    # Equation for LRGs from Chilingarian+2010
    return 0.710579*z + 10.1949*z**2 - 57.0378*z**3 + 133.141*z**4 - 99.9271*z**5

def background_estimate(cutout, bad_mask):
    box_size = 224 // 14
    bkg_initial = Background2D(cutout, box_size=box_size, mask=bad_mask)
    mesh = bkg_initial.background_mesh 
    
    Y, X = np.ogrid[:mesh.shape[0], :mesh.shape[1]]

    box = (X < mesh.shape[1]  - 1) & (X > 0) & (Y < mesh.shape[0]) & (Y > 0)

    vals = mesh[~box]

    box_square = np.argwhere(~box)
    tck = bisplrep(*box_square.T, vals)
    znew = bisplev(np.arange(14), np.arange(14), tck)
    bkg = zoom(znew, np.array(cutout.shape) / np.array([14, 14]), mode='reflect')

    return bkg

def calc_icl_frac(cutout, bad_mask, z):
    if bad_mask[112,112]:
        # Bright star mask extends over the centre of the image, get rid of it
        bad_mask = np.zeros_like(bad_mask, dtype=bool) 

    # Measure and subtract the background
    bkg = background_estimate(cutout, np.zeros_like(bad_mask, dtype=bool))
    cutout = cutout - bkg

    cutout = cutout * ~bad_mask

    set_threshold = 26
    threshold = measure_icl.sb2counts(set_threshold + 10 * np.log10(1+z) + k_corr(z))
    segm = detect_sources(cutout, threshold=threshold, npixels=10)
    new_labels = segm.data

    # Unsharp mask the image for hot mask creation
    kernel = Gaussian2DKernel(2)
    conv_img = convolve(cutout, kernel)
    unsharp = cutout - conv_img

    # Create hot mask
    hot_mask_bkg = Background2D(unsharp, box_size=16).background
    combined_mask = (new_labels > 0)
    hot_labels = measure_icl.create_hot_labels(unsharp, combined_mask, background=hot_mask_bkg, npixels=3)
    hot_mask = (hot_labels > 0)

    # Calculate 220kpc
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    radius = ((cosmo.arcsec_per_kpc_proper(z) * 220 / 0.168) / (715 / 224)).value

    # Generate the mask
    centre = (112, 112)
    Y, X = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
    dist_from_centre = np.sqrt((X - centre[0])**2 + (Y - centre[1])**2)
    circ_mask = dist_from_centre <= radius

    # Calculate surface brightness limit
    _, _, stddev = sigma_clipped_stats(cutout, mask=bad_mask) 
    sb_lim = -2.5 * np.log10(3 * stddev/(0.168 * 10)) + 2.5 * np.log10(63095734448.0194)

    # Convert image from counts to surface brightness
    np.seterr(invalid='ignore', divide='ignore')
    sb_img = measure_icl.counts2sb(cutout, z, k_corr(z))

    # Mask values below surface brightness limit
    sb_img[sb_img >= sb_lim] = np.nan

    # Mask above the surface brightness threshold
    threshold = set_threshold
    mask = sb_img >= threshold

    # Close the mask
    # mask = binary_closing(mask)

    # Convert the SB image back to counts
    counts_img = measure_icl.sb2counts(sb_img)

    # Close the nans to try and get rid of the noise that contributes to the ICL
    nans = np.isnan(counts_img) 
    # nans = binary_closing(nans) 
    not_nans = ~nans

    # Get the final image
    masked_img = counts_img * circ_mask * not_nans * ~hot_mask

    # Calculate the values
    icl = np.nansum(masked_img * mask)
    total = np.nansum(masked_img)

    return icl, total, icl / total

def run_requested_keys(args):
    keys, length, zs, new_ids = args

    cutouts = h5py.File('/srv/scratch/mltidal/generated_data_kcorr.hdf')
    masks = h5py.File('/srv/scratch/mltidal/lrg_bkg+mask_resized.hdf')

    # Find the shared memory and create a numpy array interface
    shmem = SharedMemory(name=f'iclbuf', create=False)
    fracs = np.ndarray((3, int(length/3)), buffer=shmem.buf, dtype=np.float64)

    for key in keys: 
        cutout = np.array(cutouts[str(new_ids[key])]['HDU0']['DATA'])
        bad_mask = np.array(masks[str(new_ids[key])]['MASK']).astype(bool)

        z = zs[key]

        icl, total, frac = calc_icl_frac(cutout, bad_mask, z)

        fracs[0,key] = icl
        fracs[1,key] = total
        fracs[2,key] = frac

    return

def calc_icl_frac_parallel(keys, zs):
    """
    Use multiprocessing to divide the cutouts among available cores and 
    calculate the ICL fractions.
    """
    # Use all available cores
    cores = mp.cpu_count()
    num_keys = len(keys)

    # Divide the keys up into 20 
    jobs = np.array_split(np.arange(num_keys), 20)
    length = num_keys * 3
    args = [(j, length, zs, keys) for j in jobs]

    exit = False
    try:
        # Set up the shared memory
        global mem_id
        mem_id = 'iclbuf'
        nbytes = num_keys * 3 * np.float64(1).nbytes 
        iclmem = SharedMemory(name='iclbuf', create=True, size=nbytes)

        # Start a new process for each task
        ctx = mp.get_context()
        pool = ctx.Pool(processes=cores, maxtasksperchild=1)
        try:
            for _ in tqdm.tqdm(pool.imap_unordered(run_requested_keys, args, chunksize=1), total=len(jobs)):
                pass
        except KeyboardInterrupt:
            print('Caught kbd interrupt')
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            # Copy the result
            result = np.ndarray((3, num_keys), buffer=iclmem.buf,
                                dtype=np.float64).copy()
    finally:
        # Close the shared memory
        iclmem.close()
        iclmem.unlink()
        if exit:
            sys.exit(1)
    return result

if __name__ == '__main__':
    tbl = ascii.read('/srv/scratch/z5214005/lrgs_sampled.tbl')
    zs = tbl['z']
    new_ids = tbl['new_ids']

    fracs = calc_icl_frac_parallel(new_ids, zs)

    np.save('/srv/scratch/mltidal/fracs_gendata_kcorr.npy', fracs)

from measure_sb_cut.scripts import measure_icl
from data.scripts.display_cutouts import stretch

from astropy.cosmology import FlatLambdaCDM
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats
import h5py
import numpy as np
from photutils.background import Background2D
from photutils.segmentation import SourceFinder
import scipy
from skimage.morphology import binary_closing

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import tqdm
import sys

import skimage

def calc_icl_frac(cutout, z):
    # Background estimate
    # bkg = measure_icl.background_estimate(cutout, zs[key], cosmo)
    bkg_subtracted = cutout# - bkg

    # Segment the image
    cold_labels = measure_icl.create_cold_labels(cutout, np.zeros(cutout.shape, dtype=bool), kernel_size=2, npixels=10)
    if cold_labels is None:
        cold_labels = np.zeros_like(cutout)

    # Create cold mask
    cold_mask = measure_icl.enlarge_mask(cold_labels, sigma=0.5)

    # Create the "member mask"
    mid = (cutout.shape[0] // 2, cutout.shape[1] // 2)
    # Remove the central blob from the cold labels
    central_blob = cold_labels[mid[0], mid[1]]
    unblobbed = ((cold_labels != 0) & (cold_labels != central_blob)) * cold_labels
    unblobbed = measure_icl.enlarge_mask(unblobbed, sigma=0.66) # Enlarge this mask
    # Calculate new, smaller masks for central galaxies
    threshold = measure_icl.sb2counts(25 + 10 * np.log10(1 + z))
    finder = SourceFinder(20, progress_bar=False)
    deblended = finder(bkg_subtracted, threshold)
    centrals_newlabels = deblended.data * (cold_labels == central_blob)
    # Renumber
    centrals_newlabels = (np.max(cold_labels) + centrals_newlabels) * centrals_newlabels.astype(bool)
    # Combine to create our new mask
    combined_labels = unblobbed + centrals_newlabels

    # Unsharp mask the image for hot mask creation
    kernel = Gaussian2DKernel(2) 
    conv_img = convolve(np.array(cutout), kernel)
    unsharp = cutout - conv_img

    # Create hot mask
    hot_mask_bkg = Background2D(unsharp, box_size=16).background
    combined_mask = (combined_labels > 0)
    hot_labels = measure_icl.create_hot_labels(unsharp, combined_mask, background=hot_mask_bkg, npixels=3)
    hot_mask = measure_icl.enlarge_mask(hot_labels, sigma=0.3)

    # Get the BCG label
    bcg_label = combined_labels[mid[0], mid[1]]

    # Coordinates of points that are part of the BCG
    pts = np.array(np.argwhere(combined_labels == bcg_label))

    # Find points that are furthest apart
    candidates = pts[scipy.spatial.ConvexHull(pts).vertices]
    dist_mat = scipy.spatial.distance_matrix(candidates, candidates)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    pt1 = candidates[i]
    pt2 = candidates[j]

    size = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    # Make sure that the radius is >=100kpc
    # radius = np.max((size, cosmo.arcsec_per_kpc_proper(zs[num]).value * 100 * 1/0.168))
    radius = size * 1.5

    # Generate the mask
    centre = (cutout.shape[1] // 2, cutout.shape[0] // 2)
    Y, X = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
    dist_from_centre = np.sqrt((X-centre[0])**2 + (Y-centre[1])**2)
    circ_mask = dist_from_centre <= radius

    # Get the member mask
    member_mask = ((centrals_newlabels > 0) & (circ_mask > 0)) | (centrals_newlabels == 0)
    non_member_mask = ~member_mask
    non_member_mask = non_member_mask + hot_mask + unblobbed
    member_mask = ~non_member_mask

    # Calculate surface brightness limit
    _, _, stddev = sigma_clipped_stats(bkg_subtracted)
    sb_lim = -2.5 * np.log10(3 * stddev/(0.168 * 10)) + 2.5 * np.log10(63095734448.0194)

    # Convert image from counts to surface brightness
    np.seterr(invalid='ignore', divide='ignore')
    sb_img = measure_icl.counts2sb(bkg_subtracted, 0)

    # Mask values below surface brightness limit
    sb_img[sb_img >= sb_lim] = np.nan

    # Mask above the surface brightness threshold
    threshold = 25 + 10 * np.log10(1 + z)
    mask = sb_img >= threshold
    
    # Close the mask
    mask = binary_closing(mask)

    # Convert the SB image back to counts
    counts_img = measure_icl.sb2counts(sb_img) 

    # Close the nans to try and get rid of the noise that contributes to the ICL?
    nans = np.isnan(counts_img)
    nans = binary_closing(nans)
    not_nans = ~nans

    # Display the final image
    masked_img = counts_img * member_mask * circ_mask * not_nans

    icl = np.nansum(masked_img * mask)
    total = np.nansum(masked_img)

    return icl, total, icl / total

def run_requested_keys(args):
    keys, length, zs, new_ids = args

    cutouts = h5py.File('/srv/scratch/mltidal/generated_data_deep_v2.hdf')
    # cutouts = h5py.File('/srv/scratch/z5214005/hsc_icl/cutouts.hdf')

    # Find the shared memory and create a numpy array interface
    shmem = SharedMemory(name=f'iclbuf', create=False)
    fracs = np.ndarray((3, int(length/3)), buffer=shmem.buf, dtype=np.float64)

    # Parameters
    cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)

    for key in keys: 
        cutout = np.array(cutouts[str(new_ids[key])]['HDU0']['DATA'])
        z = zs[key]

        icl, total, frac = calc_icl_frac(cutout, z)

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
    tbl = ascii.read('/srv/scratch/z5214005/lrgs_dud_sampled.tbl')
    zs = tbl['z']
    new_ids = tbl['new_ids']

    fracs = calc_icl_frac_parallel(new_ids, zs)

    np.save('/srv/scratch/mltidal/fracs_gendata_deepv2.npy', fracs)

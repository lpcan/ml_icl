"""
Automatically measure the training dataset using multiprocessing
"""

import measurement_helpers
from measurement_helpers import stretch

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
from photutils.segmentation import detect_sources

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import tqdm
import sys

import skimage

def k_corr(z):
    # Equation from Chilingarian et al. assuming g-r colour of 0.7
    return 1.111*z - 1.101*z**2 - 75.050*z**3 + 295.888*z**4 - 295.390*z**5

from scipy.interpolate import bisplrep, bisplev
from scipy.ndimage import zoom 

def background_estimate_2d(cutout, bad_mask):
    spacing = 10
    # Fit and subtract off any gradients in the image
    box_size = 224 // 14
    bkg_initial = Background2D(cutout, box_size=box_size)
    mesh = bkg_initial.background_mesh

    Y, X = np.ogrid[:mesh.shape[0], :mesh.shape[1]]

    box = (X < mesh.shape[1] - 1) & (X > 0) & (Y < mesh.shape[0]) & (Y > 0)

    vals = mesh[~box]

    box_square = np.argwhere(~box)
    tck = bisplrep(*box_square.T, vals)
    znew = bisplev(np.arange(14), np.arange(14), tck)
    bkg = zoom(znew, np.array(cutout.shape) / np.array([14, 14]), mode='reflect')

    # Subtract off any remaining constant background
    Y, X = np.ogrid[:224, :224]
    radii = np.expand_dims(np.arange(-1, 112, spacing), (1, 2))
    
    distances = np.expand_dims(np.sqrt((X - 112)**2 + (Y - 112)**2), 0)
    
    circles = (distances <= radii)
    annuli = np.diff(circles, axis=0)

    cutout_ex = np.expand_dims(cutout, axis=0)

    cutout_masked = cutout_ex * annuli

    masks = (annuli * ~bad_mask)

    means = []

    for i, annulus in enumerate(cutout_masked):
        mean, _, _ = sigma_clipped_stats(annulus[masks[i]])
        means.append(mean)
    
    mean_of_means = 2 * np.min(means) / 3

    bkg = bkg + mean_of_means

    return bkg

def calc_icl_frac(cutout, bad_mask, z, return_mask=False):
    if bad_mask[112,112]:
        # Bright star mask extends over the centre of the image, get rid of it
        bad_mask = np.zeros_like(bad_mask, dtype=bool) 

    # Background estimate
    # bkg = measure_icl.background_estimate(cutout, zs[key], cosmo)
    bkg = background_estimate_2d(cutout, bad_mask) 
    bkg_subtracted = cutout - bkg

    bkg_subtracted = bkg_subtracted * ~bad_mask

    mid = (cutout.shape[0] // 2, cutout.shape[1] // 2)

    # Mask all galaxies other than the BCG
    threshold = measurement_helpers.sb2counts(26 + 10 * np.log10(1 + z) + k_corr(z))
    segmented = detect_sources(bkg_subtracted, threshold=threshold, npixels=10)
    labels = segmented.data
    eroded = skimage.morphology.binary_erosion((labels > 0), np.ones((3,3)))
    markers, _ = scipy.ndimage.label(eroded)
    distance = scipy.ndimage.distance_transform_edt(labels > 0)
    watershed_labels = skimage.segmentation.watershed(-distance, markers, mask=(labels > 0))
    # Need to renumber watershed labels to prevent two different objects with same label
    deblended = np.where(watershed_labels > 0, watershed_labels + np.max(labels), labels)

    # Calculate new, smaller masks for central galaxies
    # threshold = measurement_helpers.sb2counts(26 + 10 * np.log10(1 + z) + k_corr(z))
    # deblended = detect_sources(bkg_subtracted, threshold=threshold, npixels=10)
    # Combine to create our new mask
    combined_labels = labels

    # Unsharp mask the image for hot mask creation
    kernel = Gaussian2DKernel(2) 
    conv_img = convolve(np.array(cutout), kernel)
    unsharp = cutout - conv_img

    # Create hot mask
    hot_mask_bkg = Background2D(unsharp, box_size=16).background
    combined_mask = (combined_labels > 0)
    hot_labels = measurement_helpers.create_hot_labels(unsharp, combined_mask, background=hot_mask_bkg, npixels=3)
    hot_mask = (hot_labels > 0)

    # Get the BCG label
    bcg_label = combined_labels[mid[0], mid[1]]

    # Coordinates of points that are part of the BCG
    # pts = np.array(np.argwhere(combined_labels == bcg_label))

    # # Find points that are furthest apart
    # candidates = pts[scipy.spatial.ConvexHull(pts).vertices]
    # dist_mat = scipy.spatial.distance_matrix(candidates, candidates)
    # i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    # pt1 = candidates[i]
    # pt2 = candidates[j]

    # size = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

    # Make sure that the radius is >=100kpc
    # radius = np.max((size, cosmo.arcsec_per_kpc_proper(zs[num]).value * 100 * 1/0.168))
    radius = cutout.shape[0] / 2
    # radius = size * 1.5

    # Generate the mask
    centre = (cutout.shape[1] // 2, cutout.shape[0] // 2)
    Y, X = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
    dist_from_centre = np.sqrt((X-centre[0])**2 + (Y-centre[1])**2)
    circ_mask = dist_from_centre <= radius

    # Get the member mask
    non_member_labels = [label for label in np.unique(deblended)
                         if label != bcg_label and label != 0]
    enlarged = measurement_helpers.enlarge_mask(np.isin(deblended, non_member_labels), 
                                        sigma=0.3)
    enlarged = enlarged * ~(deblended == bcg_label)
    
    non_member_mask = enlarged | hot_mask

    # member_mask = (combined_labels == bcg_label)

    # Calculate surface brightness limit
    Y, X = np.ogrid[:224, :224]
    box = (X < 224 - 16) & (X > 16) & (Y < 224-16) & (Y > 16)
    edges = ~box
    _, _, stddev = sigma_clipped_stats(bkg_subtracted, mask=(bad_mask & edges))
    sb_lim = -2.5 * np.log10(3 * stddev/(0.168 * 10)) + 2.5 * np.log10(63095734448.0194)

    # Convert image from counts to surface brightness
    np.seterr(invalid='ignore', divide='ignore')
    sb_img = measurement_helpers.counts2sb(bkg_subtracted, 0)

    # Mask values below surface brightness limit
    sb_img[sb_img >= sb_lim] = np.nan

    # Mask above the surface brightness threshold
    # threshold = 25 + 10 * np.log10(1 + z)
    # mask = sb_img >= threshold
    mask = sb_img >= 26 + 10 * np.log10(1 + z) + measurement_helpers.k_corr(z)
    # mask = ~combined_mask & ~hot_mask
    
    # Close the mask
    # mask = binary_closing(mask)

    # Convert the SB image back to counts
    counts_img = measurement_helpers.sb2counts(sb_img) 

    # Close the nans to try and get rid of the noise that contributes to the ICL?
    nans = np.isnan(counts_img)
    nans = binary_closing(nans)
    not_nans = ~nans

    # Display the final image
    masked_img = counts_img * ~non_member_mask * circ_mask * not_nans# * ~hot_mask

    icl = np.nansum(masked_img * mask)
    total = np.nansum(masked_img)

    if return_mask:
        return icl, total, icl / total, counts_img * not_nans * ~hot_mask * mask
    else:
        return icl, total, icl / total

def run_requested_keys(args):
    keys, length, zs, new_ids = args

    cutouts = h5py.File('/srv/scratch/z5214005/generated_data_iclnoise.hdf')
    masks = h5py.File('/srv/scratch/z5214005/lrg_cutouts_300kpc_resized.hdf')

    # Find the shared memory and create a numpy array interface
    shmem = SharedMemory(name=f'iclbuf', create=False)
    fracs = np.ndarray((3, int(length/3)), buffer=shmem.buf, dtype=np.float64)

    # Parameters
    cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)

    for key in keys: 
        cutout = np.array(cutouts[str(new_ids[key])]['HDU0']['DATA'])
        # bad_mask = np.array(masks[str(new_ids[key])]['MASK']).astype(bool)
        bad_mask = np.array(masks[str(new_ids[key])]['HDU1']['DATA']).astype(bool) 
        z = zs[key]

        icl, total, frac = calc_icl_frac(cutout, bad_mask, z)
        if icl is None:
            print(key)

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
    tbl = ascii.read('/srv/scratch/z5214005/lrgs_sampled_1405.tbl')
    zs = tbl['z']
    new_ids = tbl['new_ids']

    fracs = calc_icl_frac_parallel(new_ids, zs)

    np.save('/srv/scratch/mltidal/fracs_gendata_maskbcgonly.npy', fracs)

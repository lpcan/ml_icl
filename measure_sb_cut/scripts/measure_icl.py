"""
Measure the ICL fraction in the images within a redshift-dependent radius 
with background subtraction.
"""

from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats
import h5py
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import os
from photutils.background import Background2D
import scipy
from scipy import spatial
from scipy.interpolate import CloughTocher2DInterpolator
import sys
import tqdm
import warnings
from astropy.utils.exceptions import AstropyWarning

warnings.simplefilter('ignore', category=AstropyWarning)

from astropy.convolution import Gaussian2DKernel, convolve
from astropy.table import join
from photutils.segmentation import detect_threshold, detect_sources

def background_estimate(cutout, z, cosmo, mask=None):
    """
    Returns an estimate of the 2D background of `cutout`. The background is 
    measured in boxes of size 50px around the edges of the image, and the
    background is interpolated over the entire image. 
    """
    # Run photutil's Background2D for the low resolution grid
    box_size = cutout.shape[0] // 14
    bkg_initial = Background2D(cutout, box_size=box_size, mask=mask)
    mesh = bkg_initial.background_mesh
    
    # Extract just the edges of the mesh
    Y, X = np.ogrid[:mesh.shape[0], :mesh.shape[1]]
    y_cen, x_cen = (mesh.shape[0] // 2, mesh.shape[1] // 2) 

    box_cen = (box_size - 1) / 2.0

    # Create a mask to cover the internal 350 kpc
    px_dist = cosmo.arcsec_per_kpc_proper(z) * 350 * 1/0.168
    size = int(np.ceil(px_dist.value / box_size))
    box = (X > x_cen - size) & (X < x_cen + size) & (Y > y_cen - size) & (Y < y_cen + size)

    # Get values from the mesh corresponding to these coordinates
    vals = mesh[~box]

    # Array of coordinates in image units
    real_square = np.argwhere(~box) * box_size + box_cen

    # Interpolate between the edges of the square
    interp = CloughTocher2DInterpolator(real_square, vals)
    x = np.arange(np.min(real_square[:,0]), np.max(real_square[:,0]))
    y = np.arange(np.min(real_square[:,1]), np.max(real_square[:,1]))
    x, y = np.meshgrid(x, y) # 2D grid
    z = interp(x, y).T

    # Expand the image to the correct size
    edge_widths = (
        (int(np.abs(cutout.shape[0] - z.shape[0])/2), 
        int(np.ceil(np.abs(cutout.shape[0] - z.shape[0])/2))),
        (int(np.abs(cutout.shape[1] - z.shape[1])/2),
        int(np.ceil(np.abs(cutout.shape[1] - z.shape[1])/2)))
    )
    bkg = np.pad(z, pad_width=edge_widths, mode='reflect')

    # Return background estimation
    return bkg

def create_circular_mask(z, img, cosmo, radius):
    """
    Returns a circular mask of 130kpc radius for given `z` and given cosmology
    `cosmo` for image `img`.
    """
    # Calculate the radius in pixels
    arcsec_to_px = 1/0.168
    radius_px = (cosmo.arcsec_per_kpc_proper(z) * radius).value * arcsec_to_px
    
    # Generate the mask
    centre = (img.shape[1] // 2, img.shape[0] // 2)
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist_from_centre = np.sqrt((X-centre[0])**2 + (Y-centre[1])**2)

    mask = dist_from_centre <= radius_px

    return mask

def get_member_locs(idx, merged, cutout_shape):
    # Get cluster location
    cluster_ra = merged[merged['ID'] == idx]['RA_cl'][0]
    cluster_dec = merged[merged['ID'] == idx]['Dec_cl'][0]

    # Get the cluster members
    c_members = merged[merged['ID'] == idx]
    ras = c_members['RA']
    decs = c_members['Dec']
    
    # Get offsets in degrees and convert to pixels
    ra_offsets = ras - cluster_ra
    dec_offsets = decs - cluster_dec
    x_offsets = ra_offsets * 3600 / 0.168
    y_offsets = dec_offsets * 3600 / 0.168

    # Get pixel locations
    centre = (cutout_shape[1] // 2, cutout_shape[0] // 2)
    x_locs_all = centre[0] - x_offsets # ra decreases from left to right
    y_locs_all = centre[1] + y_offsets
    x_locs = x_locs_all[(x_locs_all >= 0) & (x_locs_all < cutout_shape[1]) & (y_locs_all >= 0) & (y_locs_all < cutout_shape[0])]
    y_locs = y_locs_all[(x_locs_all >= 0) & (x_locs_all < cutout_shape[1]) & (y_locs_all >= 0) & (y_locs_all < cutout_shape[0])]
    
    return (x_locs, y_locs)

def counts2sb(counts, z):
    return 2.5 * np.log10(63095734448.0194 / counts) + 5 * np.log10(0.168) - 10 * np.log10(1+z)

def sb2counts(sb): # without reaccounting for dimming
    return 10**(-0.4*(sb - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168)))

def create_cold_labels(cutout, bad_mask):
    # First apply the bright star mask
    mask_input = np.array(cutout)
    mask_input[bad_mask] = np.nan

    # Smooth the image to help detect extended bright sources
    kernel = Gaussian2DKernel(5)
    kernel.normalize()
    convolved = convolve(mask_input, kernel)
    
    # Background estimate
    background = Background2D(cutout, box_size=cutout.shape[0] // 14, mask=bad_mask).background
    
    # Detect threshold
    threshold = detect_threshold(mask_input, nsigma=1.1, background=background, mask=bad_mask)

    # Detect sources
    segm = detect_sources(convolved, threshold=threshold, npixels=40)

    return segm.data

def create_hot_labels(unsharp, bad_mask, background):
    # Detect threshold
    threshold = detect_threshold(unsharp, nsigma=1.2, background=background, mask=bad_mask)
    
    # Detect sources
    segm = detect_sources(unsharp, threshold, npixels=7, mask=bad_mask)

    return segm.data

def enlarge_mask(labels, sigma=1):
    # Make the mask larger by convolving
    kernel = Gaussian2DKernel(sigma)
    mask = convolve(labels, kernel).astype(bool)

    return mask

def calc_icl_frac(args):
    """
    Calculate the ratio of light below the threshold to the total light in the 
    image to get a rough estimate of the icl fraction.
    """
    keys, length, zs, richness, merged = args

    base_path = os.path.dirname(__file__)
    cutouts = h5py.File('/srv/scratch/z5214005/cutouts_550.hdf') #'/../processed/cutouts.hdf')

    # Find the shared memory and create a numpy array interface
    shmem = SharedMemory(name=f'iclbuf', create=False)
    fracs = np.ndarray((3, int(length/3)), buffer=shmem.buf, dtype=np.float64)

    maskshmem = SharedMemory(name='maskbuf', create=False)
    masks = np.ndarray((int(length/3), 2208, 2208), buffer=maskshmem.buf, dtype=np.float64)

    # Parameters
    cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)

    # Go through the assigned clusters and calculate the icl fraction
    for key in keys:
        if key == 4:
            fracs[:,key] = np.nan
            continue # bad image
        # Get the image data
        cutout = np.array(cutouts[str(key)]['HDU0']['DATA'])

        # Create the bright objects mask
        BAD = 1
        BRIGHT_OBJECT = 512
        mask_data = np.array(cutouts[str(key)]['HDU1']['DATA']).astype(int)
        bad_mask = np.array(mask_data & (BAD | BRIGHT_OBJECT)).astype(bool)

        # Create the circular mask
        radius = np.min((richness[key] * 17 - 125, 350))
        circ_mask = create_circular_mask(zs[key], cutout, cosmo, radius)

        # Check if this cutout should be excluded because too much is masked
        inner_frac_masked = np.sum(bad_mask * circ_mask) / np.sum(circ_mask)
        mid = (bad_mask.shape[0] // 2, bad_mask.shape[1] // 2)

        if inner_frac_masked > 0.2 or bad_mask[mid]:
            # >20% of inner region masked or middle pixel is masked
            fracs[:,key] = np.nan
            continue

        # Background estimate
        bkg = background_estimate(cutout, zs[key], cosmo, mask=bad_mask)
        bkg_subtracted = cutout - bkg

        # Segment the image
        cold_labels = create_cold_labels(cutout, bad_mask)

        # Create the cold mask from the labels
        cold_mask = enlarge_mask(cold_labels, sigma=2)

        # Unsharp mask the image for hot mask creation
        kernel = Gaussian2DKernel(5)
        conv_img = convolve(np.array(cutout), kernel)
        unsharp = np.array(cutout) - conv_img

        # Create the hot mask
        hot_mask_bkg = background_estimate(unsharp, z=zs[key], cosmo=cosmo, mask=(bad_mask))
        hot_labels = create_hot_labels(unsharp, (bad_mask + cold_mask), background=hot_mask_bkg)
        hot_mask = enlarge_mask(hot_labels, sigma=1)

        # Mark the cluster members in the cold mask
        x_locs, y_locs = get_member_locs(int(key), merged, cutout.shape)
        c_members = cold_labels[y_locs.astype(int), x_locs.astype(int)]

        # Mask out non cluster members
        member_mask = np.isin(cold_labels, c_members) | (cold_labels == 0)
        non_member_mask = ~member_mask
        non_member_mask = enlarge_mask(non_member_mask, sigma=2)
        non_member_mask = non_member_mask + hot_mask
        # Member mask keeps the background and cluster members
        member_mask = ~non_member_mask

        # Make the circular mask
        # Get the BCG's label
        mid = (cutout.shape[0] // 2, cutout.shape[1] // 2)
        bcg_label = cold_labels[mid[0], mid[1]]

        # Coordinates of points that are part of the BCG
        pts = np.array(np.argwhere(cold_labels == bcg_label))

        # Find points that are furthest apart
        candidates = pts[spatial.ConvexHull(pts).vertices]
        dist_mat = spatial.distance_matrix(candidates, candidates)
        i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        pt1 = candidates[i]
        pt2 = candidates[j]

        size = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

        # Make sure that the radius is >=100kpc
        radius = np.max((size, cosmo.arcsec_per_kpc_proper(zs[key]).value * 100 * 1/0.168))

        # Generate the mask
        centre = (cutout.shape[1] // 2, cutout.shape[0] // 2)
        Y, X = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
        dist_from_centre = np.sqrt((X-centre[0])**2 + (Y-centre[1])**2)
        circ_mask = dist_from_centre <= radius

        # Check again for exclusion
        inner_frac_masked = np.sum(bad_mask * circ_mask) / np.sum(circ_mask)
        if inner_frac_masked > 0.2:
            # >20% of inner region masked or middle pixel is masked
            fracs[:,key] = np.nan
            continue

        # Calculate surface brightness limit (from Cristina's code (Roman+20))
        _, _, stddev = sigma_clipped_stats(bkg_subtracted, mask=bad_mask)
        sb_lim = -2.5 * np.log10(3*stddev/(0.168 * 10)) + 2.5 * np.log10(63095734448.0194)

        # Convert image from counts to surface brightness
        np.seterr(invalid='ignore', divide='ignore')
        sb_img = counts2sb(bkg_subtracted, 0)

        # Mask out the values below surface brightness limit
        sb_img[sb_img >= sb_lim] = np.nan

        # Mask above the surface brightness threshold
        threshold = 25 + 10 * np.log10(1 + zs[int(key)])
        mask = sb_img > threshold

        # Convert the SB image back to counts
        counts_img = sb2counts(sb_img)

        masked_img = counts_img * ~bad_mask * member_mask * circ_mask

        # Save the mask
        l = np.max((np.min(np.nonzero(circ_mask)), 0))
        h = np.min((np.max(np.nonzero(circ_mask)), np.min(cutout.shape)-1))
        masks[key,0:h-l,0:h-l] = (cutout * ~bad_mask * member_mask * circ_mask * mask)[l:h,l:h]

        fracs[0,key] = np.nansum(masked_img * mask)
        fracs[1,key] = np.nansum(masked_img)
        fracs[2,key] = fracs[0,key] / fracs[1,key]

    return

def calc_icl_frac_parallel(cutouts, zs, richness, merged):
    """
    Use multiprocessing to divide the cutouts among available cores and 
    calculate the ICL fractions.
    """
    # Use all available cores
    cores = mp.cpu_count()

    # Divide the keys up into 20 
    jobs = np.array_split(np.arange(len(cutouts.keys())), 20)
    length = len(cutouts.keys()) * 3
    args = [(j, length, zs, richness, merged) for j in jobs]

    exit = False
    try:
        # Set up the shared memory
        global mem_id
        mem_id = 'iclbuf'
        nbytes = len(cutouts.keys()) * 3 * np.float64(1).nbytes 
        iclmem = SharedMemory(name='iclbuf', create=True, size=nbytes)
        nbytes = len(cutouts.keys()) * 2208**2 * np.float64(1).nbytes # Max size of circ mask is 2208^2 px
        maskmem = SharedMemory(name='maskbuf', create=True, size=nbytes)

        # Start a new process for each task
        ctx = mp.get_context()
        pool = ctx.Pool(processes=cores, maxtasksperchild=1)
        try:
            for _ in tqdm.tqdm(pool.imap_unordered(calc_icl_frac, args, chunksize=1), total=len(jobs)):
                pass
        except KeyboardInterrupt:
            print('Caught kbd interrupt')
            pool.close()
            exit = True
        else:
            pool.close()
            pool.join()
            # Copy the result
            result = np.ndarray((3, len(cutouts.keys())), buffer=iclmem.buf,
                                dtype=np.float64).copy()
            masks = np.ndarray((len(cutouts.keys()), 2208, 2208), buffer=maskmem.buf,
                               dtype=np.float64).copy()
    finally:
        # Close the shared memory
        iclmem.close()
        iclmem.unlink()
        maskmem.close()
        maskmem.unlink()
        if exit:
            sys.exit(1)
    return result, masks

if __name__ == '__main__':
    # Parameters and things
    stddev = 0.017359
    cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)
    base_path = os.path.dirname(__file__)

    # Load the cutouts and the corresponding redshifts
    cutouts = h5py.File('/srv/scratch/z5214005/cutouts_550.hdf') #'/../processed/cutouts.hdf')
    tbl = ascii.read('/srv/scratch/z5214005/camira_final.tbl', #'/../processed/camira_final.tbl', 
                    names=['ID', 'Name', 'RA [deg]', 'Dec [deg]', 'z', 'Richness', 'BCG z'])
    zs = tbl['z']
    richness = tbl['Richness']
    
    # Load the cluster member catalogue
    members = ascii.read('/srv/scratch/z5214005/camira_s20a_dud_member.dat', 
                        names=['RA_cl', 'Dec_cl', 'Richness', 'z_cl', 'RA', 'Dec', 'M', 'w'])

    # Match this catalogue to the cluster catalogue
    merged = join(members, tbl, keys_left=['RA_cl', 'Dec_cl'], keys_right=['RA [deg]', 'Dec [deg]'])
    merged = merged['ID', 'Name', 'RA_cl', 'Dec_cl', 'z_cl', 'RA', 'Dec']

    fracs, masks = calc_icl_frac_parallel(cutouts, zs, richness, merged)

    ranked = np.argsort(fracs[2])
    mask = ~np.isnan(fracs[2][ranked]) # Don't include nans in the top 5
    top_5 = ranked[mask][-5:][::-1]
    bottom_5 = ranked[:5]
    print(list(zip(top_5, fracs[2][top_5])))
    print(list(zip(bottom_5, fracs[2][bottom_5])))

    np.save('/srv/scratch/z5214005/bcgicl_fracs.npy', fracs) 
    np.save('/srv/scratch/z5214005/bcgicl_masks.npy', masks)

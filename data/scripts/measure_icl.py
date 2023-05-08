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
from photutils.aperture import CircularAnnulus
from photutils.background import Background2D
from scipy.interpolate import CloughTocher2DInterpolator
import sys
import tqdm

def background_estimate(cutout, cosmo, mask=None):
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

    # Create a mask to cover the internal 250 kpc
    px_dist = cosmo.arcsec_per_kpc_proper(0.2565) * 250 * 1/0.168
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

def create_circular_mask(z, img, cosmo):
    """
    Returns a circular mask of 130kpc radius for given `z` and given cosmology
    `cosmo` for image `img`.
    """
    # Calculate the radius in pixels
    arcsec_to_px = 1/0.168
    radius = (cosmo.arcsec_per_kpc_proper(z) * 200).value * arcsec_to_px
    
    # Generate the mask
    centre = (img.shape[1] // 2, img.shape[0] // 2)
    Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
    dist_from_centre = np.sqrt((X-centre[0])**2 + (Y-centre[1])**2)

    mask = dist_from_centre <= radius

    return mask

def counts2sb(counts, z):
    return 2.5 * np.log10(63095734448.0194 / counts) + 5 * np.log10(0.168) - 10 * np.log10(1+z)

def sb2counts(sb): # without reaccounting for dimming
    return 10**(-0.4*(sb - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168)))

def calc_icl_frac(args):
    """
    Calculate the ratio of light below the threshold to the total light in the 
    image to get a rough estimate of the icl fraction.
    """
    keys, length, zs = args
    base_path = os.path.dirname(__file__)
    cutouts = h5py.File(base_path + '/../../cutouts_550.hdf') #'/../processed/cutouts.hdf')

    # Find the shared memory and create a numpy array interface
    shmem = SharedMemory(name=f'iclbuf', create=False)
    fracs = np.ndarray(length, buffer=shmem.buf, dtype=np.float64)

    # Parameters
    cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)

    # Go through the assigned clusters and calculate the icl fraction
    for key in keys:
        # Get the image data
        cutout = np.array(cutouts[str(key)]['HDU0']['DATA'])

        # Create the bright objects mask
        BAD = 1
        BRIGHT_OBJECT = 512
        mask_data = np.array(cutouts[str(key)]['HDU1']['DATA']).astype(int)
        bad_mask = np.array(mask_data & (BAD | BRIGHT_OBJECT)).astype(bool)

        # Create the circular mask
        circ_mask = create_circular_mask(zs[key], cutout, cosmo)

        # Check if this cutout should be excluded because too much is masked
        inner_frac_masked = np.sum(bad_mask * circ_mask) / np.sum(circ_mask)
        mid = (bad_mask.shape[0] // 2, bad_mask.shape[1] // 2)

        if inner_frac_masked > 0.3 or bad_mask[mid]:
            # >30% of inner region masked or middle pixel is masked
            fracs[key] = np.nan
            continue

        # First background estimate
        bkg = background_estimate(cutout, cosmo, mask=bad_mask)
        bkg_subtracted = cutout - bkg

        # # Secondary background estimate via radial profile
        # fluxes = []
        # px_threshold = (cosmo.arcsec_per_kpc_proper(zs[key]) * 350).value * 1/0.168 # Measure >350kpc away from BCG
        # centre = (bkg_subtracted.shape[0] // 2, bkg_subtracted.shape[1] // 2)
        # r_in = px_threshold
        # r_out = px_threshold + 20

        # while r_out < np.min(centre):
        #     # Create the circular aperture
        #     aperture = CircularAnnulus(centre, r_in=r_in, r_out=r_out)
        #     mask = aperture.to_mask()

        #     # Get the image and mask data inside this annulus
        #     annulus = mask.cutout(bkg_subtracted, fill_value=np.nan)
        #     mask_cutout = mask.cutout(bad_mask, fill_value=False)

        #     # Calculate the sigma clipped average of values in the annulus
        #     mean, _, _ = sigma_clipped_stats(annulus, mask=mask_cutout)
        #     fluxes.append(mean)

        #     # Update the radii
        #     r_in += 20
        #     r_out += 20

        # sky_value = np.nanmedian(fluxes)

        # bkg_subtracted = bkg_subtracted - sky_value

        # Calculate surface brightness limit (from Cristina's code (Roman+20))
        _, _, stddev = sigma_clipped_stats(bkg_subtracted, mask=bad_mask)
        sb_lim = -2.5 * np.log10(3*stddev/(0.168 * 120)) + 2.5 * np.log10(63095734448.0194)

        # Mask the image
        masked_img = bkg_subtracted * circ_mask

        # Convert image from counts to surface brightness
        np.seterr(invalid='ignore', divide='ignore')
        sb_img = counts2sb(masked_img, 0)

        # Mask out the values below surface brightness limit
        sb_img[sb_img >= sb_lim] = np.nan

        # Mask above the surface brightness threshold
        threshold = 25 + 10 * np.log10(1 + zs[int(key)])
        mask = sb_img > threshold

        # Convert the SB image back to counts
        counts_img = sb2counts(sb_img)

        # Weight image using the inverse variance
        weighted_img = counts_img * ~bad_mask

        fracs[key] = np.sum((weighted_img * mask)[~np.isnan(counts_img)]) / np.sum((weighted_img)[~np.isnan(counts_img)])

    return

def calc_icl_frac_parallel(cutouts, zs):
    """
    Use multiprocessing to divide the cutouts among available cores and 
    calculate the ICL fractions.
    """
    # Use all available cores
    cores = mp.cpu_count()

    # Divide the keys up into 20 
    jobs = np.array_split(np.arange(len(cutouts.keys())), 20)
    length = len(cutouts.keys())
    args = [(j, length, zs) for j in jobs]

    exit = False
    try:
        # Set up the shared memory
        global mem_id
        mem_id = 'iclbuf'
        nbytes = len(cutouts.keys()) * np.float64(1).nbytes 
        iclmem = SharedMemory(name='iclbuf', create=True, size=nbytes)

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
            result = np.ndarray((len(cutouts.keys())), buffer=iclmem.buf,
                                dtype=np.float64).copy()
    finally:
        # Close the shared memory
        iclmem.close()
        iclmem.unlink()
        if exit:
            sys.exit(1)
    return result 

if __name__ == '__main__':
    # Parameters and things
    stddev = 0.017359
    cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)
    base_path = os.path.dirname(__file__)

    # Load the cutouts and the corresponding redshifts
    cutouts = h5py.File(base_path + '/../../cutouts_550.hdf') #'/../processed/cutouts.hdf')
    tbl = ascii.read(base_path + '/../../dud_only.tbl', #'/../processed/camira_final.tbl', 
                    names=['ID', 'Name', 'RA', 'Dec', 'z', 'Richness', 'BCG z'])
    zs = tbl['z']

    fracs = calc_icl_frac_parallel(cutouts, zs)

    ranked = np.argsort(fracs)
    mask = ~np.isnan(fracs[ranked]) # Don't include nans in the top 5
    top_5 = ranked[mask][-5:][::-1]
    bottom_5 = ranked[:5]
    print(list(zip(top_5, fracs[top_5])))
    print(list(zip(bottom_5, fracs[bottom_5])))

    np.save('fracs.npy', fracs) 
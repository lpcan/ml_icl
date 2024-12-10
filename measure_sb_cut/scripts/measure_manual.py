"""
Measure the ICL fraction of the finetuning dataset with the surface brightness 
threshold method.
"""

from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii
from astropy.stats import sigma_clipped_stats
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from photutils.background import Background2D
from photutils.segmentation import deblend_sources, detect_sources
import scipy
from scipy.interpolate import bisplrep, bisplev
from scipy.ndimage import zoom
import skimage
import sys
from unagi import hsc

import measurement_helpers

BAD = 1
NO_DATA = (1 << 8)
BRIGHT_OBJECT = (1 << 9)
cmap = matplotlib.colormaps['viridis']
cmap.set_bad(cmap(0))
cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)
pdr2 = hsc.Hsc(dr='pdr2', rerun='pdr2_dud',config_file=None)

def get_xy(ras, decs, original_shape, header, size=224):
    """
    Get the pixel locations of the cluster members
    """

    member_ras = ras
    member_decs = decs
        
    w = wcs.WCS(header)

    coords = SkyCoord(member_ras, member_decs, unit='deg')
    coords = wcs.utils.skycoord_to_pixel(coords, w)
    x_coords = coords[0] / (original_shape[1] / size)
    y_coords = coords[1] / (original_shape[0] / size)

    return x_coords, y_coords

def get_members(cluster, original_shape, header, size=224, halfsize_kpc=300):
    """
    Return cluster member pixel locations
    """

    # cluster = tbl[idx]
    half_size = (cosmo.arcsec_per_kpc_proper(cluster['z_cl']) * halfsize_kpc).value

    query = f"""SELECT
                object_id
                , ra
                , dec
                , r_cmodel_mag
                , g_cmodel_mag
                , photoz_best
                , photoz_err68_min
                , photoz_err68_max
                , photoz_std_best
                , photoz_err95_min
                , photoz_err95_max
                FROM
                pdr2_dud.forced
                JOIN pdr2_dud.photoz_demp USING (object_id)
                WHERE
                    coneSearch(coord, {cluster['RA [deg]']}, {cluster['Dec [deg]']}, {half_size})
            ;"""

    result = pdr2.sql_query(query, verbose=True)

    z = cluster['z_cl']
    photoz_members = result[(result['photoz_best'] < (z + 3*0.05*(1+z))) & (result['photoz_best'] > (z - 3*0.05*(1+z)))]
    coords = get_xy(photoz_members['ra'], photoz_members['dec'], original_shape, header, size=size)
    
    return coords

def background_rms(cutout, bad_mask):
    rms_mesh = Background2D(cutout, box_size=19, mask=bad_mask, exclude_percentile=20).background_rms_mesh_masked
    Y, X = np.ogrid[:rms_mesh.shape[0], :rms_mesh.shape[1]]
    box = (X < rms_mesh.shape[1] - 1) & (X > 0) & (Y < rms_mesh.shape[0] - 1) & (Y > 0)
    rms = np.nanmedian(rms_mesh[~box])

    return rms
    
def radial_profile(cutout, mask):
    shape = cutout.shape
    spacing = 10
    # Subtract off any remaining constant background
    Y, X = np.ogrid[:shape[0], :shape[1]]
    size = min(shape) 
    radii = np.expand_dims(np.arange(-1, size/2, spacing), (1, 2))

    distances = np.expand_dims(np.sqrt((X - size/2)**2 + (Y - size/2)**2), 0)

    circles = (distances <= radii)
    annuli = np.diff(circles, axis=0)

    cutout_ex = np.expand_dims(cutout, axis=0)

    cutout_masked = cutout_ex * annuli

    masks = (annuli * ~mask)

    means = []
    stds = []

    for i, annulus in enumerate(cutout_masked):
        mean, _, std = sigma_clipped_stats(annulus[masks[i]])
        means.append(mean)
        stds.append(std)

    means = np.array(means)
    stds = np.array(stds)

    return means, stds

def create_non_member_mask(cutout, cluster, mask, original_shape, member_coords, aggressive=True, sigma=0.3):
    # Segment and deblend the cutout
    threshold = measurement_helpers.sb2counts(26 + 10 * np.log10(1 + cluster['z_cl']) + measurement_helpers.k_corr(cluster['z_cl']))
    segmented = detect_sources(cutout, threshold=threshold, npixels=npixels_cold)
    deblended = deblend_sources(cutout, segmented, npixels=npixels_cold, progress_bar=False).data

    if not aggressive:
        labels = segmented.data
        eroded = skimage.morphology.binary_erosion((labels > 0), np.ones((3,3)))
        markers, _ = scipy.ndimage.label(eroded)
        distance = scipy.ndimage.distance_transform_edt(labels > 0)
        watershed_labels = skimage.segmentation.watershed(-distance, markers, mask=(labels > 0))
        # Need to renumber watershed labels to prevent two different objects with same label
        deblended = np.where(watershed_labels > 0, watershed_labels + np.max(labels), labels)

    x_coords, y_coords = member_coords

    member_labels = [deblended[int(y_coords[i]), int(x_coords[i])]
                     for i in range(len(x_coords))
                     if deblended[int(y_coords[i]), int(x_coords[i])] != 0]
    non_member_labels = [label for label in np.unique(deblended)
                         if label not in member_labels and label != 0]
    enlarged = measurement_helpers.enlarge_mask(np.isin(deblended, non_member_labels), 
                                        sigma=sigma)
    enlarged = enlarged * ~np.isin(deblended, member_labels)
    
    # Add in the hot mask
    kernel = Gaussian2DKernel(2)
    conv_img = convolve(np.array(cutout), kernel)
    unsharp = cutout - conv_img
    hot_mask_bkg = Background2D(unsharp, box_size=16).background
    combined_mask = (deblended > 0)
    hot_labels = measurement_helpers.create_hot_labels(unsharp, combined_mask,
                                               background=hot_mask_bkg, npixels=3)
    hot_mask = (hot_labels > 0)

    return enlarged | hot_mask

def background_estimate_2d(cutout, bad_mask, constant=True, multiplier=1, mask_initial=False):
    shape = cutout.shape
    size = min(shape)
    spacing = 10
    # Fit and subtract off any gradients in the image
    box_size = size // 14
    if not mask_initial:
        bkg_initial = Background2D(cutout, box_size=box_size)
    else:
        bkg_initial = Background2D(cutout, box_size=box_size, mask=bad_mask)
        
    mesh = bkg_initial.background_mesh

    Y, X = np.ogrid[:mesh.shape[0], :mesh.shape[1]]

    box = (X < mesh.shape[1] - 1) & (X > 0) & (Y < mesh.shape[0]) & (Y > 0)

    vals = mesh[~box]

    box_square = np.argwhere(~box)
    tck = bisplrep(*box_square.T, vals)
    znew = bisplev(np.arange(14), np.arange(14), tck)
    bkg = zoom(znew, np.array(cutout.shape) / np.array([14, 14]), mode='reflect')

    # Subtract off any remaining constant background
    Y, X = np.ogrid[:shape[0], :shape[1]]
    radii = np.expand_dims(np.arange(-1, size//2, spacing), (1, 2))

    distances = np.expand_dims(np.sqrt((X - size//2)**2 + (Y - size//2)**2), 0)

    circles = (distances <= radii)
    annuli = np.diff(circles, axis=0)

    cutout_ex = np.expand_dims(cutout, axis=0)

    cutout_masked = cutout_ex * annuli

    masks = (annuli * ~bad_mask)

    means = []

    for i, annulus in enumerate(cutout_masked):
        mean, _, _ = sigma_clipped_stats(annulus[masks[i]])
        means.append(mean)

    mean_of_means = multiplier * np.min(means) / 3

    if constant:
        bkg = np.ones_like(bkg) * mean_of_means
    else:
        bkg = bkg + mean_of_means

    return bkg

def calc_icl_frac(cutout, bad_mask, idx, z, original_shape, tbl, header, constant=True, seg_threshold=26, aggressive=True):
    if bad_mask[112,112]:
        # Bright star mask extends over the centre of the image, get rid of it
        bad_mask = np.zeros_like(bad_mask, dtype=bool)

    plt.figure(figsize=(12,4))
    plt.subplot(141)
    plt.imshow(measurement_helpers.stretch(cutout) * ~bad_mask, cmap=cmap)

    # Background estimate
    bkg = background_estimate_2d(cutout, bad_mask, constant=constant) 
    bkg_subtracted = cutout - bkg

    bkg_subtracted = bkg_subtracted * ~bad_mask

    plt.subplot(142)
    plt.imshow(measurement_helpers.stretch(bkg_subtracted), cmap=cmap)

    x_loc, y_loc = get_members(tbl[idx], original_shape, header)

    plt.scatter(x_loc, y_loc, s=40, edgecolor='white', facecolor='none')

    non_member_mask = create_non_member_mask(bkg_subtracted, tbl[idx], mask=bad_mask, original_shape=original_shape, member_coords = (x_loc, y_loc), aggressive=aggressive)

    # Create the circular mask
    centre = (cutout.shape[1] // 2, cutout.shape[0] // 2)
    Y, X = np.ogrid[:cutout.shape[0], :cutout.shape[1]]
    dist_from_centre = np.sqrt((X-centre[0])**2 + (Y-centre[1])**2)
    circ_mask = dist_from_centre <= 112

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
    mask = sb_img >= seg_threshold + 10 * np.log10(1 + z) + measurement_helpers.k_corr(z)

    # Convert the SB image back to counts
    counts_img = measurement_helpers.sb2counts(sb_img) 

    nans = np.isnan(counts_img)
    not_nans = ~nans

    # Display the final image
    masked_img = counts_img * ~non_member_mask * circ_mask * not_nans

    plt.subplot(143)
    plt.imshow(measurement_helpers.stretch(masked_img), cmap=cmap)

    plt.subplot(144)
    plt.imshow(measurement_helpers.stretch(masked_img) * mask, cmap=cmap)

    icl = np.nansum(masked_img * mask)
    total = np.nansum(masked_img)

    icl_values = (masked_img * mask)[(masked_img * mask) != 0] # get rid of masked values
    icl_values = icl_values[~np.isnan(icl_values)] # get rid of nans
    icl_size = float(np.size(icl_values))
    
    total_values = masked_img[masked_img != 0]
    total_values = total_values[~np.isnan(total_values)]
    total_size = float(np.size(total_values))
    
    rms = background_rms(cutout, bad_mask)
    
    icl_err = rms*np.sqrt(icl_size)
    total_err = rms*np.sqrt(total_size)

    print(f'Result = [{icl:.5f}, {total:.5f}, {icl / total:.5f}]\nError = [{icl_err:.5f}, {total_err:.5f}, {(np.sqrt(np.square(icl_err / icl) + np.square(total_err / total))):.5f}]')
    plt.show()

    return icl, total, icl / total

def run_one():
    """
    Manually run one cluster image
    """
    idx = 51
    constant = True

    cutout = np.array(cutouts[str(idx)]['HDU0']['DATA'])
    mask = (np.array(cutouts[str(idx)]['HDU1']['DATA']).astype(int) & (BAD | BRIGHT_OBJECT | NO_DATA)).astype(bool)
    header = cutouts[str(idx)]['HDU0']['HEADER'][()]

    # cutout = cutout * ~mask
    original_shape = cutout.shape
    cutout = skimage.transform.resize(cutout, (224,224))
    mask = skimage.transform.resize(mask, (224,224))

    # Manual star mask if needed
    mask_centres = [(70, 62), (200, 150)]
    mask_sizes = [30,30]
    manual_star_mask = np.zeros_like(mask, dtype=bool)
    for i, mask_centre in enumerate(mask_centres):
        Y, X = np.ogrid[:224, :224]
        dist_from_centre = np.sqrt((X-mask_centre[1])**2 + (Y-mask_centre[0])**2)
        manual_star_mask = manual_star_mask | (dist_from_centre <= mask_sizes[i]) 

    # alpha = np.arctan(200/54)
    # beta = np.pi / 2 - alpha
    # thetas = np.arctan(Y / (X - 170))
    # phis = np.arctan((224 - X) / (200 - Y))
    # manual_star_mask = manual_star_mask | ((thetas < alpha) & (thetas > 0) & (phis < beta) & (phis > 0))
    mask = mask | manual_star_mask

    calc_icl_frac(cutout, mask, tbl['z_cl'][idx], original_shape=original_shape, tbl=tbl, header=header, aggressive=False, constant=constant)

    # Show the pre and post-subtracted radial profiles
    spacing = 10

    radii = np.expand_dims(np.arange(-1, 112, spacing), (1, 2))
    distances_px = np.squeeze(radii[1:]) - spacing / 2
    px2kpc = 300 / 112
    distances_kpc = distances_px * px2kpc

    means, stds = radial_profile(cutout, mask)
    means_bkg, stds_bkg = radial_profile(cutout - background_estimate_2d(cutout, mask, constant=constant), mask)

    high_dists = distances_kpc[distances_kpc > 80]
    plt.figure(figsize=(6,2))
    plt.plot(high_dists, means[distances_kpc > 80], c='b', label='1')
    plt.fill_between(high_dists, (means+stds)[distances_kpc > 80], (means-stds)[distances_kpc > 80], alpha=0.2, color='b')

    plt.plot(high_dists, means_bkg[distances_kpc > 80], c='r', label='2')
    plt.fill_between(high_dists, (means_bkg+stds_bkg)[distances_kpc > 80], (means_bkg-stds_bkg)[distances_kpc > 80], alpha=0.2, color='r')
    plt.ylabel('Average flux (counts)')
    plt.xlabel('Distance (kpc)')

    plt.axhline(0, c='gray', linestyle='--')

    plt.legend(['Before subtraction', '_', 'After subtraction'])
    plt.show()


if __name__ == '__main__':
    # Read table and cutouts file
    tbl = ascii.read('/srv/scratch/z5214005/camira_final.tbl')
    cutouts = h5py.File('/srv/scratch/z5214005/cutouts_300/cutouts_300.hdf')
    # First pass run through all images with default parameters
    for idx in range(125):
        print(f'======================={idx}=======================')
        cutout = np.array(cutouts[str(idx)]['HDU0']['DATA'])
        mask = (np.array(cutouts[str(idx)]['HDU1']['DATA']).astype(int) & (BAD | BRIGHT_OBJECT | NO_DATA)).astype(bool)

        original_shape = cutout.shape
        cutout = skimage.transform.resize(cutout, (224,224))
        mask = skimage.transform.resize(cutout, (224,224))

        calc_icl_frac(cutout, mask, tbl['z_cl'][idx], original_shape, tbl, aggressive=False)

        # Show the pre and post-subtracted radial profiles
        spacing = 10

        radii = np.expand_dims(np.arange(-1, 112, spacing), (1,2))
        distances_px = np.squeeze(radii[1:]) - spacing / 2
        px2kpc = 300 / 112
        distances_kpc = distances_px * px2kpc

        means, stds = radial_profile(cutout, mask)
        means_bkg, std_bkg = radial_profile(cutout - background_estimate_2d(cutout, mask, constant=True), mask)

        high_dists = distances_kpc[distances_kpc > 80]
        plt.figure(figsize=(6,2))
        plt.plot(high_dists, means[distances_kpc > 80], c='b', label='1')
        plt.fill_between(high_dists, (means+stds)[distances_kpc > 80], (means-stds)[distances_kpc > 80], alpha=0.2, color='b')

        plt.plot(high_dists, means_bkg[distances_kpc > 80], c='r', label='2')
        plt.fill_between(high_dists, (means_bkg+stds_bkg)[distances_kpc > 80], (means_bkg-stds_bkg)[distances_kpc > 80], alpha=0.2, color='r')
        plt.ylabel('Average flux (counts)')
        plt.xlabel('Distance (kpc)')

        plt.axhline(0, c='gray', linestyle='--')

        plt.legend(['Before subtraction', '_', 'After subtraction'])
        plt.show()

    
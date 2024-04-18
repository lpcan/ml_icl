from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import ascii
from astropy.modeling.functional_models import Sersic2D
from astropy.table import join
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage.morphology import binary_opening
import skimage

from photutils.segmentation import detect_sources
from astropy.stats import sigma_clipped_stats, mad_std

# Convenience function for plotting the images
stddev = 0.017359
def stretch(img):
    return np.arcsinh(np.clip(img, a_min=0.0, a_max=10.0) / stddev)

def k_corr(z):
    # Equation for LRGs from Chilingarian+2010
    return 0.710579*z + 10.1949*z**2 - 57.0378*z**3 + 133.141*z**4 - 99.9271*z**5

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

def create_profile(cutout, bright_parts, labels, label, threshold, z, central=False):
    # Find centre of blob
    blob = bright_parts * (labels == label)
    coords = np.where(blob == 1)
    centre = (int(np.sum(coords[0]) / coords[0].shape), int(np.sum(coords[1]) / coords[1].shape))

    # Figure out what r_eff of the profile should be
    opened_blob = binary_opening(blob, np.ones((2,2)))

    if np.sum(opened_blob) == 0:
        return np.zeros_lik(cutout), (0,0,0,0)
    
    edges = scipy.spatial.ConvexHull(np.argwhere(opened_blob)) # Convex hull of central blob

    distances = scipy.spatial.distance.cdist([centre], np.argwhere(opened_blob)[edges.vertices])[0] # Distances to edges of shape

    if central:
        quantile = 1
    else:
        quantile = 0.25
    size_frac = 10

    # Try with these parameters, but for very irregularly shaped galaxies, r_eff
    # may need to be more conservative
    while size_frac > 8 and quantile >= 0:
        r_eff = np.quantile(distances, quantile)
        r_eff = np.random.uniform(0.25*r_eff, r_eff)

        # Generate some random parameters for the profile
        amplitude = threshold
        n = 1 # exponential profile
        ellip = np.random.uniform(low=0, high=0.5)
        theta = np.random.uniform(low=0, high=2*np.pi)

        # Generate the model
        model = Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=centre[1], y_0=centre[0], ellip=ellip, theta=theta)
        x,y = np.meshgrid(np.arange(cutout.shape[1]), np.arange(cutout.shape[0]))

        icl_img = np.clip(model(x,y), a_min=None, a_max=threshold)
        icl_img = np.where(bright_parts, 0, icl_img)

        sb_limit = 30.2 + 10 * np.log10(1+z) # Calculate the sb limit
        limit = 10**(-0.4*(sb_limit - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts

        size_frac = (np.sum(icl_img > limit) / np.sum(blob))
        quantile -= 0.25
        if not central:
            break

    return icl_img, (amplitude, ellip, theta, r_eff)

def inject_icl(cutout_id, cutouts, z, seg_threshold=25):
    
    cutout = cutouts[str(cutout_id)]['HDU0']['DATA']

    sb_threshold = seg_threshold + 10 * np.log10(1+z) + k_corr(z) # Calculate mask threshold at this z
    threshold = 10**(-0.4*(sb_threshold - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts

    # Some convoluted stuff so the central blob is not over or undersegmented
    og_labels = detect_sources(cutout, threshold, npixels=20).data
    eroded = skimage.morphology.binary_erosion((og_labels > 0), np.ones((3,3)))
    markers, _ = scipy.ndimage.label(eroded)
    distance = scipy.ndimage.distance_transform_edt(og_labels > 0)
    watershed_labels = skimage.segmentation.watershed(-distance, markers, mask=(og_labels > 0))
    # Need to renumber watershed labels to prevent two different objects with same label
    labels = np.where(watershed_labels > 0, watershed_labels + np.max(og_labels), og_labels)

    # Figure out what to use as r_eff
    bright_parts = (cutout > threshold)

    centre = (cutout.shape[0] // 2, cutout.shape[1] // 2)
    central_blob = bright_parts * (labels == labels[centre[0], centre[1]])
    if np.sum(central_blob) == 0:
        return None

    lsb_img = np.zeros_like(cutout)
    for label in np.unique(watershed_labels+np.max(og_labels)):
        if label == 0 or (np.sum(labels == label) < 250 and label != labels[centre[0], centre[1]]):
            continue
        if label == labels[centre[0], centre[1]]:
            icl_img, (amplitude, ellip, theta, r_eff) = create_profile(cutout, bright_parts, labels, label, threshold, z, central=True)
        else:
            icl_img, _ = create_profile(cutout, bright_parts, labels, label, threshold, z)
        
        lsb_img += icl_img
    
    lsb_img = np.clip(lsb_img, a_min=None, a_max=threshold)

    # Expand the masks of non central galaxies
    non_central_galaxies = bright_parts * ~central_blob 
    kernel = Gaussian2DKernel(7) # Large ish kernel
    non_central_blurred = binary_opening(non_central_galaxies) # Erase tiny bright parts (get rid of specks in ICL area)
    non_central_blurred = convolve(non_central_blurred, kernel) # Expand masks
    # Put just the blurred edges into the original mask
    non_central_blurred = np.where(non_central_galaxies | central_blob, non_central_galaxies, non_central_blurred)

    final_bright_parts = non_central_galaxies + central_blob

    # Calculate the new artificial ICL fraction
    sb_limit = 30.2 + 10 * np.log10(1+z) # Calculate the sb limit
    limit = 10**(-0.4*(sb_limit - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts
    icl = np.sum(lsb_img[lsb_img > limit])

    _, med, _ = sigma_clipped_stats(cutout)
    total = np.sum(((np.array(cutout) - med) * central_blob) + lsb_img)
    
    # Generate the final image
    # Add some noise to the ICL 
    std = 0.006
    noise = np.random.normal(loc=0, scale=std, size=cutout.shape)
    noisy_icl = lsb_img + noise

    # Add to the bright parts
    img = (cutout * final_bright_parts) + noisy_icl

    img = img.astype('<f4')

    result = {}
    result['img'] = img
    result['icl'] = icl
    result['total'] = total
    result['amplitude'] = amplitude
    result['ellip'] = ellip
    result['theta'] = theta
    result['r_eff'] = r_eff

    return result

if __name__ == '__main__':
    cutouts = h5py.File('/srv/scratch/z5214005/lrg_cutouts_resized.hdf')
    tbl = ascii.read('/srv/scratch/z5214005/lrgs_sampled.tbl')

    # # Load the cluster member catalogue
    # members = ascii.read('data/raw/camira_s20a_wide_member.dat', 
    #                     names=['RA_cl', 'Dec_cl', 'Richness', 'z_cl', 'RA', 'Dec', 'M', 'w'])

    # # Match this catalogue to the cluster catalogue
    # merged = join(members, tbl, keys_left=['RA_cl', 'Dec_cl'], keys_right=['RA [deg]', 'Dec [deg]'])
    # merged = merged['ID', 'Name', 'RA_cl', 'Dec_cl', 'z_cl_1', 'RA', 'Dec']

    generated_data = h5py.File('/srv/scratch/mltidal/generated_data_injectotherlsb.hdf', 'w')
    # fracs = []
    # finder = SourceFinder(npixels=20, progress_bar=False, nlevels=8)

    for num in range(len(tbl)):
        print(f'{num}', end='\r')
        cutout_id = tbl['new_ids'][num]
        z = tbl['z'][tbl['new_ids'] == cutout_id]

        result = inject_icl(cutout_id, cutouts, z, seg_threshold=26)

        if result is None:
            continue

        # Add to file
        generated_data[f'{cutout_id}/HDU0/DATA'] = result['img']
        generated_data[f'{cutout_id}/FRAC'] = result['icl'] / result['total']
        generated_data[f'{cutout_id}/ICL'] = result['icl']
        generated_data[f'{cutout_id}/TOTAL'] = result['total']
        generated_data[f'{cutout_id}/PARAMS/AMP'] = result['amplitude']
        generated_data[f'{cutout_id}/PARAMS/ELLIP'] = result['ellip']
        generated_data[f'{cutout_id}/PARAMS/THETA'] = result['theta']
        generated_data[f'{cutout_id}/PARAMS/R_EFF'] = result['r_eff']

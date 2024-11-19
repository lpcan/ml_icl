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
    # Equation from Chilingarian et al. assuming g-r colour of 0.7
    return 1.111*z - 1.101*z**2 - 75.050*z**3 + 295.888*z**4 - 295.390*z**5

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

def inject_icl(cutout_id, cutouts, z, seg_threshold=25):
    
    cutout = cutouts[str(cutout_id)]['HDU0']['DATA']

    sb_threshold = seg_threshold + 10 * np.log10(1+z) + k_corr(z) # Calculate mask threshold at this z
    threshold = 10**(-0.4*(sb_threshold - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts

    # Some convoluted stuff so the central blob is not over or undersegmented
    labels = detect_sources(cutout, threshold, npixels=20).data
    eroded = skimage.morphology.binary_erosion((labels > 0), np.ones((3,3)))
    markers, _ = scipy.ndimage.label(eroded)
    distance = scipy.ndimage.distance_transform_edt(labels > 0)
    watershed_labels = skimage.segmentation.watershed(-distance, markers, mask=(labels > 0))
    # Need to renumber watershed labels to prevent two different objects with same label
    labels = np.where(watershed_labels > 0, watershed_labels + np.max(labels), labels)

    # Figure out what to use as r_eff
    bright_parts = (cutout > threshold)

    centre = (cutout.shape[0] // 2, cutout.shape[1] // 2)
    central_blob = bright_parts * (labels == labels[centre[0], centre[1]])
    if np.sum(central_blob) == 0:
        return None

    # Update estimation of central blob centroid
    coords = np.where(central_blob == 1)
    centre = (int(np.sum(coords[0]) / coords[0].shape), int(np.sum(coords[1]) / coords[1].shape))
    
    # # Find the cluster members in the image
    # x_locs, y_locs = get_member_locs(num, merged, cutout.shape)
    # c_members = labels[y_locs.astype(int), x_locs.astype(int)]
    # member_mask = np.isin(labels, c_members) & labels.astype(bool)
    member_mask = central_blob 

    # Expand the masks of non central galaxies
    non_central_galaxies = bright_parts * ~central_blob 
    kernel = Gaussian2DKernel(7) # Large ish kernel
    non_central_blurred = binary_opening(non_central_galaxies) # Erase tiny bright parts (get rid of specks in ICL area)
    non_central_blurred = convolve(non_central_blurred, kernel) # Expand masks
    # Put just the blurred edges into the original mask
    non_central_blurred = np.where(non_central_galaxies | central_blob, non_central_galaxies, non_central_blurred)

    final_bright_parts = non_central_galaxies + central_blob

    # Figure out what r_eff of the profile should be
    opened_blob = binary_opening(central_blob, np.ones((2,2)))
    # labelled_blob = skimage.measure.label(opened_blob)
    # opened_blob = opened_blob * (labelled_blob == labelled_blob[centre[0], centre[1]])

    edges = scipy.spatial.ConvexHull(np.argwhere(opened_blob)) # Convex hull of central blob

    distances = scipy.spatial.distance.cdist([centre], np.argwhere(opened_blob)[edges.vertices])[0] # Distances to edges of shape
    
    quantile = 1
    size_frac = 10
    
    # Try with these parameters, but for very irregularly shaped galaxies, r_eff
    # may need to be more conservative
    while size_frac > 8 and quantile >= 0:
        r_eff = np.quantile(distances, quantile)

        # Generate some random parameters for the profile
        amplitude = threshold
        n = 1 # exponential profile
        # n = np.random.choice([1,2,3,4,5,6,7,8])
        ellip = np.random.uniform(low=0, high=0.5)
        theta = np.random.uniform(low=0, high=2*np.pi)

        # Generate the model
        model = Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=centre[1], y_0=centre[0], ellip=ellip, theta=theta)
        x,y = np.meshgrid(np.arange(cutout.shape[1]), np.arange(cutout.shape[0]))

        icl_img = np.clip(model(x,y), a_min=None, a_max=threshold)
        # icl_img = np.where(central_blob, 0, icl_img)
        icl_img = np.where(final_bright_parts, 0, icl_img)

        # Calculate the new artificial ICL fraction
        sb_limit = 30.2 + 10 * np.log10(1+z) # Calculate the sb limit
        limit = 10**(-0.4*(sb_limit - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts
        icl = np.sum(icl_img[icl_img > limit])

        _, med, _ = sigma_clipped_stats(cutout)
        total = np.sum(((np.array(cutout) - med) * central_blob) + icl_img)
        
        size_frac = (np.sum(icl_img > limit) / np.sum(central_blob))
        quantile -= 0.25
    
    # Generate the final image
    # Add some noise to the ICL 
    std = mad_std(cutout * ~bright_parts) # Standard deviation of the background (+icl)
    noise = np.random.normal(loc=0, scale=std, size=cutout.shape)
    sb_limit = 29 + 10 * np.log10(1+z) # Calculate the sb limit
    limit = 10**(-0.4*(sb_limit - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts
    icl_img_norm = icl_img / threshold
    # noise = noise * icl_img_norm
    noisy_icl = icl_img + noise

    # Add the ICL image to the non bright parts
    # bkg = cutout * ~final_bright_parts
    # icl_img_norm = icl_img / threshold # this version goes between 0 and 1
    # bkg = bkg * (1 - icl_img_norm) # fade out when the ICL is there. Might need this to be a mask instead
    # icl_img = bkg + noisy_icl
    icl_img = noisy_icl
    # Add to the bright parts
    img = (cutout * final_bright_parts) + icl_img

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
    cutouts = h5py.File('/srv/scratch/z5214005/lrg_cutouts_300kpc_resized.hdf')
    tbl = ascii.read('/srv/scratch/z5214005/lrgs_sampled_1405.tbl')

    # # Load the cluster member catalogue
    # members = ascii.read('data/raw/camira_s20a_wide_member.dat', 
    #                     names=['RA_cl', 'Dec_cl', 'Richness', 'z_cl', 'RA', 'Dec', 'M', 'w'])

    # # Match this catalogue to the cluster catalogue
    # merged = join(members, tbl, keys_left=['RA_cl', 'Dec_cl'], keys_right=['RA [deg]', 'Dec [deg]'])
    # merged = merged['ID', 'Name', 'RA_cl', 'Dec_cl', 'z_cl_1', 'RA', 'Dec']

    generated_data = h5py.File('/srv/scratch/z5214005/generated_data_gaussianbkg.hdf', 'w')
    # fracs = []
    # finder = SourceFinder(npixels=20, progress_bar=False, nlevels=8)

    for num in range(len(tbl)):
        print(f'{num}', end='\r')
        cutout_id = tbl['new_ids'][num]
        z = tbl['z'][tbl['new_ids'] == cutout_id]

        result = inject_icl(cutout_id, cutouts, z, seg_threshold=25)

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

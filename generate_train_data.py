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

from photutils.segmentation import SourceFinder
from astropy.stats import sigma_clipped_stats, mad_std

# Convenience function for plotting the images
stddev = 0.017359
def stretch(img):
    return np.arcsinh(np.clip(img, a_min=0.0, a_max=10.0) / stddev)

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

cutouts = h5py.File('/srv/scratch/z5214005/lrg_cutouts_resized.hdf')
tbl = ascii.read('/srv/scratch/z5214005/lrgs_sampled.tbl')
zs = tbl['z']

# # Load the cluster member catalogue
# members = ascii.read('data/raw/camira_s20a_wide_member.dat', 
#                     names=['RA_cl', 'Dec_cl', 'Richness', 'z_cl', 'RA', 'Dec', 'M', 'w'])

# # Match this catalogue to the cluster catalogue
# merged = join(members, tbl, keys_left=['RA_cl', 'Dec_cl'], keys_right=['RA [deg]', 'Dec [deg]'])
# merged = merged['ID', 'Name', 'RA_cl', 'Dec_cl', 'z_cl_1', 'RA', 'Dec']

generated_data = h5py.File('/srv/scratch/mltidal/generated_data_experiment.hdf', 'w')
fracs = []
finder = SourceFinder(npixels=20, progress_bar=False, nlevels=8)

for num in range(len(tbl)):
    print(f'{num}', end='\r')
    cutout_id = tbl['new_ids'][num]
    cutout = cutouts[cutout_id]['HDU0']['DATA']

    z = zs[num]
    sb_threshold = 26 + 10 * np.log10(1+z) # Calculate mask threshold at this z
    threshold = 10**(-0.4*(sb_threshold - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts

    # Figure out what to use as r_eff
    bright_parts = (cutout > threshold)
    labels = finder(cutout, threshold).data
    # labels, _ = scipy.ndimage.label(bright_parts) # Label each bright section
    centre = (cutout.shape[0] // 2, cutout.shape[1] // 2)
    central_blob = bright_parts * (labels == labels[centre[0], centre[1]])
    if np.sum(central_blob) == 0:
        continue

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

    final_bright_parts = non_central_blurred + central_blob

    # Figure out what r_eff of the profile should be
    opened_blob = binary_opening(central_blob, np.ones((2,2)))
    labelled_blob = skimage.measure.label(opened_blob)
    opened_blob = opened_blob * (labelled_blob == labelled_blob[centre[0], centre[1]])

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
        ellip = np.random.uniform(low=0, high=0.5)
        theta = np.random.uniform(low=0, high=2*np.pi)

        # Generate the model
        model = Sersic2D(amplitude=amplitude, r_eff=r_eff, n=n, x_0=centre[1], y_0=centre[0], ellip=ellip, theta=theta)
        x,y = np.meshgrid(np.arange(cutout.shape[1]), np.arange(cutout.shape[0]))

        icl_img = np.clip(model(x,y), a_min=None, a_max=threshold)
        icl_img = np.where(central_blob, 0, icl_img)

        # Calculate the new artificial ICL fraction
        sb_limit = 28 + 10 * np.log10(1+z) # Calculate the sb limit
        limit = 10**(-0.4*(sb_limit - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts
        icl = np.sum(icl_img[icl_img > limit])

        _, med, _ = sigma_clipped_stats(cutout)
        total = np.sum(((np.array(cutout) - med) * central_blob) + icl_img)
        
        size_frac = (np.sum(icl_img > limit) / np.sum(central_blob))
        quantile -= 0.25
    
    # Generate the final image
    # Add to the bright parts
    img_no_noise = (cutout * final_bright_parts) + icl_img

    # Generate some noise
    std = mad_std(cutout * ~bright_parts) # Standard deviation of the background (+icl)
    noise = np.random.normal(loc=0, scale=std, size=cutout.shape)

    # Add to final image
    img = img_no_noise + noise
    img = img.astype('<f4')

    # Add to file
    generated_data[f'{cutout_id}/HDU0/DATA'] = img
    generated_data[f'{cutout_id}/FRAC'] = icl / total
    generated_data[f'{cutout_id}/ICL'] = icl
    generated_data[f'{cutout_id}/TOTAL'] = total
    generated_data[f'{cutout_id}/PARAMS/AMP'] = amplitude
    generated_data[f'{cutout_id}/PARAMS/ELLIP'] = ellip
    generated_data[f'{cutout_id}/PARAMS/THETA'] = theta
    generated_data[f'{cutout_id}/PARAMS/R_EFF'] = r_eff
    
    # plt.figure(figsize=(16,4))
    # plt.subplot(141)
    # plt.imshow(stretch(cutout))
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Original image')
    # plt.subplot(142)
    # plt.imshow(stretch((cutout * bright_parts)))
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Thresholded image')
    # plt.subplot(143)
    # plt.imshow(stretch(img_no_noise))
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Added ICL profile')
    # plt.subplot(144)
    # plt.imshow(stretch(img))
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Final image')
    # # plt.suptitle(str(icl / total))
    # plt.savefig('asdf.png')
    
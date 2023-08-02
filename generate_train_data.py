from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import ascii
from astropy.modeling.functional_models import Sersic2D
from astropy.table import join
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage.morphology import binary_opening

from photutils.segmentation import SourceFinder
from astropy.stats import sigma_clipped_stats

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

cutouts = h5py.File('lrg_cutouts_resized.hdf')
tbl = ascii.read('data/processed/lrgs_sampled.tbl')
zs = tbl['z']

# # Load the cluster member catalogue
# members = ascii.read('data/raw/camira_s20a_wide_member.dat', 
#                     names=['RA_cl', 'Dec_cl', 'Richness', 'z_cl', 'RA', 'Dec', 'M', 'w'])

# # Match this catalogue to the cluster catalogue
# merged = join(members, tbl, keys_left=['RA_cl', 'Dec_cl'], keys_right=['RA [deg]', 'Dec [deg]'])
# merged = merged['ID', 'Name', 'RA_cl', 'Dec_cl', 'z_cl_1', 'RA', 'Dec']

generated_data = h5py.File('generated_data.hdf', 'w')
fracs = []
finder = SourceFinder(npixels=20, progress_bar=False)

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
    edges = scipy.spatial.ConvexHull(np.argwhere(central_blob)) # Convex hull of central blob

    distances = scipy.spatial.distance.cdist([centre], np.argwhere(central_blob)[edges.vertices])[0] # Distances to edges of shape
    r_eff = np.random.choice(distances) # Vague estimate of size of central blob

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

    # Add to the bright parts
    img_no_noise = (cutout * final_bright_parts) + icl_img #np.clip(model(x,y), a_min=None, a_max=threshold)

    # Generate some noise
    std = np.std(cutout * ~bright_parts) # Standard deviation of the background (+icl)
    noise = np.random.normal(loc=0, scale=std, size=cutout.shape)

    # Add to final image
    # img = img_no_noise + noise
    img = cutout + icl_img
    img = img.astype('<f4')

    # Calculate the new artificial ICL fraction
    sb_limit = 28 + 10 * np.log10(1+z) # Calculate the sb limit
    limit = 10**(-0.4*(sb_limit - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts
    icl = np.sum(icl_img[icl_img > limit])

    # total = np.sum(img_no_noise) # Total brightness (without considering noise)
    _, med, _ = sigma_clipped_stats(cutout)
    total = np.sum(((np.array(cutout) - med) * member_mask) + icl_img)

    # Add to file
    generated_data[f'{cutout_id}/HDU0/DATA'] = img
    generated_data[f'{cutout_id}/FRAC'] = icl / total
    generated_data[f'{cutout_id}/ICL'] = icl
    generated_data[f'{cutout_id}/TOTAL'] = total
    
    # plt.figure(figsize=(8,4))
    # plt.subplot(131)
    # plt.imshow(stretch(icl_img))
    # plt.subplot(132)
    # plt.imshow(stretch((cutout * member_mask) + icl_img))
    # plt.subplot(133)
    # plt.imshow(stretch(img))
    # plt.suptitle(str(icl / total))
    # plt.show()

    fracs.append(icl / total)

plt.hist(fracs, 15)
# plt.xlim(0,0.3)
plt.xlabel('ICL fraction')
plt.ylabel('Counts')
plt.show()
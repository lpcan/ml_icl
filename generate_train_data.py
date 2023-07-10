from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import ascii
from astropy.modeling.functional_models import Sersic2D
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from skimage.morphology import binary_opening

# Convenience function for plotting the images
stddev = 0.017359
def stretch(img):
    return np.arcsinh(np.clip(img, a_min=0.0, a_max=10.0) / stddev)

cutouts = h5py.File('/srv/scratch/z5214005/hsc_icl/cutouts.hdf')
zs = ascii.read('/srv/scratch/z5214005/camira_final.tbl')['z_cl']

generated_data = h5py.File('/srv/scratch/z5214005/generated_data_400.hdf', 'w')
fracs = []

for num in range(126,526):
    print(f'{num}', end='\r')
    cutout = cutouts[str(num)]['HDU0']['DATA']

    z = zs[num]
    sb_threshold = 25 + 10 * np.log10(1+z) # Calculate mask threshold at this z
    threshold = 10**(-0.4*(sb_threshold - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts

    # Figure out what to use as r_eff
    bright_parts = (cutout > threshold)
    labels, _ = scipy.ndimage.label(bright_parts) # Label each bright section
    centre = (cutout.shape[0] // 2, cutout.shape[1] // 2)
    central_blob = bright_parts * (labels == labels[centre[0], centre[1]])
    if np.sum(central_blob) == 0:
        continue
    edges = scipy.spatial.ConvexHull(np.argwhere(central_blob)) # Convex hull of central blob

    distances = scipy.spatial.distance.cdist([centre], np.argwhere(central_blob)[edges.vertices])[0] # Distances to edges of shape
    r_eff = np.median(distances) # Vague estimate of size of central blob

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
    n = np.random.randint(low=1, high=11)
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
    img = img_no_noise + noise

    # Calculate the new artificial ICL fraction
    sb_limit = 27 + 10 * np.log10(1+z) # Calculate the sb limit
    limit = 10**(-0.4*(sb_limit - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168))) # Convert to counts
    icl = np.sum(icl_img[icl_img > limit])
    # total = np.sum(img_no_noise) # Total brightness (without considering noise)
    total = np.sum((cutout * central_blob) + icl_img)
    # Add to file
    generated_data[f'{num}/HDU0/DATA'] = img
    generated_data[f'{num}/FRAC'] = icl / total
    generated_data[f'{num}/ICL'] = icl
    generated_data[f'{num}/TOTAL'] = total
    
    # plt.figure(figsize=(8,4))
    # plt.subplot(131)
    # plt.imshow(stretch(icl_img))
    # plt.subplot(132)
    # plt.imshow(stretch(img_no_noise))
    # plt.subplot(133)
    # plt.imshow(stretch(img))
    # plt.suptitle(str(icl / total))
    # plt.show()

    fracs.append(icl / total)

plt.hist(fracs)
# plt.xlim(0,0.3)
plt.xlabel('ICL fraction')
plt.ylabel('Counts')
plt.show()
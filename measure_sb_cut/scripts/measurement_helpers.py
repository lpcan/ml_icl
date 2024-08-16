"""
Helper functions for measuring ICL fraction
"""

from astropy.convolution import convolve, Gaussian2DKernel
import numpy as np
from photutils.background import Background2D
from photutils.segmentation import detect_threshold, detect_sources
from scipy.interpolate import CloughTocher2DInterpolator

def k_corr(z):
    # Equation from Chilingarian et al. assuming g-r colour of 0.7
    return 1.111*z - 1.101*z**2 - 75.050*z**3 + 295.888*z**4 - 295.390*z**5

# Arcsinh stretch the images for plotting
def stretch(cutout):
    stddev = 0.017359
    return np.arcsinh(np.clip(cutout, a_min=0.0, a_max=10.0) / stddev)

def counts2sb(counts, z, kcorr = 0):
    return 2.5 * np.log10(63095734448.0194 / counts) + 5 * np.log10(0.168) - 10 * np.log10(1+z) - kcorr

def sb2counts(sb): # without reaccounting for dimming
    return 10**(-0.4*(sb - 2.5*np.log10(63095734448.0194) - 5.*np.log10(0.168)))

def enlarge_mask(labels, sigma=1):
    # Make the mask larger by convolving
    kernel = Gaussian2DKernel(sigma)
    mask = convolve(labels, kernel).astype(bool)

    return mask

def create_hot_labels(unsharp, bad_mask, background, npixels=7):
    # Detect threshold
    threshold = detect_threshold(unsharp, nsigma=1.2, background=background, mask=bad_mask)
    
    # Detect sources
    segm = detect_sources(unsharp, threshold, npixels=npixels, mask=bad_mask)

    return segm.data

def background_estimate(cutout, z, cosmo, mask=None):
    """
    Old background estimate code
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
    if cosmo is not None:
        px_dist = cosmo.arcsec_per_kpc_proper(z) * 350 * 1/0.168
        size = int(np.ceil(px_dist.value / box_size))
        box = (X > x_cen - size) & (X < x_cen + size) & (Y > y_cen - size) & (Y < y_cen + size)

    if cosmo is None or np.count_nonzero(box == False) == 0:
        # Image is completely covered, just use the edges
        box = (X < mesh.shape[1] - 1) & (X > 0) & (Y < mesh.shape[0] - 1) & (Y > 0)

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
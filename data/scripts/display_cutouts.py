"""
Display the cutouts in an interactive matplotlib window. Can optionally provide directory to get cutouts from 
as a command line argument, otherwise defaults to "../raw/cutouts/".
"""

import glob
from astropy.io import fits
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
import sys
import re

# Functions to enable sorting in "natural order" (i.e. same order as is in the catalogue). Code from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

# Arcsinh stretch the images for plotting
def stretch(cutout):
    stddev = 0.017359
    return np.arcsinh(np.clip(cutout, a_min=0.0, a_max=10.0) / stddev)

# Class to enable navigating between images
class CutoutCollection:
    def __init__(self, ax, path, ind=0, fits=True):
        self.fits = fits
        self.ind = ind
        if fits:
            self.files = glob.glob(f'{path}*.fits')
            names = [f.split('/')[-1] for f in self.files]
            self.files = [f for _,f in sorted(zip(names, self.files), key= lambda pair: natural_keys(pair[0]))] # Natural sort to put in same order as catalogue
        else:
            self.file = h5py.File(path)
            self.files = list(self.file.keys())

        self.ax = ax
        self.plot()

    # Go to next image
    def next(self, event):
        if self.ind < len(self.files) - 1:
            self.ind += 1
            self.plot()
    
    # Go to previous image
    def prev(self, event):
        self.ind -= 1
        self.plot()

    # Redraw plot
    def plot(self):
        if self.fits:
            file = fits.open(self.files[self.ind])
            image = file[1].data
        else:
            image = np.array(self.file[self.files[self.ind]]['HDU0']['DATA'])
        self.ax.clear()
        self.ax.imshow(stretch(image), cmap='gray_r')
        if self.fits:
            self.ax.set_title(f'Cluster {self.ind}')
        else:
            self.ax.set_title(f'Cluster {self.files[self.ind]}')
        plt.draw()

def main(path, fits=True):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    callback = CutoutCollection(ax, path, fits=fits)
    axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if 'fits' in path:
            is_fits = True
        else:
            is_fits = False
        main(path, fits=is_fits)
    else:
        script_dir = os.path.dirname(__file__) # Want paths relative to the script, not relative to current working directory
        main(os.path.join(script_dir, "../raw/cutouts/"))

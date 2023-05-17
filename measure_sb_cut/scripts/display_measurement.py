from astropy.cosmology import FlatLambdaCDM
from astropy.io import ascii
import h5py
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import os

# Function for displaying the images
def stretch(cutout):
    stddev = 0.017359
    return np.arcsinh(np.clip(cutout, a_min=0.0, a_max=10.0) / stddev)

# Parameters
script_dir = os.path.dirname(__file__)
cutouts_path = os.path.join(script_dir, '../cutouts_550.hdf')
fracs_path = os.path.join(script_dir, '../../fracs.npy')

cosmo = FlatLambdaCDM(H0=68.4, Om0=0.301)
zs = ascii.read(os.path.join(script_dir, '../../data/processed/camira_final.tbl'))['z_cl']

# Load the fractions and the cutouts
fracs = np.load(fracs_path)[2]
cutouts = h5py.File(cutouts_path)

# Rank the clusters according to fracs
ranked = np.argsort(fracs)
mask = ~np.isnan(fracs[ranked]) # Don't include nans
ranked = ranked[mask][::-1]

# Store the images
images = []
for r in ranked:
    img = cutouts[str(r)]['HDU0']['DATA']

    # Cut down to 200 kpc size
    x_cen, y_cen = (img.shape[0] // 2, img.shape[1] // 2)
    size = int((cosmo.arcsec_per_kpc_proper(zs[r]) * 200 * 1/0.168).value)
    img = img[x_cen-size:x_cen+size, y_cen-size:y_cen+size]

    images.append(img)

# Display them all in a grid
fig = plt.figure(figsize=(12,4), dpi=150)
axes = fig.subplots(int(np.ceil(len(ranked) / 20)), 20)
for i in range(len(ranked)):
    # Show the image in a subplot
    axes.flat[i].imshow(stretch(images[i]))
    axes.flat[i].set_title(f'{i}', pad=0., fontdict={'fontsize': 5})
    axes.flat[i].set_xticks([])
    axes.flat[i].set_yticks([])

for i in range(len(ranked), len(axes.flat)):
    fig.delaxes(axes.flat[i])

fig.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=0.2)

gs = GridSpec(1,1)

class View():
    def __init__(self):
        self.all_visible = True
        self.opos = None
        self.otitle = None

    def view(self, evt):
        if evt.inaxes:
            if self.all_visible:
                for ax in fig.axes:
                    if ax != evt.inaxes:
                        ax.set_visible(False)
                self.opos = evt.inaxes.get_position()
                evt.inaxes.set_position(gs[0].get_position(fig))
                self.otitle = int(evt.inaxes.get_title())
                idx = ranked[self.otitle]
                evt.inaxes.set_title(f'{idx}: {fracs[idx]}')
                self.all_visible = False
            else:
                for ax in fig.axes:
                    ax.set_visible(True)
                evt.inaxes.set_position(self.opos)
                evt.inaxes.set_title(self.otitle, pad=0., fontdict={'fontsize': 5})
                self.all_visible = True
                self.opos = None
                self.otitle = None
            fig.canvas.draw_idle()

v = View()
fig.canvas.mpl_connect('button_press_event', v.view)

plt.show()

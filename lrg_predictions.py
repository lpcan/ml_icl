# Compare the model's predictions of ICL on BCGs and non-BCGs
import numpy as np
import h5py
from astropy.io import ascii
from tensorflow import keras
import matplotlib.pyplot as plt
import skimage

from supervised_model_prob import ImageRegressor

# Instantiate the model
model_checkpoint = 'checkpoint-sup-expdatacont2-final.ckpt'
model = ImageRegressor((224,224,1))
negloglik = lambda y, p_y: -p_y.log_prob(y)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=negloglik)
model.load_weights(model_checkpoint).expect_partial()

# Get keys of LRGs in the cutouts files
lrgs_tbl = ascii.read('/srv/scratch/z5214005/lrgs_sampled.tbl')
_, idx = np.unique(lrgs_tbl['old_ids'], return_index=True)
lrgs = lrgs_tbl['new_ids'][idx]

# Get the model's prediction for each of the LRGs
lrg_predictions = []
cutouts = h5py.File('/srv/scratch/z5214005/lrg_cutouts_resized.hdf')

for lrg in lrgs[:2376]:
    print(f'{lrg}\r', end='')
    cutout = np.array(cutouts[lrg]['HDU0']['DATA'])
    img = np.clip(cutout, a_min=0, a_max=10)
    img = np.arcsinh(img / 0.017359)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)

    lrg_predictions.append(model(img).mean().numpy().squeeze())

# Get the model's prediction for BCGs
cutouts = h5py.File('/srv/scratch/z5214005/hsc_icl/cutouts.hdf')
bcg_predictions = []

for bcg in cutouts.keys():
    print(f'{bcg}\r', end='')
    cutout = np.array(cutouts[bcg]['HDU0']['DATA'])
    img = skimage.transform.resize(cutout, (224, 224))
    img = np.clip(img, a_min=0, a_max=10)
    img = np.arcsinh(img / 0.017359)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, -1)

    bcg_predictions.append(model(img).mean().numpy().squeeze())

# Create a histogram showing both of these distributions
plt.subplot(211)
_, bins, _ = plt.hist(lrg_predictions, facecolor='b', alpha=0.5, edgecolor='b')
# plt.yscale('log')
plt.title('ICL predictions for LRGs')
plt.xlabel('ICL fraction')
plt.ylabel('Count')
plt.subplot(212)
plt.hist(bcg_predictions, bins=bins, facecolor='b', alpha=0.5, edgecolor='b')
# plt.yscale('log')
plt.title('ICL predictions for BCGs')
plt.xlabel('ICL fraction')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('asdf.png')
plt.close()

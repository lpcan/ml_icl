"""
Code to automatically tune data augmentation hyperparameters. 
Vaguely following code from https://keras.io/guides/keras_tuner/getting_started/#custom-metric-as-the-objective
"""

from augmentations import augmenter, val_augmenter
from model import NNCLR

import keras_tuner
import numpy as np
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

temperature = 0.1
queue_size = 1000
input_shape = (224, 224, 1)

stddev = 0.017359 # Calculated elsewhere from first 1000 cutouts
shuffle_buffer = 5000 # buffer size for dataset.shuffle
batch_size = 16

def compute_similarity(p, z):
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return tf.reduce_sum((p * z), axis=1)

def get_spearman_coeff(y_true, y_pred):
    return spearmanr(y_pred, y_true).statistic

# Define a HyperModel to do the hyperparameter tuning
class HyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = NNCLR(input_shape=input_shape,
                      temperature=temperature,
                      queue_size=queue_size)
        # Define a new augmenter with tuneable hyperparameters
        self.shuffle_fraction = hp.Float('shuffle_fraction', min_value=0, max_value=1, step=0.1)
        aug = augmenter(input_shape=input_shape,
                        crop_ratio=0.75,
                        crop_jitter_max=0.1,
                        shuffle_fraction=self.shuffle_fraction)
        model.contrastive_augmenter = aug # Replace the augmenter in the model

        # Also tune the optimiser learning rate
        lr = hp.Float('lr', min_value=1e-4, max_value=1e-3, sampling='log')

        model.compile(
            contrastive_optimizer=keras.optimizers.Adam(learning_rate=lr)
        )

        return model
    
    def fit(self, hp, model, data, validation_data, **kwargs):
        model.fit(data, **kwargs)

        # Calculate Spearman's rank coefficient (validation test)
        x_val, fracs = validation_data
        augmenter = val_augmenter(input_shape=input_shape, shuffle_fraction = self.shuffle_fraction)
        for batch in x_val:
            aug_x = augmenter(batch)
        y_pred = model.encoder(aug_x)
        
        # Rank the encodings
        cluster_51 = y_pred[51]
        cluster_51 = np.expand_dims(cluster_51, 0)
        similarities = compute_similarity(y_pred, cluster_51)
        ordered = np.argsort(similarities)[::-1]
        ordered = ordered[~np.isnan(fracs[ordered])]
        ranked = np.argsort(ordered)

        # Rank the fracs
        fracs_ordered = np.argsort(fracs)
        mask = ~np.isnan(fracs[fracs_ordered])
        fracs_ordered = fracs_ordered[mask][::-1]
        fracs_ranked = np.argsort(fracs_ordered)    
    
        coeff = get_spearman_coeff(fracs_ranked, ranked)
        
        return 1 - coeff # want the coefficient to be minimised

tuner = keras_tuner.RandomSearch(
    hypermodel=HyperModel(),
    # No objective, since objective is the return value of HyperModel.fit()
    max_trials=10,
    max_consecutive_failed_trials=1,
    overwrite=True,
    directory='.',
    project_name='hp-tuning'
)

# Preprocessing function
def preprocess(data):
    image = data['image']
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image

# Prepare the dataset
initial_dataset = (tfds.load('hsc_icl', split='train', shuffle_files=True)
           .shuffle(buffer_size=shuffle_buffer)
           .batch(batch_size, drop_remainder=True)
)

# Preprocess the dataset
dataset = initial_dataset.map(preprocess)

# Prepare the validation data
val_init_dataset = tfds.load('hsc_icl', split='train')
dud_only_dataset = val_init_dataset.filter(lambda x: x['id'] < 125)
np_val_data = dud_only_dataset.as_numpy_iterator()
validation_imgs = sorted(np_val_data, key=lambda x: x['id'])
validation_imgs = list(map(preprocess, validation_imgs))
validation_imgs = tf.data.Dataset.from_tensors(validation_imgs)

fracs = np.load('/srv/scratch/z5214005/fracs.npy')
fracs[:,46] = np.nan # Bad image
fracs = fracs[2]

validation_data = (validation_imgs, fracs)

tuner.search(
    data = dataset,
    validation_data = validation_data,
    epochs = 50
)

print(tuner.results_summary())

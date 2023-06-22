from model import NNCLR

import numpy as np
from scipy.stats import spearmanr
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Global parameters
fracs_path = '/srv/scratch/z5214005/fracs.npy'
input_shape = (224,224,1)
temperature = 0.1
queue_size = 1000
stddev = 0.017359

@tf.function
def compute_similarity(p, z):
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return tf.reduce_sum((p * z), axis=1)

def preprocess(data):
    image = data['image']
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image

def calc_spearman(embeddings):
    """
    Given a set of embeddings, calculate the Spearman rank coefficient with 
    respect to precalculated values stored in a numpy array at `fracs_path`.
    Embeddings are assumed to be in the same order in both arrays.
    """
    # Rank the calculated fractions
    fracs = np.load(fracs_path)
    fracs_ordered = np.argsort(fracs[2])
    mask = ~np.isnan(fracs[2][fracs_ordered])
    fracs_ordered = fracs_ordered[mask][::-1]
    fracs_ranked = np.argsort(fracs_ordered)

    # Rank model embeddings with respect to similarity to cluster 51
    cluster_51 = np.expand_dims(embeddings[51], 0)
    similarities = compute_similarity(embeddings, cluster_51)
    ordered = np.argsort(similarities)[::-1]
    ordered = ordered[~np.isnan(fracs[2][ordered])]
    rankings = np.argsort(ordered)

    return spearmanr(rankings, fracs_ranked).statistic

def eval_model(checkpoint_path = 'checkpoint.ckpt'):
    # Load the saved model
    model = NNCLR(input_shape = input_shape, 
                  temperature = temperature,
                  queue_size = queue_size)
    model.compile(contrastive_optimizer=keras.optimizers.Adam())
    model.load_weights(checkpoint_path).expect_partial()

    # Prepare the validation data
    val_init_dataset = tfds.load('hsc_icl', split='train')
    dud_only_dataset = val_init_dataset.filter(lambda x: x['id'] < 125)
    np_val_data = dud_only_dataset.as_numpy_iterator()
    validation_imgs = sorted(np_val_data, key=lambda x: x['id'])
    validation_imgs = list(map(preprocess, validation_imgs))
    validation_imgs = tf.data.Dataset.from_tensors(validation_imgs)

    # Get the embeddings
    from augmentations import val_augmenter
    augmenter = val_augmenter(input_shape=(224,224,1))
    for batch in validation_imgs:
        validation_imgs = augmenter(batch)
    embeddings = model.encoder(validation_imgs)

    # Calculate the Spearman rank coefficient
    coeff = calc_spearman(embeddings)

    return coeff

if __name__ == '__main__':
    print(f'Spearman coefficient for checkpointed model = {eval_model()}')
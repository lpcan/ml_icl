from augmentations import augmenter

import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Hyperparameters
AUTOTUNE = tf.data.AUTOTUNE
shuffle_buffer = 5000 # buffer size for dataset.shuffle

temperature = 0.1
queue_size = 10000
input_shape = (224, 224, 1)

num_epochs = 50
num_images = 2377
batch_size = 16

stddev = 0.017359 # Calculated elsewhere from first 1000 cutouts

# Preprocessing function
def preprocess(data):
    image = data['image']
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image

def test_augmenter():
    # Prepare the dataset
    initial_dataset = (tfds.load('hsc_icl', split='train', shuffle_files=True)
            .shuffle(buffer_size=shuffle_buffer)
            .batch(batch_size)
    )
    dataset = initial_dataset.map(preprocess)

    # Instantiate augmenter
    contrastive_augmenter = augmenter(input_shape)

    # Augment an image 4 separate times
    for batch in dataset.take(1):
        augs = []
        for _ in range(4):
            aug = contrastive_augmenter(batch)[0]
            augs.append(aug)
        img = batch[0]
        plt.figure(figsize=(4,4), dpi=150)
        for i in range(4):
            plt.subplot(2,2,i+1)
            plt.imshow(tf.squeeze(augs[i]))

        plt.show()
        plt.imshow(img)
        plt.show()
        break # Just show one example rather than all in the batch

def mutual_information_exp():
    from finetune import val_preprocess, checkpoint_path
    from model import NNCLR

    import tensorflow_datasets as tfds
    import tensorflow as tf
    import h5py
    import numpy as np
    from sklearn.feature_selection import mutual_info_regression
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Load train images
    ds = tfds.load('finetuning_data', split='train')
    np_ds = ds.as_numpy_iterator()
    np_ds = sorted(np_ds, key=lambda x : x['id'])
    np_ds = np.array(list(map(val_preprocess, np_ds)))
    dataset = tf.data.Dataset.from_tensors(np_ds)

    # Get all the fraction components
    generated_data = h5py.File('/srv/scratch/z5214005/generated_data.hdf')
    fracs = np.zeros((3, 400))
    for key in generated_data.keys():
        fracs[0][int(key) - 126] = generated_data[key]['ICL'][()]
        fracs[1][int(key) - 126] = generated_data[key]['TOTAL'][()]
        fracs[2][int(key) - 126] = generated_data[key]['FRAC'][()]

    # Load the model
    base_model = NNCLR(input_shape=(224,224,1), temperature=0.1, queue_size=1000)
    base_model.load_weights(checkpoint_path).expect_partial()
    encoder = base_model.encoder

    # Predict the values
    X = encoder.predict(dataset)

    # Calculate mutual information
    mi0 = mutual_info_regression(X, fracs[0])
    mi1 = mutual_info_regression(X, fracs[1])

    mi_diff = mi0 - mi1 # This is a number between -1 and 1, where -1 represents strong importance for fracs[1] and 1 is strong important for fracs[0]
    mi1 = -mi1

    matrix = np.row_stack([mi0, mi1, mi_diff])
    plt.figure(figsize=(30,3), dpi=200)
    sns.heatmap(matrix, cmap='seismic', center=0, annot=False)
    plt.savefig('asdf.png')     
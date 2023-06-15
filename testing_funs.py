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

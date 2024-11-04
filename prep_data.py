"""
Prepare datasets for the model to train on
"""
import h5py
import numpy as np
import skimage
import tensorflow as tf
import tensorflow_datasets as tfds

BAD = 1
NO_DATA = (1 << 8)
BRIGHT_OBJECT = (1 << 9)

def preprocess(image, label):
    """
    Preprocess a given image and label (for training set)
    """
    # Clip and arcsinh scale the images
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / 0.017359)
    # Do not allow label = 0 (divide by zero error)
    label = tf.math.maximum(label, 1e-9)
    return image, label

def prepare_training_data():
    """
    Load and split the tensorflow training dataset into train and test set
    """
    dataset, validation_dataset = (tfds.load('supervised_data', split=['train[:90%]', 'train[90%:]'], data_dir='/srv/scratch/mltidal/tensorflow_datasets', as_supervised=True)
    )

    dataset = dataset.batch(50)
    validation_dataset = validation_dataset.batch(100)

    dataset = dataset.map(preprocess)
    validation_dataset = validation_dataset.map(preprocess)

    return dataset, validation_dataset

def prepare_test_data(hdf_path='data/processed/cutouts_300.hdf', save=False):
    """
    Process test data and save as a numpy array
    """
    cutouts = h5py.File(hdf_path)
    images = []

    for idx in range(125):
        # Load the original cutout and mask
        cutout = np.array(cutouts[str(idx)]['HDU0']['DATA'])
        big_mask = (np.array(cutouts[str(idx)]['HDU1']['DATA']).astype(int) & (BAD | NO_DATA | BRIGHT_OBJECT))

        # Resize everything
        mask = skimage.transform.resize(big_mask, (224,224))
        img = skimage.transform.resize(cutout, (224,224))

        # Arcsinh scale images
        img = np.clip(img, a_min=0, a_max=10)
        img = np.arcsinh(img / 0.017359) * ~(mask > 0)

        # Place into array
        img = np.expand_dims(img, -1)
        images.append(img)
    images = np.array(images)

    if save:
        np.save('prepared_images.npy')
    return images

def prepare_run_data(image_path):
    cutouts = h5py.File(image_path)
    images = []

    keys = sorted(cutouts.keys())

    for key in keys:
        cutout = np.array(cutouts[key]['DATA'])
        mask = np.array(cutouts[key]['MASK']).astype(int) & (BAD | NO_DATA | BRIGHT_OBJECT)

        # Resize
        mask = skimage.transform.resize(mask, (224,224))
        img = skimage.transform.resize(cutout, (224,224))

        # Scale the images and apply the mask
        img = np.clip(img, a_min=0, a_max=10)
        img = np.arcsinh(img / 0.017359) * ~(mask > 0)

        # Place into array
        img = np.expand_dims(img, -1)
        images.append(img)

    images = np.array(images) 

    return images

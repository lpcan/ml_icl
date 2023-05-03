"""hsc_icl dataset."""

import glob
import h5py
import numpy as np
import skimage
import tensorflow as tf
import tensorflow_datasets as tfds

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hsc_icl dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(224, 224, 1), dtype=tf.float32),
            'id': tf.int16,
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Data should already be downloaded - run the builder from the same directory as the cutouts file

    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        # These kwargs are passed to _generate_examples()
        gen_kwargs={'path': './cutouts.hdf'}
      ),
    ]

  def _generate_examples(self, path):
    """Yields examples."""
    cutouts = h5py.File(path, 'r')

    # Go through the examples
    for id in cutouts.keys():
      cutout = cutouts[id]

      im = create_img(cutout)

      yield id, {'image': im, 'id': id}

def create_img(cutout, target_size=224):
    """
    Convenience function to take in the fits cutout file and return an image with the correct size
    """
    # Retrieve image data
    img = np.array(cutout['HDU0']['DATA'])

    # Resize image to desired cutout size
    resized = skimage.transform.resize(img, (target_size, target_size))
    
    resized = np.expand_dims(resized, axis=2)

    return resized
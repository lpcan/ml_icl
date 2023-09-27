"""
Define new data augmentations to be used by the model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_cv

class RandomResizedCrop(layers.Layer):
    """
    Randomly crop the images and resize the image back to original size. 
    Aspect ratio will be changed with probability `prob_ratio_change`.
    Crop is offset with a max of `jitter_max`.
    """

    def __init__(self, ratio, prob_ratio_change, jitter_max):
        super().__init__()
        self.ratio = ratio
        self.prob_ratio_change = prob_ratio_change
        self.jitter_max = jitter_max

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]

        # Determine the maximum allowed zoom amount
        mid = [height // 2, width // 2]

        norths = tf.reverse(images[:, 0:mid[0], mid[1], :], axis=[1])
        souths = images[:, mid[0]:, mid[1], :]
        wests = tf.reverse(images[:, mid[0], 0:mid[1], :], axis=[1])
        easts = images[:, mid[0], mid[1]:]

        max_widths = tf.ones(batch_size, dtype=tf.dtypes.int32) * 20 # Impose minimum width of 40x40 box

        for lines in [norths, souths, wests, easts]:
            widths = tf.squeeze(tf.math.argmax(lines < 0.006, axis=1, output_type=tf.dtypes.int32)) # Find half widths of peaks
            widths_fixed = tf.where(widths == 0, height // 2, widths) # If not found, default to normal size of image
            max_widths = tf.math.maximum(max_widths, widths_fixed) # Find the new maximum widths

        max_zoom = tf.cast(max_widths * 2 / height, dtype=tf.dtypes.float32)

        # Draw uniformly random scales up to the maximum zoom amount
        random_scales = tf.random.uniform((batch_size,), minval=max_zoom, maxval=1.)

        # Apply image stretching with probability `prob_ratio_change`
        random_ratios = tf.random.uniform((batch_size,), self.ratio[0], self.ratio[1]) # Draw uniformly distributed aspect ratios
        random_nums = tf.random.uniform((batch_size,), 0.0, 1.0)
        random_ratios = tf.where(random_nums > self.prob_ratio_change, tf.ones((batch_size,)), random_ratios)

        # Define the bounding boxes of the crop
        new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
        new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)

        jitter_y = tf.random.uniform((batch_size,), -self.jitter_max, self.jitter_max)
        jitter_x = tf.random.uniform((batch_size,), -self.jitter_max, self.jitter_max)

        # Don't jitter more than zoom_scale
        jitter_y = tf.math.minimum(random_scales, jitter_y)
        jitter_x = tf.math.minimum(random_scales, jitter_x)

        height_offsets = tf.clip_by_value((1-new_heights)/2 + jitter_y, 0, 1-new_heights)
        width_offsets = tf.clip_by_value((1-new_widths)/2 + jitter_x, 0, 1-new_widths)

        bounding_boxes = tf.stack(
            [
                height_offsets, 
                width_offsets, 
                height_offsets + new_heights,
                width_offsets + new_widths,
            ],
            axis=1,
        )

        # Crop and resize
        images = tf.image.crop_and_resize(
            images, bounding_boxes, tf.range(batch_size), (height, width)
        )

        return images
    
class RandomGaussianNoise(layers.Layer):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def call(self, images):
        # Add some random noise
        images += 3 * self.stddev * tf.random.normal(tf.shape(images))
        return images

def augmenter(input_shape, crop_ratio=3/4, crop_prob=0.5,
              crop_jitter_max=0.1, cutout_height_width=0.1):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Normalization(mean=0.948, variance=1.108**2),
            layers.RandomFlip(mode="horizontal_and_vertical"),
            RandomResizedCrop(ratio=(crop_ratio, 1/crop_ratio), prob_ratio_change=crop_prob, jitter_max=crop_jitter_max),
            RandomGaussianNoise(stddev=0.017359),
            keras_cv.layers.preprocessing.RandomCutout(cutout_height_width, cutout_height_width),
        ]
    )

def val_augmenter(input_shape):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Normalization(mean=0.948, variance=1.108**2),
            layers.RandomFlip(mode='horizontal_and_vertical'),
            RandomGaussianNoise(stddev=0),
        ]
    )
from augmentations import *

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

class AddSersic(layers.Layer):
    def __init__(self):
        super().__init__()

    def compute_sersic(self, x, y, amplitudes, r_effs, ns, x_0s, y_0s, ellips, thetas):
        # Adapted from https://docs.astropy.org/en/stable/_modules/astropy/modeling/functional_models.html#Sersic2D
        bn = tfp.math.igammainv(2.0 * ns, 0.5)[:,tf.newaxis,tf.newaxis]
        a, b = r_effs[:,tf.newaxis,tf.newaxis], ((1 - ellips) * r_effs)[:,tf.newaxis,tf.newaxis]
        cos_theta, sin_theta = tf.math.cos(thetas)[:,tf.newaxis,tf.newaxis], tf.math.sin(thetas)[:,tf.newaxis,tf.newaxis]
        x_maj = (x - x_0s[:,tf.newaxis,tf.newaxis]) * cos_theta + (y - y_0s[:,tf.newaxis,tf.newaxis]) * sin_theta
        x_min = -(x - x_0s[:,tf.newaxis,tf.newaxis]) * sin_theta + (y - y_0s[:,tf.newaxis,tf.newaxis]) * cos_theta
        z = tf.math.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitudes[:,tf.newaxis,tf.newaxis] * tf.math.exp(-bn * (z ** (1 / ns[:,tf.newaxis,tf.newaxis]) - 1))

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

        min_widths = tf.ones(batch_size, dtype=tf.dtypes.int32) * height

        for lines in [norths, souths, wests, easts]:
            widths = tf.squeeze(tf.math.argmax(lines < 0.006, axis=1, output_type=tf.dtypes.int32)) # Find half widths of peaks
            widths_fixed = tf.where(widths == 0, height // 2, widths) # If not found, default to normal size of image
            min_widths = tf.math.minimum(min_widths, widths_fixed) # Find the new maximum widths

        # Create the parameters
        x, y = tf.meshgrid(tf.range(height, dtype=tf.dtypes.float32), tf.range(height, dtype=tf.dtypes.float32))
        x, y = tf.repeat(x[tf.newaxis,:], batch_size, axis=0), tf.repeat(y[tf.newaxis,:], batch_size, axis=0)
        amplitudes = tf.random.uniform([2, batch_size], minval=0.014, maxval=0.048)
        r_effs = tf.cast(min_widths, tf.dtypes.float32)
        ns = tf.ones([batch_size])
        x_0s, y_0s = tf.ones([batch_size]) * 112, tf.ones([batch_size]) * 112
        ellips = tf.random.uniform([2, 16], minval=0, maxval=0.5)
        thetas = tf.random.uniform([2, 16], minval=0, maxval=2*tf.constant(math.pi))

        # Generate random ICL profiles
        profiles_1 = self.compute_sersic(x, y, amplitudes[0], r_effs, ns, x_0s, y_0s, ellips[0], thetas[0])
        profiles_2 = self.compute_sersic(x, y, amplitudes[1], r_effs, ns, x_0s, y_0s, ellips[1], thetas[1])

        # Arcsinh stretch the profiles
        profiles_1 = tf.clip_by_value(profiles_1, 0.0, 10.0)
        profiles_2 = tf.clip_by_value(profiles_2, 0.0, 10.0)
        profiles_1 = tf.math.asinh(profiles_1 / 0.017359)
        profiles_2 = tf.math.asinh(profiles_2 / 0.017359)

        # Shuffle the second set to randomly pair them with set 1
        perm = tf.random.shuffle(tf.range(batch_size))
        profiles_2 = tf.gather(profiles_2, perm)

        # Clip to above a threshold
        profiles_1 = tf.where(profiles_1 < 0.0046, tf.zeros(tf.shape(profiles_1)), profiles_1)
        profiles_2 = tf.where(profiles_2 < 0.0046, tf.zeros(tf.shape(profiles_2)), profiles_2)

        # Calculate numbers (not actual fractions but close enough)
        icl_1 = tf.math.reduce_sum(profiles_1, axis=[1,2])
        icl_2 = tf.math.reduce_sum(profiles_2, axis=[1,2])
        total_1 = tf.math.reduce_sum(images, axis=[1,2,3])
        total_2 = tf.math.reduce_sum(images, axis=[1,2,3])
        fracs_1 = icl_1 / total_1
        fracs_2 = icl_2 / total_2

        # Reshape the profiles tensors
        profiles_1 = tf.expand_dims(profiles_1, axis=-1)
        profiles_2 = tf.expand_dims(profiles_2, axis=-1)

        # Create the final images
        images_1 = images + profiles_1
        images_2 = tf.gather(images, perm) + profiles_2

        # Scale set 2 to match icl fractions of set 1
        corrections = fracs_2 - fracs_1
        images_2 = images_2 + corrections[:,tf.newaxis,tf.newaxis,tf.newaxis]

        # Return the images pairs
        return images_1, images_2
        

def augmenter(input_shape, crop_ratio=3/4, crop_prob=0.5,
              crop_jitter_max=0.1, cutout_height_width=0.1):
    inputs = layers.Input(shape=input_shape)
    x = layers.Normalization(mean=0.948, variance=1.108**2)(inputs)
    x1, x2 = AddSersic()(x)
    x1 = layers.RandomFlip(mode='horizontal_and_vertical')(x1)
    x2 = layers.RandomFlip(mode='horizontal_and_vertical')(x2)
    # x1 = RandomResizedCrop(ratio=(crop_ratio, 1/crop_ratio), prob_ratio_change=crop_prob, jitter_max=crop_jitter_max)(x1)
    # x2 = RandomResizedCrop(ratio=(crop_ratio, 1/crop_ratio), prob_ratio_change=crop_prob, jitter_max=crop_jitter_max)(x2)
    x1 = RandomGaussianNoise(stddev=0.017359)(x1)
    x2 = RandomGaussianNoise(stddev=0.017359)(x2)
    outputs = x1, x2

    return keras.Model(inputs, outputs)

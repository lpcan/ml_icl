"""
Define new data augmentations to be used by the model
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RandomResizedCrop(layers.Layer):
    """
    Randomly crop the images and resize the image back to original size. 
    Aspect ratio will be changed with probability `prob_ratio_change`.
    Crop is offset with a max of `jitter_max`.
    """

    def __init__(self, scale, ratio, prob_ratio_change, jitter_max):
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.prob_ratio_change = prob_ratio_change
        self.jitter_max = jitter_max

    def call(self, images):
        batch_size = tf.shape(images)[0]
        height = tf.shape(images)[1]
        width = tf.shape(images)[2]
        
        # Draw uniformly random scales
        random_scales = tf.random.uniform((batch_size,), self.scale[0], self.scale[1])
        
        # Apply image stretching with probability `prob_ratio_change`
        random_ratios = tf.random.uniform((batch_size,), self.ratio[0], self.ratio[1]) # Draw uniformly distributed aspect ratios
        random_nums = tf.random.uniform((batch_size,), 0.0, 1.0)
        random_ratios = tf.where(random_nums > self.prob_ratio_change, tf.ones((batch_size,)), random_ratios)

        # Define the bounding boxes of the crop
        new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
        new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)

        jitter_y = tf.random.uniform((batch_size,), -self.jitter_max, self.jitter_max)
        jitter_x = tf.random.uniform((batch_size,), -self.jitter_max, self.jitter_max)

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

def augmenter(input_shape):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.RandomFlip(mode="horizontal_and_vertical"),
            RandomResizedCrop(scale=(0.5, 1.0), ratio=(3/4, 4/3), prob_ratio_change=0.5, jitter_max=0.1),
            RandomGaussianNoise(stddev=0.017359),
            # other augmentations? rotation, gaussian smoothing, etc?
        ]
    )
"""
Define new data augmentations to be used by the model
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

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

class RadialShuffle(layers.Layer):
    def __init__(self, input_shape, shuffle_fraction=0.5):
        super().__init__()
        self.shuffle_fraction = shuffle_fraction
        self.height = input_shape[0]
        self.width = input_shape[1]

        # Precomputations
        centre = (input_shape[0] // 2, input_shape[1] // 2)
        Y, X = np.ogrid[:224, :224]
        # Prepare the radial and angle maps
        self.dist_map = tf.convert_to_tensor(np.round(np.sqrt((X-centre[0])**2 + (Y-centre[1])**2)).astype(int))
        ax2_indices = np.zeros(self.dist_map.shape)
        self.num_rows = self.dist_map[0][0] + 1
        dists, counts = np.unique(self.dist_map, return_counts = True)
        for dist in range(self.num_rows):
            ax2_indices[self.dist_map == dist] = np.arange(counts[dist]) # These are the 'angle' indices
        self.num_cols = np.max(counts)
        self.ax2_indices = tf.convert_to_tensor(ax2_indices.astype(int))

    def process_one(self, image):
        # Select `shuffle_fraction` random pixels
        random_px = tf.random.uniform(image.shape) < self.shuffle_fraction
        masked_img = tf.squeeze(tf.where(random_px == True, image, np.nan))

        # Insert values to shuffle into the polar projection
        dists_masked = self.dist_map[~tf.math.is_nan(masked_img)]
        ax2_masked = self.ax2_indices[~tf.math.is_nan(masked_img)]

        pol_ind = tf.stack([dists_masked, ax2_masked], axis=1) # [[x1, y1], [x2, y2]] format
        polar_array = tf.fill((self.num_rows, self.num_cols), value=np.nan)
        polar_array = tf.tensor_scatter_nd_update(polar_array, 
                                                  indices=pol_ind, 
                                                  updates=masked_img[~tf.math.is_nan(masked_img)])
        
        # Shuffle along the second axis. Code from https://stackoverflow.com/a/74870471
        rnd = tf.argsort(tf.random.uniform(polar_array.shape), axis=1)
        rnd = tf.concat([
            tf.repeat(
                tf.range(polar_array.shape[0])[...,tf.newaxis,tf.newaxis], 
                tf.shape(rnd)[1], axis=1), 
            rnd[...,tf.newaxis]], 
            axis=2)
        pol_arr_shuffled = tf.gather_nd(polar_array, rnd, batch_dims=0)

        # Place the not nan values back where we expect to find them
        nan_mask = tf.math.is_nan(pol_arr_shuffled)
        sorted_inds = tf.argsort(dists_masked)
        dists_sorted = tf.gather(dists_masked, sorted_inds)
        ax2_sorted = tf.gather(ax2_masked, sorted_inds)
        by_row_inds = tf.stack([dists_sorted, ax2_sorted], axis=1)
        pol_arr_shuffled = tf.tensor_scatter_nd_update(pol_arr_shuffled, 
                                                       indices=by_row_inds,
                                                       updates=pol_arr_shuffled[~nan_mask])

        # Create the final image
        vals_to_insert = tf.gather_nd(pol_arr_shuffled, pol_ind)
        image = tf.squeeze(image)
        image = tf.tensor_scatter_nd_update(image,
                                            indices=tf.reshape(tf.where(~tf.math.is_nan(masked_img)), [-1, 2]),
                                            updates=vals_to_insert)
        image = tf.expand_dims(image, 2)

        return image

    def call(self, images):
        # TODO: make this sensible (currently horrible)
        return tf.map_fn(self.process_one, images)

class RandomDrop(layers.Layer):
    def __init__(self, drop_percentage=0.2):
        super().__init__()
        self.drop_percentage = drop_percentage

    def call(self, images):
        # Randomly mask out `drop_percentage` of the pixels in the image
        random_px = tf.random.uniform(tf.shape(images))
        masked_imgs = tf.where(random_px < self.drop_percentage, np.nan, images)
        return masked_imgs

def augmenter(input_shape, crop_ratio=3/4, crop_prob=0.5,
              crop_jitter_max=0.1, drop_percentage=0.2):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Normalization(mean=0.948, variance=1.108**2),
            layers.RandomFlip(mode="horizontal_and_vertical"),
            # RadialShuffle(input_shape=input_shape, shuffle_fraction=shuffle_fraction),
            RandomResizedCrop(ratio=(crop_ratio, 1/crop_ratio), prob_ratio_change=crop_prob, jitter_max=crop_jitter_max),
            RandomGaussianNoise(stddev=0.017359),
            RandomDrop(drop_percentage=drop_percentage)
        ]
    )

def val_augmenter(input_shape, shuffle_fraction=0.5):
    return keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Normalization(mean=0.948, variance=1.108**2),
            # RadialShuffle(input_shape=input_shape, shuffle_fraction=shuffle_fraction),
        ]
    )
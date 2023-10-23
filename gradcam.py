# Adapted from https://keras.io/examples/vision/grad_cam/ and https://github.com/tensorflow/tensorflow/issues/44462 

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib
import h5py
import skimage
from astropy.io import ascii

from supervised_model import ImageRegressor
    
def make_gradcam_heatmap(img, model, last_conv_layer_name='conv2d_187'):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer_model = keras.models.Model(model.get_layer('encoder').input, [model.get_layer('encoder').get_layer(last_conv_layer_name).output, model.get_layer('encoder').get_layer('add_59').output])
    
    # Now create another model that maps those activations to the actual output
    # (because Tensorflow doesn't like getting intermediate layer outputs from
    # subclassed models)
    input = keras.Input(shape=last_conv_layer_model.output[1].shape[1:])
    encoder_output = model.get_layer('encoder').layers[-1](input)
    x = model.layers[-2](encoder_output)
    output = model.layers[-1](x)

    grad_model = keras.Model(input, output)

    # Prepare the image
    img_array = model.layers[0](img)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        out1, last_conv_layer_output = last_conv_layer_model(img_array)
        print(out1)

        tape.watch(last_conv_layer_output)

        pred = grad_model(last_conv_layer_output).mean()
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    # Here there is only one output neuron, so we are interested in what
    # features of the image cause this neuron to be closer to 1 (in a multiclass
    grads = tape.gradient(pred, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the output
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualisation purposes, normalise the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

if __name__=='__main__':
    # Load the data
    # cutouts = h5py.File('/srv/scratch/mltidal/generated_data_wparams.hdf')
    cutouts = h5py.File('/srv/scratch/mltidal/vd_cutouts.hdf')

    # Get an example image and prepare it for the network
    tbl = ascii.read('/srv/scratch/z5214005/lrgs_sampled.tbl')
    keys = tbl['new_ids']
    example_num = 39620
    example = np.array(cutouts['45']['HDU0']['DATA'])
    example = skimage.transform.resize(example, (224,224))
    example = np.clip(example, a_min=0, a_max=10)
    example = np.arcsinh(example / 0.017359)
    example = np.expand_dims(example, 0)
    example = np.expand_dims(example, -1)

    # Build the model
    model = ImageRegressor((224,224,1))
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=negloglik)
    model.load_weights('checkpoint-sup-newdatacont.ckpt').expect_partial()

    # Show what the model predicts
    pred = model.predict(example)
    print(f'Predicted: {pred}')

    # Generate the activation heatmap
    heatmap = make_gradcam_heatmap(example, model)

    # Display the heatmap
    # Generate a colourised version of the example image
    img = ((example - np.min(example)) / (np.max(example) - np.min(example))).squeeze()
    img = np.uint8(255*img) # rescale
    viridis = matplotlib.colormaps['gray_r']
    viridis_colours = viridis(np.arange(256))[:,:3] # get rgb values of the colourmap
    img = viridis_colours[img]

    # Generate a colourised version of the heatmap
    heatmap = np.uint8(255*heatmap)
    jet = matplotlib.colormaps.get_cmap('jet')
    jet_colours = jet(np.arange(256))[:,:3]
    jet_heatmap = jet_colours[heatmap]
    jet_heatmap = skimage.transform.resize(jet_heatmap, (img.shape[1], img.shape[0]))

    # Superimpose on the original image
    superimposed_img = jet_heatmap * 0.4 + img * 0.6
    # superimposed_img = keras.utils.array_to_img(superimposed_img)
    plt.imshow(superimposed_img)
    plt.savefig('asdf.png')
    plt.close()

    # superimposed_img.save('asdf.png')

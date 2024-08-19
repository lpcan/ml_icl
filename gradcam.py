# Adapted from https://keras.io/examples/vision/grad_cam/ and https://github.com/tensorflow/tensorflow/issues/44462 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
import pickle
import h5py
import skimage
from astropy.io import ascii

from model import ImageRegressor
from data.scripts.display_cutouts import stretch
from measure_sb_cut.scripts.measure import calc_icl_frac
from augmentations import val_augmenter

MODEL = 'iclnoise-final'

def make_grad_model(model, last_conv_layer_name='conv2d_187'):
    # Get the name of the final convolutional layer in the encoder
    last_conv_layer_name = None
    final_layer_name = None
    for layer in model.get_layer('encoder').layers:
        if 'conv' in layer.name:
            last_conv_layer_name = layer.name
        elif 'add' in layer.name:
            final_layer_name = layer.name

    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer_model = keras.models.Model(model.get_layer('encoder').input, [model.get_layer('encoder').get_layer(last_conv_layer_name).output, model.get_layer('encoder').get_layer(final_layer_name).output])
    
    # Now create another model that maps those activations to the actual output
    # (because Tensorflow doesn't like getting intermediate layer outputs from
    # subclassed models)
    input = keras.Input(shape=last_conv_layer_model.output[1].shape[1:])
    encoder_output = model.get_layer('encoder').layers[-1](input)
    x = encoder_output
    for layer in model.layers[2:]:
        x = layer(x)
    output = x
    # x = model.layers[-2](encoder_output)
    # output = model.layers[-1](x)

    grad_model = keras.Model(input, output)

    return last_conv_layer_model, grad_model

def make_gradcam_heatmap(batch, augmenter, last_conv_layer_model, grad_model):
    # Prepare the image
    img_array = augmenter(batch)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        out1, last_conv_layer_output = last_conv_layer_model(img_array)

        tape.watch(last_conv_layer_output)

        pred = grad_model(last_conv_layer_output).mean()
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    # Here there is only one output neuron, so we are interested in what
    # features of the image cause this neuron to be closer to 1 (in a multiclass
    grads = tape.gradient(pred, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the output
    # then sum all the channels to obtain the heatmap class activation
    heatmaps = tf.matmul(last_conv_layer_output, pooled_grads[:,tf.newaxis,:,tf.newaxis])

    heatmaps = tf.squeeze(heatmaps, -1) # Remove final dimension
    img_array = tf.squeeze(img_array, -1)

    # For visualisation purposes, normalise the heatmap between 0 and 1
    maxes = tf.math.reduce_max(heatmaps, axis=(1, 2))
    heatmaps = tf.maximum(heatmaps, 0) / tf.reshape(maxes, (-1, 1, 1))
    
    return heatmaps.numpy(), img_array.numpy()

if __name__=='__main__':
    # Load the data
    print('Loading images...')
    fracs = np.load('/srv/scratch/mltidal/fracs_manual_photoz.npy')[2]

    tbl = ascii.read('/srv/scratch/z5214005/camira_final.tbl')
    zs = tbl['z_cl']
    cutouts = h5py.File('/srv/scratch/z5214005/cutouts_300/cutouts_300.hdf')

    not_nans = np.where(~np.isnan(fracs))[0]

    dataset = np.load('badmaskimgs_300kpc.npy')
    dataset = dataset[not_nans]

    # Work out what the finetuning splits were
    k = 5
    rng = np.random.default_rng(seed=24)
    idxs = np.arange(len(not_nans))
    rng.shuffle(idxs)
    original_order = np.argsort(idxs)
    splits = np.array_split(idxs, k)

    # Get the original predictions
    with open(f'/srv/scratch/mltidal/finetuning_results/{MODEL}.pkl', 'rb') as fp:
        results, err_l, err_h = pickle.load(fp)
    
    results = np.array([i for row in results for i in row])
    results = results[original_order]

    # Build the model
    model = ImageRegressor((224,224,1))
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=negloglik)
    augmenter = val_augmenter((224,224,1))

    print('Generating gradcam outputs...')

    # Need to generate gradcam outputs for the test sets of the five different models
    gradcams = np.zeros((len(not_nans), 56, 56))
    images = np.zeros((len(not_nans), 224, 224))

    for run in range(k):
        test_set = splits[run]
        test_ds = dataset[test_set]

        model.load_weights(f'/srv/scratch/mltidal/finetuning_results/checkpoints/checkpoint-{MODEL}-split{run}.ckpt').expect_partial()
        last_conv_layer_model, grad_model = make_grad_model(model)
        heatmaps, img_arrays = make_gradcam_heatmap(test_ds, augmenter, last_conv_layer_model, grad_model)

        gradcams[test_set, :, :] = heatmaps
        images[test_set, :, :] = img_arrays
        
    cmap = matplotlib.colormaps['viridis']
    cmap.set_bad(cmap(0))

    print('Creating PDF...')

    # Generate an output PDF with the results
    with PdfPages('outputs.pdf') as pdf:
        plt.figure(figsize=(8,12), dpi=150)

        for i, idx in enumerate(not_nans):
            row = i % 6
            sp_row = row * 4
            plt.subplot(6, 4, sp_row+1)

            # Show the original image
            image = images[i]
            plt.imshow(image)
            plt.title(str(idx))
            plt.xticks([])
            plt.yticks([]) 

            # Show the gradcam output and model prediction
            plt.subplot(6, 4, sp_row+2) 
            image = ((image - np.min(image)) / (np.max(image) - np.min(image))).squeeze()
            image = np.uint8(255*image) # rescale
            gray = matplotlib.colormaps['gray_r']
            gray_colours = gray(np.arange(256))[:,:3] # get rgb values of the colourmap
            image = gray_colours[image]

            # Generate a colourised version of the heatmap
            heatmap = np.uint8(255 * gradcams[i])
            jet = matplotlib.colormaps.get_cmap('jet')
            jet_colours = jet(np.arange(256))[:,:3]
            jet_heatmap = jet_colours[heatmap]
            jet_heatmap = skimage.transform.resize(jet_heatmap, (image.shape[1], image.shape[0]))

            # Superimpose on the original image
            superimposed_img = jet_heatmap * 0.4 + image * 0.6
            plt.imshow(superimposed_img.squeeze())
            plt.title(results[i])
            plt.xticks([])
            plt.yticks([])

            # Show the masked version of the image
            plt.subplot(6, 4, sp_row+3)
            cutout = skimage.transform.resize(cutouts[str(idx)]['HDU0']['DATA'], (224,224))
            bad_mask = (np.array(cutouts[str(idx)]['HDU1']['DATA']).astype(int) & (1 | (1 << 8) | (1 << 9)))
            bad_mask = skimage.transform.resize(bad_mask.astype(bool), (224,224))
            _, _, _, mask = calc_icl_frac(cutout, bad_mask, zs[idx], return_mask=True)
            plt.imshow(stretch(mask) * ~bad_mask, cmap=cmap, interpolation='none')
            plt.title(fracs[idx])
            plt.xticks([])
            plt.yticks([])

            # Show the rank comparison
            plt.subplot(6, 4, sp_row+4)
            plt.scatter(fracs[not_nans], results, c='gray', marker='.')
            plt.scatter(fracs[idx], results[i], c='r')
            plt.plot([0, 0.35], [0, 0.35], 'k--')
            plt.xticks(fontsize='x-small')
            plt.yticks(fontsize='x-small')

            if sp_row + 4 == 24:
                # end of page
                pdf.savefig()
                plt.close()
                plt.figure(figsize=(8,12), dpi=150)
        pdf.savefig()
        plt.close()

# Adapted from https://keras.io/examples/vision/grad_cam/ and https://github.com/tensorflow/tensorflow/issues/44462 

import tensorflow as tf
from tensorflow import keras

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

from model import NNCLR
from augmentations import augmenter

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import pickle
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Parameters
input_shape = (224,224,1)
temperature = 0.1
queue_size = 1000
checkpoint_path = '/srv/scratch/z5214005/checkpoint'
stddev = 0.017359 # Calculated elsewhere from first 1000 cutouts
num_epochs = 100

num_components = 16

# Preprocessing function
def preprocess(image, label):
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image, label

def val_preprocess(data):
    image = data['image']
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image

def regression_model():
    # Load the pretrained encoder
    base_model = NNCLR(input_shape=input_shape, temperature=temperature, queue_size=queue_size)
    base_model.load_weights(checkpoint_path).expect_partial()
    base_model.encoder.trainable = False
    from augmentations import val_augmenter
    # Instantiate the augmenter
    model_augmenter = augmenter(input_shape=input_shape)

    # Define the model with added regression head
    inputs = keras.Input(shape=input_shape)
    x = model_augmenter(inputs)
    x = base_model.encoder(x)

    s = tf.reduce_sum(base_model.encoder.losses)
    base_model.encoder.add_loss(lambda: -s)
        
    x = keras.layers.Dense(256, activation='tanh')(x)

    # Probabilistic modelling - from Francois's tutorial
    x = keras.layers.Dense(units=num_components*3)(x)
    outputs = tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
        tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=tf.expand_dims(t[..., :num_components], -2)),
                              components_distribution=tfp.distributions.Beta(1 + tf.nn.softplus(tf.expand_dims(t[..., num_components:2*num_components], -2)),
                                                               1 + tf.nn.softplus(tf.expand_dims(t[..., 2*num_components:],-2)))), 1))(x)    
    # outputs = keras.layers.Dense(1, activation='sigmoid')(x) # Regression layer

    return keras.Model(inputs, outputs)

def prepare_validation_data(fracs_path='/srv/scratch/z5214005/precalc_fracs/fracs.npy'):
    val_init_dataset = tfds.load('hsc_icl', split='train')
    dud_only_dataset = val_init_dataset.filter(lambda x: x['id'] < 125)
    np_val_data = dud_only_dataset.as_numpy_iterator()
    validation_imgs = sorted(np_val_data, key=lambda x: x['id'])
    validation_imgs = np.array(list(map(val_preprocess, validation_imgs)))
    fracs_all = np.load(fracs_path)[2]
    fracs = fracs_all[~np.isnan(fracs_all)]
    validation_imgs = validation_imgs[~np.isnan(fracs_all)]

    return (validation_imgs, fracs)

def make_graph(model, validation_data, filename='val_graph.png'):
    # Look at the performance of the model
    validation_imgs, fracs = validation_data

    # Run the model and calculate the Spearman coefficient
    predictions = model(validation_imgs).mean().numpy().squeeze()
    rankings = np.argsort(np.argsort(predictions)[::-1])
    fracs_ordered = np.argsort(fracs)[::-1]
    fracs_rankings = np.argsort(fracs_ordered)
    print(spearmanr(rankings, fracs_rankings).statistic)

    # Plot the predicted values against the true values
    plt.scatter(np.arange(0, len(fracs)), fracs[fracs_ordered])
    plt.scatter(np.arange(0, len(fracs)), predictions[fracs_ordered])
    plt.legend(['Target', 'Actual'])
    plt.xlabel('Clusters (ranked)')
    plt.ylabel('Fraction')
    plt.savefig(filename)
    plt.close()

    # Print the loss
    mse = tf.keras.losses.MeanSquaredError()
    print(mse(fracs, predictions).numpy())

def finetune():
    model = regression_model()

    # Using probabilistic model - change loss
    negloglik = lambda y, p_y: -p_y.log_prob(y)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
                loss=negloglik)

    # Instantiate the dataset
    ds = (tfds.load('finetuning_data', split='train', shuffle_files=True, as_supervised=True)
        .shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        .batch(50)
    )
    dataset = ds.map(preprocess)

    # Prepare validation data
    validation_data = prepare_validation_data()
    validation_dataset = (tf.data.Dataset.from_tensor_slices(validation_data)
                          .batch(125)
    )

    # Create a checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='checkpoint-finetune.ckpt',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Train the model
    train_history = model.fit(dataset, validation_data=validation_dataset,  epochs=num_epochs, callbacks=[cp_callback])

    # Save the history
    with open('history.pkl', 'wb') as f:
        pickle.dump(train_history.history, f)

    make_graph(model, validation_data)

    ds_np = list(dataset.unbatch().as_numpy_iterator())
    images = []
    labels = []
    for thing in ds_np:
        images.append(thing[0])
        labels.append(thing[1])
    images = np.array(images)
    labels = np.array(labels)

    make_graph(model, (images, labels), filename='train_graph.png')

if __name__ == '__main__':
    finetune()
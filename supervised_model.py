import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import sys

import resnet_cifar10_v2
from augmentations import augmenter

N = 2
DEPTH = N*9+2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1
stddev = 0.017359

# Idk where to put this for now
from scipy.stats import gaussian_kde
def get_weights(dataset):
    dataset = dataset.unbatch()
    fracs = []
    for element in dataset:
        fracs.append(element[1])
    kernel = gaussian_kde(fracs)
    pts = np.arange(0, max(fracs), 0.01)
    probs = kernel(pts)
    weights = 1/probs
    weights = tf.clip_by_value(weights, 1, 5)

    return tf.convert_to_tensor(weights)

# Preprocessing function with weights
def preprocess_with_weights(image, label, weights):
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image, label, tf.gather(weights, tf.cast(tf.math.round(label * 100), tf.dtypes.int32))

# Preprocessing function
def preprocess(image, label):
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image, label

# Load and split the dataset into train and test set
def prepare_data():
    ds = (tfds.load('supervised_data', split='train', data_dir='/srv/scratch/z5214005/tensorflow_datasets', as_supervised=True)
    )

    dataset, validation_dataset = keras.utils.split_dataset(ds, left_size=0.9, right_size=0.1, shuffle=False)
    dataset = dataset.batch(50)
    validation_dataset = validation_dataset.batch(100)

    weights = get_weights(dataset)

    # dataset = dataset.map(lambda x,y: preprocess_with_weights(x, y, weights))
    dataset = dataset.map(preprocess)
    validation_dataset = validation_dataset.map(preprocess)
    
    return dataset, validation_dataset

def encoder(input_shape):
    inputs = layers.Input(input_shape, name="encoder_input")
    x = resnet_cifar10_v2.stem(inputs)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    outputs = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

    return keras.Model(inputs, outputs, name="encoder")

# Define the model
class ImageRegressor(keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.augmenter = augmenter(input_shape)
        self.encoder = encoder(input_shape)
        self.regressor = keras.Sequential([
            layers.Input((256), name='regressor'),
            layers.Dense(2048, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, inputs):
        x = self.augmenter(inputs) 
        x = self.encoder(x)
        outputs = self.regressor(x)
        return outputs

# Compile and train the model
def train(model, train_data, val_data, epochs=100, file_ext=''):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoint-sup-{file_ext}.ckpt',
        save_best_only=True,
        save_weights_only=True
    )

    stop_callback = keras.callbacks.EarlyStopping(patience=10)

    train_history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[cp_callback, stop_callback])

    return model

def binned_plot(dataset, Y, filename='binned_plot.png', n=10, percentiles=[35, 50], ax=None, **kwargs):
    unbatched = dataset.unbatch()
    
    labels = []
    for i, thing in enumerate(unbatched):
        labels.append(thing[1])
    X = np.array(labels)

    # Calculation
    calc_percent = []
    for p in percentiles:
        if p < 50:
            calc_percent.append(50-p)
            calc_percent.append(50+p)
        elif p == 50:
            calc_percent.append(50)
        else:
            raise Exception('Percentile > 50')
    
    bin_edges = np.linspace(X.min()*0.9999, X.max()*1.0001, n+1)

    dtype = [(str(i), 'f') for i in calc_percent]
    bin_data = np.zeros(shape=(n,), dtype=dtype)

    for i in range(n):
        y = Y[(X >= bin_edges[i]) & (X < bin_edges[i+1])]

        if len(y) == 0:
            continue

        y_p = np.percentile(y, calc_percent)

        bin_data[i] = tuple(y_p)

    # Plotting
    if ax is None:
        f, ax = plt.subplots()

    bin_centers = [np.mean(bin_edges[i:i+2]) for i in range(n)]
    for p in percentiles:
        if p == 50:
            ax.plot(bin_centers, bin_data[str(p)], **kwargs)
        else:
            ax.fill_between(bin_centers,
                            bin_data[str(50-p)],
                            bin_data[str(50+p)],
                            alpha=0.2,
                            **kwargs)
    
    # Plot the expected line
    ax.plot(np.linspace(bin_centers[0],bin_centers[-1],10),np.linspace(bin_centers[0],bin_centers[-1],10),'k--')
    
    f.savefig(fname=filename)
    plt.close()

    return bin_data, bin_edges

def scatter_plot(dataset, predictions, filename):
    unbatched = dataset.unbatch()

    # Grab the first 1000 elements for plotting
    labels = []
    for i, thing in enumerate(unbatched):
        if i == 1000:
            break
        labels.append(thing[1])
    labels = np.array(labels)

    ordered = np.argsort(labels)[::-1]

    plt.plot(np.arange(0, len(labels)), labels[ordered])
    plt.scatter(np.arange(0, len(labels)), predictions[ordered], color='C1')
    plt.fill_between(np.arange(0, len(labels)), labels[ordered]-0.04, labels[ordered]+0.04, alpha=0.2)
    plt.legend(['Target', 'Actual'])
    plt.xlabel('Clusters (ranked)')
    plt.ylabel('Fraction') 
    plt.savefig(filename)
    plt.close()

def plot_results(model, train_data, val_data, file_ext):  
    # Create scatter plots showing the spread of data
    predictions = model.predict(train_data).squeeze()
    scatter_plot(train_data, predictions[:1000], f'train_graph-{file_ext}.png')
    
    predictions = model.predict(val_data).squeeze()
    scatter_plot(val_data, predictions[:1000], f'val_graph-{file_ext}.png')

    # Create binned plot
    binned_plot(val_data, predictions, percentiles=[35,45,50], color='b', filename=f'binned_plot-{file_ext}.png')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_ext = sys.argv[1]
    else:
        file_ext = ''

    dataset, validation_dataset = prepare_data()

    model = ImageRegressor((224,224,1))
    # model.load_weights('checkpoint-sup-x100.ckpt').expect_partial()

    model = train(model, dataset, validation_dataset, file_ext=file_ext)

    plot_results(model, dataset, validation_dataset, file_ext=file_ext)

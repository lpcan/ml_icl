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
num_components = 16

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
    ds = (tfds.load('supervised_data', split='train', data_dir='/srv/scratch/mltidal/tensorflow_datasets', as_supervised=True)
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
    def __init__(self, input_shape, mean=0.948, std=1.108):
        super().__init__()
        self.augmenter = augmenter(input_shape, mean=mean, std=std)
        self.encoder = encoder(input_shape)
        self.regressor = keras.Sequential([
            layers.Input((256), name='regressor'),
            layers.Dense(2048, activation='relu'),
            # layers.Dense(256, activation='relu'),
            # layers.Dense(1)
        ])
        self.prob_output = keras.Sequential([
            layers.Dense(units=num_components*3),
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
                tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=tf.expand_dims(t[..., :num_components], -2)),
                                                    components_distribution=tfp.distributions.Beta(1 + tf.nn.softplus(tf.expand_dims(t[..., num_components:2*num_components], -2)),
                                                                                                   1 + tf.nn.softplus(tf.expand_dims(t[..., 2*num_components:],-2)))), 1))
        ])

    def call(self, inputs):
        x = self.augmenter(inputs) 
        x = self.encoder(x)
        x = self.regressor(x)
        outputs = self.prob_output(x)
        return outputs

# Compile and train the model
def train(model, train_data, val_data, epochs=100, file_ext=''):
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoint-sup-{file_ext}.ckpt',
        save_best_only=True,
        save_weights_only=True
    )

    stop_callback = keras.callbacks.EarlyStopping(patience=10)

    train_history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[cp_callback])

    model.save_weights(f'checkpoint-sup-{file_ext}-final.ckpt')

    return model

def binned_plot(dataset, Y, filename='binned_plot.png', n=10, percentiles=[35, 50], ax=None, **kwargs):
    unbatched = dataset.unbatch()
    
    labels = []
    for i, thing in enumerate(unbatched):
        labels.append(thing[1])
    X = np.array(labels)

    print(f'MAE = {np.mean(np.abs(X-Y))}')

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

    empty_bins = []

    for i in range(n):
        y = Y[(X >= bin_edges[i]) & (X < bin_edges[i+1])]

        if len(y) == 0:
            empty_bins.append(i)
            continue

        y_p = np.percentile(y, calc_percent)

        bin_data[i] = tuple(y_p)

    # Plotting
    if ax is None:
        f, ax = plt.subplots()

    bin_centers = [np.mean(bin_edges[i:i+2]) for i in range(n)]

    # Remove empty bins
    bin_data = np.delete(bin_data, empty_bins, 0)
    for bin_num in empty_bins[::-1]: # in reverse order so indices are still valid
        bin_centers.pop(bin_num)

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
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='b', label='Mean prediction'),
                       Patch(facecolor='b', alpha=0.4,
                         label='70th percentile'),
                       Patch(facecolor='b', alpha=0.2,
                         label='90th percentile')]
    plt.legend(handles=legend_elements)
    
    plt.xlabel('Expected fraction')
    plt.ylabel('Predicted fraction')
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
    train_subset = train_data.take(20)
    val_subset = val_data#.take(10) 
    # Create scatter plots showing the spread of data
    predictions = []
    for batch in train_subset:
        predictions.append(model(batch[0]).mean().numpy().squeeze())
    predictions = np.array(predictions).flatten()
    scatter_plot(train_subset, predictions, f'train_graph-{file_ext}.png')
    
    predictions = []
    for batch in val_subset:
        predictions.append(model(batch[0]).mean().numpy().squeeze())
    predictions = np.array(predictions).flatten()
    scatter_plot(val_subset, predictions, f'val_graph-{file_ext}.png')

    # Create binned plot
    binned_plot(val_subset, predictions, n=20, percentiles=[35,45,50], color='b', filename=f'binned_plot-{file_ext}.png')

def plot_results_mode(model, train_data, val_data, file_ext):
    train_subset = train_data.take(20)
    val_subset = val_data.take(10) 
    # Create scatter plots showing the spread of data
    x = np.arange(0, 1, 0.001)
    predictions = []
    for batch in train_subset:
        outputs = model(batch[0])
        logps = []
        for i in x:
            logps.append(outputs.log_prob(i).numpy())
        logps = np.stack(logps)
        predictions.append(x[np.exp(logps).argmax(axis=0)])
    predictions = np.array(predictions).flatten()
    scatter_plot(train_subset, predictions, f'train_graph-{file_ext}.png')
    
    predictions = []
    for batch in val_subset:
        outputs = model(batch[0])
        logps = []
        for i in x:
            logps.append(outputs.log_prob(i).numpy())
        logps = np.stack(logps)
        predictions.append(x[np.exp(logps).argmax(axis=0)])
    predictions = np.array(predictions).flatten()
    scatter_plot(val_subset, predictions, f'val_graph-{file_ext}.png')

    # Create binned plot
    binned_plot(val_subset, predictions, n=20, percentiles=[35,45,50], color='b', filename=f'binned_plot-{file_ext}.png')

def plot_loss(jobnumber):
    f = open(f'sup_train.pbs.o{jobnumber}')
    loss = []
    val_loss = []
    for line in f:
        words = line.split(' ')
        if 'loss:' in words:
            loss.append(float(words[7]))
            val_loss.append(float(words[-1]))
    
    plt.plot(np.arange(len(loss)), loss)
    plt.plot(np.arange(len(val_loss)), val_loss)
    plt.legend(['Train loss', 'Val loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Loss graph')
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_ext = sys.argv[1]
    else:
        file_ext = ''

    dataset, validation_dataset = prepare_data()

    model = ImageRegressor((224,224,1))

    negloglik = lambda y, p_y: -p_y.log_prob(y)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=negloglik)
    
    model.load_weights('checkpoint-sup-expdatacont1-final.ckpt').expect_partial()

    model = train(model, dataset, validation_dataset, epochs=100, file_ext=file_ext)

    plot_results(model, dataset, validation_dataset, file_ext=file_ext)

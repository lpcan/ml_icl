
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
import sys
import wandb
from wandb.keras import WandbMetricsLogger

import resnet_cifar10_v2
from augmentations import augmenter
from contrastive_model.scripts.model import NNCLR

N = 6
DEPTH = N*9+2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1
stddev = 0.017359
num_components = 16

# Preprocessing function
def preprocess(image, label):
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    label = tf.math.maximum(label, 1e-9)
    label = label
    # label = tf.math.tanh(label / 400) # / 2868 # normalise the value between 0 and 1
    return image, label

from scipy.stats import gaussian_kde
def get_weights(dataset):
    dataset = dataset.unbatch()
    fracs = []
    for element in dataset:
        fracs.append(element[1])
    kernel = gaussian_kde(fracs)
    pts = np.arange(0, np.round(max(fracs), 2), 0.01)
    probs = kernel(pts)
    weights = 1/probs
    # Squash the weights between 1 and 10
    # weights = ((weights - tf.math.reduce_min(weights)) / (tf.math.reduce_max(weights) - tf.math.reduce_min(weights)) * 19) + 1
    # Clip weights between 1 and 10, then stretch to between 1 and 19. 
    weights = tf.clip_by_value(weights, 1, 10) * 5 - 1
    return tf.convert_to_tensor(weights)

# Preprocessing function with weights
def preprocess_with_weights(image, label, weights):
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image, label, tf.gather(weights, tf.cast(tf.math.round(label * 100), tf.dtypes.int32))

# Load and split the dataset into train and test set
def prepare_data():
    dataset, validation_dataset = (tfds.load('supervised_data', split=['train[:90%]', 'train[90%:]'], data_dir='/srv/scratch/mltidal/tensorflow_datasets', as_supervised=True)
    )

    # dataset, validation_dataset = keras.utils.split_dataset(ds, left_size=0.9, right_size=0.1, shuffle=False)
    dataset = dataset.batch(50)
    validation_dataset = validation_dataset.batch(100)
    # weights = get_weights(dataset)
    # dataset = dataset.map(lambda x,y: preprocess_with_weights(x, y, weights))
    # validation_dataset = validation_dataset.map(lambda x,y: preprocess_with_weights(x, y, weights))
    dataset = dataset.map(preprocess)
    validation_dataset = validation_dataset.map(preprocess)
    
    return dataset, validation_dataset

def encoder(input_shape):
    inputs = layers.Input(input_shape, name="encoder_input")
    x = resnet_cifar10_v2.stem(inputs)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
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
    stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6)
    train_history = model.fit(train_data, validation_data=val_data, 
                              epochs=epochs, 
                              callbacks=[
                                  cp_callback,
                                  lr_callback, 
                                  stop_callback,
                                  WandbMetricsLogger(),
                                ],
                                verbose=2)
    print(f'Saving model as {file_ext}')
    model.save_weights(f'checkpoint-sup-{file_ext}-final.ckpt')

    # Save all the checkpoint files to wandb
    wandb.save(f'checkpoint-sup-{file_ext}.ckpt*')
    wandb.save(f'checkpoint-sup-{file_ext}-final.ckpt*')

    return model

# Results plotting - code mostly from Francois
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

    # ax.scatter(X, Y, facecolors='None', edgecolors='gray', alpha=0.3)

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
    
    plt.xlabel('Actual fraction')
    plt.ylabel('Predicted fraction')
    f.savefig(fname=filename)
    
    plt.close()

    return bin_centers, bin_data

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

def plot_results_mode(model, train_data, val_data, file_ext, send_to_wandb=False):
    val_subset = val_data.take(100) 

    x = np.arange(0, 1, 0.001)
    
    predictions = []
    for batch in val_subset:
        outputs = model(batch[0])
        logps = []
        for i in x:
            logps.append(outputs.log_prob(i).numpy())
        logps = np.stack(logps)
        predictions.append(x[np.exp(logps).argmax(axis=0)])
    predictions = np.array(predictions).flatten()
    
    # Create binned plot
    bin_centers, bin_data = binned_plot(val_subset, predictions, n=20, percentiles=[35,45,50], color='b', filename=f'binned_plot-{file_ext}.png')
    
    if send_to_wandb:
        # Just plot the central line
        data = [[x, y] for (x, y) in zip(bin_centers, bin_data['50'])]
        table = wandb.Table(data=data, columns=['Actual fraction', 'Predicted fraction'])
        wandb.log(
            {
                'binned_plot_id': wandb.plot.line(
                    table, 'Actual fraction', 'Predicted fraction', title='Binned validation plot'
                )
            }
        )

def plot_loss(jobnumbers):
    loss = []
    val_loss = []
    for job in jobnumbers:
        try:
            f = open(f'sup_train.pbs.o{job}')
        except:
            f = open(f'jobs/job_outputs/sup_train.pbs.o{job}')
        for line in f:
            words = line.split(' ')
            if 'loss:' in words:
                loss.append(float(words[words.index('loss:')+1]))
                val_loss.append(float(words[words.index('val_loss:')+1]))
    
    plt.plot(np.arange(len(loss)), loss)
    plt.plot(np.arange(len(val_loss)), val_loss)
    plt.legend(['Train loss', 'Val loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('Loss graph')
    plt.close()

def val_preprocess(data):
    image = data['image']
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image

def prepare_validation_data(fracs_path='/srv/scratch/mltidal/fracs_resized.npy'):
    val_init_dataset = tfds.load('hsc_icl', split='train', data_dir='/srv/scratch/mltidal/tensorflow_datasets')
    dud_only_dataset = val_init_dataset.filter(lambda x: x['id'] < 125)
    np_val_data = dud_only_dataset.as_numpy_iterator()
    validation_imgs = sorted(np_val_data, key=lambda x: x['id'])
    validation_imgs = np.array(list(map(val_preprocess, validation_imgs)))
    # fracs_all = np.tanh(np.load(fracs_path)[0] / 400)
    fracs_all = np.load(fracs_path)
    if len(fracs_all) == 3:
        fracs_all = fracs_all[2]
    fracs = fracs_all[~np.isnan(fracs_all)]
    validation_imgs = validation_imgs[~np.isnan(fracs_all)]
    return (validation_imgs, fracs)

def test_real_data(model, file_ext, fracs_path='/srv/scratch/mltidal/fracs_actual.npy', send_to_wandb=False):
    validation_data = prepare_validation_data(fracs_path=fracs_path)
    # Look at the performance of the model
    validation_imgs, expected = validation_data
    # Run the model and calculate the Spearman coefficient
    # predictions = model(validation_imgs).mean().numpy().squeeze()
    x = np.arange(0, 0.6, 0.0005)
    outputs = model(validation_imgs)
    logps = []
    logcs = []
    for i in x:
        logps.append(outputs.log_prob(i).numpy())
        logcs.append(outputs.log_cdf(i).numpy())
    logps = np.stack(logps)
    logcs = np.stack(logcs)
    predictions = x[np.exp(logps).argmax(axis=0)]
    # Plot the predicted values against the true values
    q15s = np.argmax(np.exp(logcs) >= 0.15, axis=0)
    q85s = np.argmax(np.exp(logcs) >= 0.85, axis=0)
    lower_errors = np.abs(predictions - x[q15s])
    upper_errors = np.abs(x[q85s] - predictions)
    # error = 0.04
    sorted_idxs = np.argsort(expected)
    binned_expected = np.array_split(expected[sorted_idxs], 5)
    binned_predicted = np.array_split(predictions[sorted_idxs], 5)

    # Calculate the median of the binned results
    x = []
    y = []
    xerr_l = []
    xerr_h = []
    for i in range(len(binned_expected)):
        x_med = np.median(binned_expected[i])
        x.append(x_med)
        y_med = np.median(binned_predicted[i])
        y.append(y_med)

        xerr_l.append(x_med - np.min(binned_expected[i]))
        xerr_h.append(np.max(binned_expected[i]) - x_med)

    plt.errorbar(expected, predictions, fmt='none', yerr=(lower_errors, upper_errors), alpha=0.2, color='gray')
    plt.plot(expected, predictions, '.', color='gray', alpha=0.3)
    plt.plot(x, y, 'or')
    plt.plot(x, y, 'r')
    plt.errorbar(x, y, fmt='none', xerr=(xerr_l, xerr_h), color='red')
    maxval = np.max([expected, predictions])
    plt.plot([0, maxval], [0, maxval], 'k--')
    plt.xlabel('Actual fraction')
    plt.ylabel('Predicted fraction')

    plt.savefig(f'test_graph-{file_ext}.png')
    plt.close()
    # Print the result
    print(f'MAE = {np.mean(np.abs(expected - predictions))}')
    print(pearsonr(expected, predictions))

    if send_to_wandb:
        data = [[x, y] for (x, y) in zip(expected, predictions)]
        table = wandb.Table(data=data, columns=['Actual', 'Predictions'])
        wandb.log({'test_graph': wandb.plot.scatter(table, 'Actual', 'Predictions')})

    return (expected, predictions)

def load_model(model_name=None, lr=1e-4):
    model = ImageRegressor((224,224,1))
    # pretrained_model = NNCLR((224,224,1), temperature=0.1, queue_size=1000)
    # pretrained_model.load_weights('checkpoint-alldata50nobatchnorm-final.ckpt').expect_partial()
    # model.encoder = pretrained_model.encoder

    negloglik = lambda y, p_y: -p_y.log_prob(y)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=negloglik)
    if model_name is not None: 
        model.load_weights(f'checkpoint-sup-{model_name}.ckpt').expect_partial()

    return model

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_ext = sys.argv[1]
    else:
        file_ext = ''

    wandb.init(
    project='ml-icl',
    config={
        'epochs': 100, 
        'train_batch_size': 50,
        'val_batch_size': 100,
        'optimizer': 'adam',
    },
    id='2blsd9rl',
    resume='must'
    )

    dataset, validation_dataset = prepare_data()

    model = load_model(model_name='withbkg', lr=1e-5)

    model = train(model, dataset, validation_dataset, epochs=100, file_ext=file_ext)

    plot_results_mode(model, dataset, validation_dataset, file_ext=file_ext, send_to_wandb=True)
    test_real_data(model, file_ext, fracs_path='/srv/scratch/mltidal/fracs_manual_updated.npy', send_to_wandb=True)

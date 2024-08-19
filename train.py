"""
Train the model and output performance plots
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import pearsonr
import sys
from tensorflow import keras
import wandb
from wandb.keras import WandbMetricsLogger
 
import prep_data as datasets
import model as modeller

def train(model, train_data, val_data, epochs=100, file_ext=''):
    # Define some callbacks for training
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoint-sup-{file_ext}.ckpt',
        save_best_only=True,
        save_weights_only=True
    )
    stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)
    lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                    patience=10, verbose=1, 
                                                    min_lr=1e-5)
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

def binned_plot(dataset, Y, ds_numpy=None, filename='binned_plot.png', n=10, percentiles=[35, 50], ax=None, **kwargs):
    """
    Display the results of the training in a binned plot
    Code adapted from @maho3 (Matt Ho) and @EiffL (Francois Lanusse)
    (https://github.com/EiffL/Tutorials/blob/2211542b4bb313673dad48a5845c51fc6d2bc109/ClusterMasses/TFP_mass_estimate.ipynb)
    """
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = 'b'

    # Make sure the dataset is in the right format
    if ds_numpy is None:
        unbatched = dataset.unbatch()
        labels = []
        for i, thing in enumerate(unbatched):
            labels.append(thing[1])
        X = np.array(labels)
    else:
        X = ds_numpy

    print(f'MAE = {np.mean(np.abs(X-Y))}')

    # Prepare arrays for percentiles and bins
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

    # Populate bin data
    for i in range(n):
        y = Y[(X >= bin_edges[i]) & (X < bin_edges[i+1])]

        if len(y) == 0:
            empty_bins.append(i)
            continue

        y_p = np.percentile(y, calc_percent)

        bin_data[i] = tuple(y_p)
    
    bin_centers = [np.mean(bin_edges[i:i+2]) for i in range(n)]
    
    # Plotting
    if ax is None:
        f, ax = plt.subplots()

    # Remove empty bins
    bin_data = np.delete(bin_data, empty_bins, 0)
    for bin_num in empty_bins[::-1]: # in reverse order so indices are still valid
        bin_centers.pop(bin_num)

    # Plot the expected line
    ax.plot(np.linspace(bin_centers[0],bin_centers[-1],10),np.linspace(bin_centers[0],bin_centers[-1],10),'k--')

    # Plot the percentile areas
    for p in percentiles:
        if p == 50:
            ax.plot(bin_centers, bin_data[str(p)], **kwargs)
        else:
            ax.fill_between(bin_centers,
                            bin_data[str(50-p)],
                            bin_data[str(50+p)],
                            alpha=0.2,
                            **kwargs)
    
    # Create legend
    legend_elements = [Line2D([0], [0], color=color, label='Mean prediction'),
                       Patch(facecolor=color, alpha=0.4,
                         label='70th percentile'),
                       Patch(facecolor=color, alpha=0.2,
                         label='90th percentile')]
    ax.legend(handles=legend_elements)

    ax.set_xlabel('Actual ICL fraction')
    ax.set_ylabel('Predicted ICL fraction')
    plt.savefig(fname=filename)
    
    plt.close()

    return bin_centers, bin_data

def plot_binned_plot(model, val_data, file_ext, send_to_wandb=False):
    """
    Get the model's predictions on the validation data by taking the mode of the
    output probability distributions. Use these for creating the binned plot
    """
    x = np.arange(0, 1, 0.001)

    predictions = []
    for batch in val_data:
        outputs = model(batch[0])
        logps = []
        for i in x: 
            logps.append(outputs.log_prob(i).numpy())
        logps = np.stack(logps)
        predictions.append(x[np.exp(logps).argmax(axis=0)])
    predictions = np.array(predictions).flatten()

    # Create binned plot
    bin_centers, bin_data = binned_plot(val_data, predictions, n=20, percentiles=[35,45,50], color='b', filename=f'binned_plot-{file_ext}.png')

    # Optionally send to wandb
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
    """
    Convoluted way to make a loss plot from the output file
    """
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

def test_real_data(model, file_ext, imgs_path, fracs_path, send_to_wandb=False):
    """
    Plot the model's performance on real data (no finetuning)
    """
    if imgs_path is None:
        test_imgs = datasets.prepare_test_data()
    else:
        test_imgs = np.load(imgs_path)
    expected = np.load(fracs_path)[2]
    test_imgs = test_imgs[~np.isnan(expected)]
    expected = expected[~np.isnan(expected)]

    # Run the model on the test images
    x = np.arange(0, 0.6, 0.0005)
    outputs = model(test_imgs)
    logps = []
    logcs = []
    for i in x:
        logps.append(outputs.log_prob(i).numpy())
        logcs.append(outputs.log_cdf(i).numpy())
    logps = np.stack(logps)
    logcs = np.stack(logcs)
    predictions = x[np.exp(logps).argmax(axis=0)]

    # Calculate the model's uncertainty
    q15s = np.argmax(np.exp(logcs) >= 0.15, axis=0)
    q85s = np.argmax(np.exp(logcs) >= 0.85, axis=0)
    lower_errors = np.abs(predictions - x[q15s])
    upper_errors = np.abs(x[q85s] - predictions)

    # Load the measurement errors
    xerror = np.load('/srv/scratch/mltidal/err_photoz.npy')
    xerror = xerror[~np.isnan(xerror)]
    xerror = expected * xerror
    sorted_idxs = np.argsort(expected)

    # Bin the values and calculate median of binned results
    binned_expected = np.array_split(expected[sorted_idxs], 5)
    binned_predicted = np.array_split(predictions[sorted_idxs], 5)
    x = []
    y = []
    xerr_l = []
    xerr_h = []
    yerr = []
    for i in range(len(binned_expected)):
        x_med = np.median(binned_expected[i])
        x.append(x_med)
        y_med = np.median(binned_predicted[i])
        y.append(y_med)

        xerr_l.append(x_med - np.min(binned_expected[i]))
        xerr_h.append(np.max(binned_expected[i]) - x_med)
        yerr.append(np.std(binned_predicted[i]))

    # Plot the predicted values against the true values
    plt.errorbar(expected, predictions, fmt='none', yerr=(lower_errors, upper_errors), xerr=xerror, alpha=0.2, color='gray')
    plt.plot(expected, predictions, '.', color='gray', alpha=0.3)
    plt.plot(x, y, 'or')
    plt.plot(x, y, 'r')
    plt.errorbar(x, y, fmt='none', xerr=(xerr_l, xerr_h), yerr=yerr, color='red')
    maxval = np.max([expected, predictions])
    plt.plot([0, maxval], [0, maxval], 'k--')
    plt.xlabel('Actual ICL fraction')
    plt.ylabel('Predicted ICL fraction')

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
    # id='vx6bs1h5', # resume wandb run
    # resume='must'
    )

    dataset, validation_dataset = datasets.prepare_training_data()

    model = modeller.load_model(model_name=None, lr=1e-4)

    # model = train(model, dataset, validation_dataset, epochs=90, file_ext=file_ext)

    plot_binned_plot(model, validation_dataset, file_ext=file_ext, send_to_wandb=True)
    test_real_data(model, file_ext, imgs_path='badmaskimgs_300kpc.npy', fracs_path='/srv/scratch/mltidal/fracs_manual_photoz.npy', send_to_wandb=True)
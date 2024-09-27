import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from tensorflow import keras
import tensorflow as tf
import pickle

from model import load_model
from prep_data import prepare_test_data

MODEL_VERSION = 'checkpoint-trained'
EPOCHS = 100
FRACS_PATH = 'fracs_manual_photoz.npy'
PREPPED_IMAGES_PATH = 'data/processed/badmaskimgs_300kpc.npy' # Set to None to prepare data from HDF file
HDF_PATH = None
SAVE_AS = 'checkpoints/test_checkpoint' # Checkpoint filename

def flatten(list):
    return np.array([i for row in list for i in row])

def prepare_data():
    # Create the finetuning dataset
    fracs = np.load(FRACS_PATH)[2]
    zeros = np.where(fracs == 0)[0]
    if len(zeros) > 0:
        fracs[zeros] = 0.00001

    not_nans = np.where(~np.isnan(fracs))[0]

    if PREPPED_IMAGES_PATH is None:
        images = prepare_test_data(HDF_PATH, save=False)
        images = images[not_nans]
    else:
        images = np.load(PREPPED_IMAGES_PATH)
        images = images[not_nans] 

    fracs = fracs[not_nans]

    return fracs, images

def create_model():
    # Create the model
    finetune_model = load_model(model_name=MODEL_VERSION, lr=1e-7)

    # Freeze the entire model other than the dense layers
    for layer in finetune_model._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())

        if len(lst_of_sublayers) == 1: # leaves of the model
            if layer.name in ['dense_1', 'dense_2']:
                layer.trainable = True
            else:
                layer.trainable = False

    return finetune_model

def final_finetune():
    # Finetune on all of the data and save the final model
    fracs, images = prepare_data()

    finetune_model = create_model()

    finetune_model.fit(images, fracs, epochs=EPOCHS)

    print(f'Saving model as {MODEL_VERSION}')
    finetune_model.save_weights(f'{SAVE_AS}.ckpt')

def finetune_one_split():
    # Finetune on just one split of the data
    fracs, images = prepare_data()

    # Split the data for training and testing
    rng = np.random.default_rng(seed=24)
    idxs = np.arange(len(fracs))
    rng.shuffle(idxs)
    splits = np.array_split(idxs, 5)
    test_set = splits[0]
    train_set = np.concatenate(splits[1:])
    test_ds = images[test_set]
    test_labels = fracs[test_set]
    train_ds = images[train_set]
    train_labels = fracs[train_set]

    # Create and train the model
    finetune_model = create_model()

    # Check that the model is functioning as we expect
    finetune_model.evaluate(test_ds, test_labels)
    predictions = finetune_model.predict(test_ds)

    print(f'MAE = {np.mean(np.abs(test_labels - predictions))}')

    # Train the model
    finetune_model.fit(train_ds, train_labels, validation_data=(test_ds, test_labels), epochs=EPOCHS)

    # Evaluate the model
    finetune_model.evaluate(test_ds, test_labels)
    x = np.arange(0, 0.6, 0.0005)
    dists = outputs.distribution.prob(x) # Get all the output probability distributions as arrays
    predictions = x[np.argmax(dists, axis=1)] # Find the peaks of the distributions
    cdfs = outputs.distribution.cdf(x) # Get the CDFs

    # Get the 15 and 85 percentile uncertainties
    q15s = np.argmax(cdfs >= 0.15, axis=1)
    q85s = np.argmax(cdfs >= 0.85, axis=1)
    lower_errors = np.abs(predictions - x[q15s])
    upper_errors = np.abs(x[q85s] - predictions)

    print(f'MAE = {np.mean(np.abs(test_labels - predictions))}')
    print(pearsonr(test_labels, predictions))

    # Plot the result of this finetuning run
    plt.errorbar(test_labels, predictions, fmt='none', yerr=(lower_errors, upper_errors), alpha=0.3)
    plt.scatter(test_labels, predictions)
    maxval = np.max([test_labels, predictions])
    plt.plot([0,maxval], [0,maxval], 'k--')
    plt.xlabel('Expected')
    plt.ylabel('Predicted')
    plt.savefig('result.png')
    plt.close()

    # Save the results so the plot can be recreated
    with open(f'{SAVE_AS}-results.pkl', 'wb') as fp:
        pickle.dump([test_results, err_l, err_h], fp)

# Finetuning on separate splits of the data
if __name__=='__main__':
    fracs, images = prepare_data()

    # Get the splits for cross-validation
    k = 5
    rng = np.random.default_rng(seed=24)
    idxs = np.arange(len(fracs))
    rng.shuffle(idxs)
    splits = np.array_split(idxs, k)

    test_results = []
    err_l = []
    err_h = []

    # Train different versions of the model
    for run in range(k):
        # Create the datasets for this split
        test_set = splits[run]
        train_set = splits.copy()
        train_set.pop(run)
        train_set = np.concatenate(train_set)

        test_ds = images[test_set]
        test_labels = fracs[test_set]
        train_ds = images[train_set]
        train_labels = fracs[train_set]

        finetune_model = create_model()

        # Check that the model is functioning as we expect
        finetune_model.evaluate(test_ds, test_labels)
        predictions = finetune_model.predict(test_ds)

        print(f'MAE = {np.mean(np.abs(test_labels - predictions))}')

        # Train the model
        finetune_model.fit(train_ds, train_labels, validation_data=(test_ds, test_labels), epochs=EPOCHS)

        # Evaluate the model
        finetune_model.evaluate(test_ds, test_labels)
        x = np.arange(0, 0.6, 0.0005)
        outputs = finetune_model(test_ds)
        logps = []
        logcs = []
        for i in x:
            logps.append(outputs.log_prob(i).numpy())
            logcs.append(outputs.log_cdf(i).numpy())
        logps = np.stack(logps)
        logcs = np.stack(logcs)
        predictions = x[np.exp(logps).argmax(axis=0)]

        q15s = np.argmax(np.exp(logcs) >= 0.15, axis=0)
        q85s = np.argmax(np.exp(logcs) >= 0.85, axis=0)
        lower_errors = np.abs(predictions - x[q15s])
        upper_errors = np.abs(x[q85s] - predictions)

        print(f'MAE = {np.mean(np.abs(test_labels - predictions))}')
        print(pearsonr(test_labels, predictions))

        test_results.append(predictions)
        err_l.append(lower_errors)
        err_h.append(upper_errors)

        # Save this version of the model as a checkpoint
        print(f'Saving model as {SAVE_AS}-split{run}')
        finetune_model.save_weights(f'{SAVE_AS}-split{run}.ckpt')

    for i in range(k):
        x = fracs[splits[i]]
        y = test_results[i]
        plt.errorbar(x, y, fmt='none', yerr=(err_l[i], err_h[i]), alpha=0.3)
        plt.scatter(x, y)

    flattened_results = flatten(test_results)
    maxval = np.max([fracs, flattened_results])
    plt.plot([0,maxval], [0,maxval], 'k--')
    plt.xlabel('Expected')
    plt.ylabel('Predicted')
    plt.savefig('result.png')
    plt.close()

    # Save the results so the plot can be recreated
    with open(f'{SAVE_AS}-results.pkl', 'wb') as fp:
        pickle.dump([test_results, err_l, err_h], fp)

    # Get overall stats
    actual = fracs[flatten(splits)]
    print(f'MAE = {np.mean(np.abs(flattened_results - actual))}')
    print(pearsonr(actual, flattened_results))

    err_l = flatten(err_l)
    err_h = flatten(err_h)

    sorted_idxs = np.argsort(actual)
    binned_results = np.array_split(flattened_results[sorted_idxs], 5)
    binned_fracs = np.array_split(actual[sorted_idxs], 5)

    # Calculate the median of the binned results
    x = []
    y = []
    xerr_l = []
    xerr_h = []
    for i in range(len(binned_results)):
        x_med = np.median(binned_fracs[i])
        x.append(x_med)
        y_med = np.median(binned_results[i])
        y.append(y_med)

        xerr_l.append(x_med - np.min(binned_fracs[i]))
        xerr_h.append(np.max(binned_fracs[i]) - x_med)

    plt.errorbar(actual, flattened_results, fmt='none', yerr=(err_l, err_h), alpha=0.2, color='gray')
    plt.plot(actual, flattened_results, '.', color='gray', alpha=0.3)
    plt.plot(x, y, 'or')
    plt.plot(x, y, 'r')
    plt.errorbar(x, y, fmt='none', xerr=(xerr_l, xerr_h), color='red', alpha=0.3)
    plt.xlabel('Expected')
    plt.ylabel('Predicted')
    plt.plot([0,maxval], [0,maxval], 'k--')

    plt.savefig('binned_results.png')

    print(f'MAE = {np.mean(np.abs(flattened_results - actual))}')
    print(pearsonr(actual, flattened_results))

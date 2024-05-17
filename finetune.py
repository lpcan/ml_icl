import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import skimage
from tensorflow import keras
import tensorflow as tf
import pickle

from supervised_model import ImageRegressor

MODEL_VERSION = '300-final'
EPOCHS = 100
FRACS_PATH = '/srv/scratch/mltidal/fracs_manual_300kpc.npy'
IMAGES_PATH = 'badmaskimgs_300kpc.npy'

def flatten(list):
    return np.array([i for row in list for i in row])

def prepare_data():
    # Create the finetuning dataset
    fracs = np.load(FRACS_PATH)[2]
    zeros = np.where(fracs == 0)[0]
    if len(zeros) > 0:
        fracs[zeros] = 0.00001

    not_nans = np.where(~np.isnan(fracs))[0]

    if IMAGES_PATH is None:
        # Prepare the images
        cutouts = h5py.File('/srv/scratch/z5214005/cutouts.hdf')
        cutouts_550 = h5py.File('/srv/scratch/z5214005/cutouts_550.hdf')
        BAD = 1
        NO_DATA = (1 << 8)
        BRIGHT_OBJECT = (1 << 9)
        images = []
        for idx in not_nans:
            cutout = np.array(cutouts[str(idx)]['HDU0']['DATA'])
            big_mask = (np.array(cutouts_550[str(idx)]['HDU1']['DATA']).astype(int) & (BAD | NO_DATA | BRIGHT_OBJECT))
            centre = big_mask.shape[0] // 2, big_mask.shape[1] // 2
            mask = big_mask[centre[0]-358:centre[0]+358,centre[1]-358:centre[1]+358]
            mask = skimage.transform.resize(mask, (224,224))
            img = skimage.transform.resize(cutout, (224,224))
            img = np.clip(img, a_min=0, a_max=10)
            img = np.arcsinh(img / 0.017359) * ~(mask > 0)
            img = np.expand_dims(img, -1)
            images.append(img)

        images = np.array(images)
    else:
        images = np.load(IMAGES_PATH)
        images = images[not_nans] 

    fracs = fracs[not_nans]

    return fracs, images

def create_model():
    # Create the model
    finetune_model = ImageRegressor((224,224,1))
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    mae = lambda y, p_y: tf.math.abs(y - p_y.mean())
    optimizer = keras.optimizers.Adam(learning_rate=1e-7)
    finetune_model.compile(optimizer=optimizer, loss=negloglik)
    finetune_model.load_weights(f'checkpoints/checkpoint-sup-{MODEL_VERSION}.ckpt').expect_partial()

    # Freeze the entire model other than the dense layers
    # for layer in finetune_model._flatten_layers():
    #     lst_of_sublayers = list(layer._flatten_layers())

    #     if len(lst_of_sublayers) == 1: # leaves of the model
    #         if layer.name in ['dense_1', 'dense_2']:
    #             layer.trainable = True
    #         else:
    #             layer.trainable = False

    return finetune_model
    # from lora import LoraLayer, RANK, ALPHA
    # import tensorflow_probability as tfp
    # # Load the original model
    # lora_model = ImageRegressor((224,224,1))
    # negloglik = lambda y, p_y: -p_y.log_prob(y)
    # steps = 2
    # boundaries = [5 * steps, 50 * steps]
    # values = [1e-7, 1e-4, 1e-5]
    # lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    # lora_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=negloglik)
    # lora_model.load_weights('checkpoint-sup-badmaskimages.ckpt').expect_partial()

    # lora_model(tf.random.uniform((1, 224, 224, 1)))

    # # Overwrite the original layer with the new LoRA layer
    # output_layers = lora_model.prob_output
    # dense_layer = output_layers.layers[0]

    # new_layer = LoraLayer(
    #     dense_layer, 
    #     rank=RANK,
    #     alpha=ALPHA,
    #     trainable=True
    # )

    # num_components = dense_layer.get_config()['units'] // 3
    # lora_model.prob_output = keras.Sequential([
    #             new_layer,
    #             tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
    #                 tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=tf.expand_dims(t[..., :num_components], -2)),
    #                                                     components_distribution=tfp.distributions.Beta(1 + tf.nn.softplus(tf.expand_dims(t[..., num_components:2*num_components], -2)),
    #                                                                                                 1 + tf.nn.softplus(tf.expand_dims(t[..., 2*num_components:],-2)))), 1))
    #         ])

    # # Do a forward pass to make sure we still have a valid chain of computation
    # lora_model(tf.random.uniform((1, 224, 224, 1)))

    # # Freeze the entire model other than the LoRA layers
    # for layer in lora_model._flatten_layers():
    #     lst_of_sublayers = list(layer._flatten_layers())

    #     if len(lst_of_sublayers) == 1: # leaves of the model
    #         if layer.name in ['lora_A', 'lora_B']:
    #             layer.trainable = True
    #         else:
    #             layer.trainable = False
    
    # return lora_model

def final_finetune():
    # Finetune on all of the data and save the final model
    fracs, images = prepare_data()

    finetune_model = create_model()

    finetune_model.fit(images, fracs, epochs=100)

    print(f'Saving model as {MODEL_VERSION}')
    finetune_model.save_weights(f'/srv/scratch/mltidal/finetuning_results/checkpoints/checkpoint-{MODEL_VERSION}.ckpt')

# Finetuning on separate splits of the data
if __name__=='__main__':
    fracs, images = prepare_data()

    # Get the splits for cross-validation
    k = 5
    rng = np.random.default_rng(seed=24)
    idxs = np.arange(len(fracs))
    rng.shuffle(idxs)
    splits = np.array_split(idxs, k)

    # sorted_idxs = np.argsort(fracs)
    # allocation = np.array([x % k for x in range(len(fracs))])
    # splits = []
    # for i in range(k):
    #     splits.append(sorted_idxs[allocation == i])
    # print(splits)

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

        # stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        # lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-8)

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
        print(f'Saving model as {MODEL_VERSION}-split{run}')
        finetune_model.save_weights(f'/srv/scratch/mltidal/finetuning_results/checkpoints/checkpoint-{MODEL_VERSION}-split{run}.ckpt')

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
    plt.savefig('asdf1.png')
    plt.close()

    # Save the results so the plot can be recreated
    with open(f'/srv/scratch/mltidal/finetuning_results/{MODEL_VERSION}.pkl', 'wb') as fp:
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

    plt.savefig('funs.png')

    print(f'MAE = {np.mean(np.abs(flattened_results - actual))}')
    print(pearsonr(actual, flattened_results))

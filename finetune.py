import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import skimage
from tensorflow import keras
import pickle

from supervised_model import ImageRegressor

MODEL_VERSION = 'deepv2-final'

# Create the finetuning dataset
fracs = np.load('/srv/scratch/mltidal/fracs_actual.npy')[2]
zeros = np.where(fracs == 0)[0]
if len(zeros) > 0:
    fracs[zeros] = 0.00001
    
not_nans = np.where(~np.isnan(fracs))[0]

# Prepare the images
cutouts = h5py.File('/srv/scratch/z5214005/hsc_icl/cutouts.hdf')
images = []
for idx in not_nans:
    cutout = np.array(cutouts[str(idx)]['HDU0']['DATA'])
    img = skimage.transform.resize(cutout, (224,224))
    img = np.clip(img, a_min=0, a_max=10)
    img = np.arcsinh(img / 0.017359)
    img = np.expand_dims(img, -1)
    images.append(img)

images = np.array(images) 

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
for i in range(k):
    # Create the datasets for this split
    test_set = splits[i]
    train_set = splits.copy()
    train_set.pop(i)
    train_set = np.concatenate(train_set)

    test_ds = images[test_set]
    test_labels = fracs[test_set]
    train_ds = images[train_set]
    train_labels = fracs[train_set]

    # Create the model
    finetune_model = ImageRegressor((224,224,1))
    negloglik = lambda y, p_y: -p_y.log_prob(y)
    finetune_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-7), loss=negloglik)
    finetune_model.load_weights(f'checkpoints/checkpoint-sup-{MODEL_VERSION}.ckpt').expect_partial()

    # Freeze the entire model other than the dense layers
    for layer in finetune_model._flatten_layers():
        lst_of_sublayers = list(layer._flatten_layers())

        if len(lst_of_sublayers) == 1: # leaves of the model
            if layer.name in ['dense_1', 'dense_2']:
                layer.trainable = True
            else:
                layer.trainable = False

    # Check that the model is functioning as we expect before the extra training of the lora layers
    finetune_model.evaluate(test_ds, test_labels)
    predictions = finetune_model.predict(test_ds)

    print(f'MAE = {np.mean(np.abs(test_labels - predictions))}')

    # stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    # lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-8)

    # Train the model
    finetune_model.fit(train_ds, train_labels, validation_data=(test_ds, test_labels), epochs=100)

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

    # plt.errorbar(test_labels, predictions, fmt='none', yerr=(lower_errors, upper_errors), alpha=0.3)
    # plt.scatter(test_labels, predictions)
    # # predictions = lora_model.predict(train_ds)
    # # plt.scatter(train_labels, predictions)
    # plt.plot([0,0.35], [0,0.35], 'k--')
    # plt.xlabel('Expected')
    # plt.ylabel('Predicted')
    # plt.savefig('asdf.png')
    # plt.close()

for i in range(k):
    x = fracs[splits[i]]
    y = test_results[i]
    plt.errorbar(x, y, fmt='none', yerr=(err_l[i], err_h[i]), alpha=0.3)
    plt.scatter(x, y)

flattened_results = [i for row in test_results for i in row]
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
actual = fracs[splits].flatten()
print(f'MAE = {np.mean(np.abs(flattened_results - actual))}')
print(pearsonr(actual, flattened_results))

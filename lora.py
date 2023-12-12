"""
Implementing LoRA finetuning method. Code mostly from 
https://keras.io/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/,
paper https://arxiv.org/pdf/2106.09685.pdf
"""

import h5py
import math
import matplotlib.pyplot as plt
import numpy as np
import skimage
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from supervised_model import ImageRegressor

RANK = 4
ALPHA = 32.0

# Creating a LoRA layer
class LoraLayer(keras.layers.Layer):
    def __init__(
            self, 
            original_layer,
            rank=8,
            alpha=32,
            trainable=False,
            **kwargs
    ):
        # We want to keep the name of this layer the same as the original dense 
        # layer
        original_layer_config = original_layer.get_config()
        name = original_layer_config['name']

        kwargs.pop('name', None)

        super().__init__(name=name, trainable=trainable, **kwargs)

        self.rank = rank
        self.alpha = alpha

        self._scale = alpha / rank

        # Layers

        # Original dense layer
        self.original_layer = original_layer
        # No matter whether we are training the model or are in inference mode, 
        # this layer should be frozen
        self.original_layer.trainable = False

        # LoRA dense layers
        self.A = keras.layers.Dense(
            units=rank, 
            use_bias=False,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode='fan_in', distribution='uniform'
            ),
            trainable = trainable,
            name=f'lora_A'
        )

        # This is a bit different to the example code, since this should match 
        # the original layer, which is an Einsum Dense layer in GPT-2
        self.B = keras.layers.Dense(
            units = original_layer_config['units'],
            use_bias = original_layer_config['use_bias'],
            kernel_initializer = 'zeros',
            trainable = trainable,
            name=f'lora_B'
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        if self.trainable:
            # If we are fine-tuning the model, we will add LoRA layers' output
            # to the original layer's output
            lora_output = self.B(self.A(inputs)) * self._scale
            return original_output + lora_output
        
        # If we are in inference mode, we "merge" the LoRA layers' weights into
        # the original layer's weights
        return original_output
    
# Load the original model
lora_model = ImageRegressor((224,224,1))
negloglik = lambda y, p_y: -p_y.log_prob(y)
steps = 2
boundaries = [5 * steps, 50 * steps]
values = [1e-7, 1e-4, 1e-5]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
lora_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=negloglik)
lora_model.load_weights('checkpoint-sup-deepcont1.ckpt').expect_partial()

lora_model(tf.random.uniform((1, 224, 224, 1)))

# Overwrite the original layer with the new LoRA layer
output_layers = lora_model.prob_output
dense_layer = output_layers.layers[0]

new_layer = LoraLayer(
    dense_layer, 
    rank=RANK,
    alpha=ALPHA,
    trainable=True
)

num_components = dense_layer.get_config()['units'] // 3
lora_model.prob_output = keras.Sequential([
            new_layer,
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
                tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=tf.expand_dims(t[..., :num_components], -2)),
                                                    components_distribution=tfp.distributions.Beta(1 + tf.nn.softplus(tf.expand_dims(t[..., num_components:2*num_components], -2)),
                                                                                                   1 + tf.nn.softplus(tf.expand_dims(t[..., 2*num_components:],-2)))), 1))
        ])

# Do a forward pass to make sure we still have a valid chain of computation
lora_model(tf.random.uniform((1, 224, 224, 1)))

# Freeze the entire model other than the LoRA layers
for layer in lora_model._flatten_layers():
    lst_of_sublayers = list(layer._flatten_layers())

    if len(lst_of_sublayers) == 1: # leaves of the model
        if layer.name in ['lora_A', 'lora_B']:
            layer.trainable = True
        else:
            layer.trainable = False

print(lora_model.summary())

# Create the finetuning dataset
fracs = np.load('/srv/scratch/mltidal/fracs_resized.npy')[2]
zeros = np.where(fracs == 0)[0]
if len(zeros) > 0:
    fracs[zeros] = 0.00001

# Sample a very small dataset of 50, evenly distributed among the fracs
sorted_idx = np.argsort(fracs)
even_spacing = np.round(np.linspace(0, len(sorted_idx)-1, 50)).astype(int) 
select_idxs = sorted_idx[even_spacing]
select_mask = np.zeros(len(fracs), dtype=bool)
select_mask[select_idxs] = True
test_idxs = np.where(select_mask == 0)[0]

# Shuffle both sets of ids so they're not given to the model in order
np.random.shuffle(select_idxs)
np.random.shuffle(test_idxs)

# Create the two datasets
cutouts = h5py.File('/srv/scratch/z5214005/hsc_icl/cutouts.hdf')

train_ds = []
train_labels = fracs[select_idxs]
for idx in select_idxs:
    cutout = np.array(cutouts[str(idx)]['HDU0']['DATA'])
    img = skimage.transform.resize(cutout, (224,224))
    img = np.clip(img, a_min=0, a_max=10)
    img = np.arcsinh(img / 0.017359)
    img = np.expand_dims(img, -1)
    train_ds.append(img)
train_ds = np.array(train_ds)

test_ds = []
test_labels = fracs[test_idxs]
for idx in test_idxs:
    cutout = np.array(cutouts[str(idx)]['HDU0']['DATA'])
    img = skimage.transform.resize(cutout, (224,224))
    img = np.clip(img, a_min=0, a_max=10)
    img = np.arcsinh(img / 0.017359)
    img = np.expand_dims(img, -1)
    test_ds.append(img)
test_ds = np.array(test_ds)

# Check that the model is functioning as we expect before the extra training of the lora layers
lora_model.evaluate(test_ds, test_labels)
predictions = lora_model.predict(test_ds)

print(f'MAE = {np.mean(np.abs(test_labels - predictions))}')

plt.scatter(test_labels, predictions)
plt.plot([0,0.35], [0,0.35], 'k--')
plt.xlabel('Expected')
plt.ylabel('Predicted')
plt.savefig('asdf1.png')
plt.close()

# Train the model
lora_model.fit(train_ds, train_labels, validation_data=(test_ds, test_labels), epochs=50)

# Merge weights and test the model
lora_layer = lora_model.prob_output.layers[0]

A_weights = lora_layer.A.kernel
B_weights = lora_layer.B.kernel
increment_weights = tf.einsum('ab,bd->ad', A_weights, B_weights) * (ALPHA / RANK)
lora_layer.original_layer.kernel.assign_add(increment_weights)

# Evaluate the model
lora_model.evaluate(test_ds, test_labels)
x = np.arange(0, 0.6, 0.0005)
outputs = lora_model(test_ds)
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

plt.errorbar(test_labels, predictions, fmt='none', yerr=(lower_errors, upper_errors), alpha=0.3)
plt.scatter(test_labels, predictions)
# predictions = lora_model.predict(train_ds)
# plt.scatter(train_labels, predictions)
plt.plot([0,0.35], [0,0.35], 'k--')
plt.xlabel('Expected')
plt.ylabel('Predicted')
plt.savefig('asdf.png')
plt.close()

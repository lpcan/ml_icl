"""
Train a self-supervised NNCLR model
"""
from model import NNCLR

import pickle
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

# Hyperparameters
AUTOTUNE = tf.data.AUTOTUNE
shuffle_buffer = 5000 # buffer size for dataset.shuffle

temperature = 0.1
queue_size = 1000
input_shape = (224, 224, 1)

num_epochs = 50
num_images = 2377
batch_size = 16

stddev = 0.017359 # Calculated elsewhere from first 1000 cutouts

# Preprocessing function
def preprocess(data):
    image = data['image']
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image

# Prepare the dataset
initial_dataset = (tfds.load('hsc_icl', split='train', shuffle_files=True)
           .shuffle(buffer_size=shuffle_buffer)
           .batch(batch_size)
)

# Preprocess the dataset
dataset = initial_dataset.map(preprocess)

# Instantiate the model
model = NNCLR(input_shape=input_shape, 
              temperature=temperature, 
              queue_size=queue_size)
model.compile(
    contrastive_optimizer=keras.optimizers.Adam()
)

# Create a checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint.ckpt',
    save_weights_only=True,
    verbose=1
)

# Train the model
train_history = model.fit(dataset, epochs=num_epochs, callbacks=[cp_callback])

# Save the history
with open('history.pkl', 'wb') as f:
    pickle.dump(train_history, f)

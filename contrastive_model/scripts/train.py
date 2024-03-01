"""
Train a self-supervised NNCLR model
"""
from model import NNCLR
from calc_spearman import eval_model

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import wandb
from wandb.keras import WandbMetricsLogger

# Hyperparameters
AUTOTUNE = tf.data.AUTOTUNE
shuffle_buffer = 10000 # buffer size for dataset.shuffle

temperature = 0.1
queue_size = 1000
input_shape = (224, 224, 1)

num_epochs = 50
batch_size = 32

stddev = 0.017359 # Calculated elsewhere from first 1000 cutouts

# Preprocessing function
def preprocess(data):
    image = data['image']
    image = tf.clip_by_value(image, 0.0, 10.0)
    image = tf.math.asinh(image / stddev)
    return image

wandb.init(
    project='ml-icl-selfsup',
    config={
        'epochs': num_epochs, 
        'batch_size': batch_size,
        'optimizer': 'adam',
    },
    id='32xl0pl8',
    resume='must'
)

# Prepare the dataset
initial_dataset = (tfds.load('all_unique', split='train', data_dir='/srv/scratch/mltidal/tensorflow_datasets', shuffle_files=True)
           .shuffle(buffer_size=shuffle_buffer)
           .batch(batch_size, drop_remainder=True)
)

# Preprocess the dataset
dataset = initial_dataset.map(preprocess)

# Instantiate the model
model = NNCLR(input_shape=input_shape, 
              temperature=temperature, 
              queue_size=queue_size)

model.compile(
    contrastive_optimizer=keras.optimizers.Adam(learning_rate=1e-3)
)
model.load_weights('checkpoint-alldata50nobatchnorm-final.ckpt')
lr_callback = keras.callbacks.ReduceLROnPlateau(monitor='c_loss', factor=0.1, patience=10, verbose=1, min_lr=1e-6)
stop_callback = keras.callbacks.EarlyStopping(monitor='c_loss', patience=15)

# Create a checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoint-alldata50nobatchnorm.ckpt',
    save_weights_only=True,
    verbose=1
)

# Train the model
train_history = model.fit(dataset, epochs=num_epochs, callbacks=[cp_callback, lr_callback, stop_callback, WandbMetricsLogger()], verbose=2)

print(f'Saving model as alldata50nobatchnorm')
model.save_weights(f'checkpoint-alldata50nobatchnorm-final.ckpt')

# Save all the checkpoint files to wandb
wandb.save(f'checkpoint-alldata50nobatchnorm.ckpt*')
wandb.save(f'checkpoint-alldata50nobatchnorm-final.ckpt*')

print(f'Spearman coefficient for checkpointed model = {eval_model("checkpoint-alldata50nobatchnorm-final.ckpt")}')

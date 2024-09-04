"""
Defines the model architecture
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import resnet_cifar10_v2

from augmentations import augmenter

N = 6
DEPTH = N*9+2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1
NUM_COMPONENTS = 16

def encoder(input_shape):
    """
    Define a ResNet-50 encoder architecture with global average pooling
    """
    inputs = layers.Input(input_shape, name='encoder_input')
    x = resnet_cifar10_v2.stem(inputs)
    x = resnet_cifar10_v2.learner(x, NUM_BLOCKS)
    x = layers.ReLU()(x)
    outputs = layers.GlobalAveragePooling2D(name='backbone_pool')(x)

    return keras.Model(inputs, outputs, name='encoder')

class ImageRegressor(keras.Model):
    """
    Define the architecture and behaviour of the full model
        - Augmenter (normalise and randomly perturb the image)
        - Encoder (ResNet-50 encoder outputting 256-dimensional encoding)
        - Regressor (Hidden layer with 2048 neurons)
        - Prob_output (Dense layer with NUM_COMPONENTS*3 neurons, parameterising
                       mixture of beta distributions as probability distribution)
    """
    def __init__(self, input_shape, mean=0.948, std=1.108):
        super().__init__()
        self.augmenter = augmenter(input_shape, mean=mean, std=std)
        self.encoder = encoder(input_shape)
        self.regressor = keras.Sequential([
            layers.Input((256), name='regressor'),
            layers.Dense(2048, activation='relu'),
        ])
        self.prob_output = keras.Sequential([
            layers.Dense(units=NUM_COMPONENTS*3),
            tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
                tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=tf.expand_dims(t[..., :NUM_COMPONENTS], -2)),
                                                    components_distribution=tfp.distributions.Beta(1 + tf.nn.softplus(tf.expand_dims(t[..., NUM_COMPONENTS:2*NUM_COMPONENTS], -2)),
                                                                                                   1 + tf.nn.softplus(tf.expand_dims(t[..., 2*NUM_COMPONENTS:],-2)))), 1))
        ])

    def call(self, inputs):
        x = self.augmenter(inputs)
        x = self.encoder(x)
        x = self.regressor(x)
        outputs = self.prob_output(x)
        return outputs
    
def load_model(model_name=None, model_path_prefix='checkpoints/', lr=1e-4):
    model = ImageRegressor((224,224,1))

    negloglik = lambda y, p_y: -p_y.log_prob(y) # Negative log likehood loss

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=negloglik)

    if model_name is not None: 
        model.load_weights(f'{model_path_prefix}{model_name}.ckpt').expect_partial()

    return model

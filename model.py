"""
"""
################################################################################
# Imports
from parameters import *


################################################################################
# keeping model architecture simple for now with fully connected (dense) layers
# Fully Connected
def build_model(num_categories):
    model = tf.keras.Sequential()

    # layer 1
    model.add(tf.keras.layers.Dense(
        units=512,
        input_shape=(MAX_WORDS, ),  # (batch, MAX_WORDS)
        activation=tf.keras.activations.relu
    ))

    # layer 2
    model.add(tf.keras.layers.Dense(
        units=num_categories,
        activation=tf.keras.activations.softmax
    ))

    return model

""" Built-in utilities to process inputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import pi

import tensorflow as tf
import keras as k
import keras.backend as K

# interface for some keras features to be acessible across sciann.
from keras.backend import is_tensor
from keras.backend import floatx
from keras.backend import set_floatx
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras.utils import plot_model
from keras.initializers import random_uniform as default_bias_initializer
from keras.initializers import glorot_normal as default_kernel_initializer


def get_activation(activation):
    """ Evaluates the activation function from a string or list of string inputs.

    # Arguments
        activation: A string pointing to the function name.

    # Returns:
        A function handle.
    """

    if isinstance(activation, list):
        return [get_activation(act) for act in activation]

    elif isinstance(activation, str):
        if hasattr(k.activations, activation):
            return getattr(k.activations, activation)
        elif hasattr(k.backend, activation):
            return getattr(k.backend, activation)
        elif hasattr(tf.math, activation):
            return getattr(tf.math, activation)
        else:
            raise ValueError(
                'Not a valid function name: ' + activation +
                ' - Please provide a valid activation '  
                'function name from Keras or Tensorflow. '
            )

    elif callable(activation):
        return activation

    else:
        raise TypeError(
            'Please provide a valid input: ' + type(activation) +
            ' - Expecting a function name or function handle. '
        )

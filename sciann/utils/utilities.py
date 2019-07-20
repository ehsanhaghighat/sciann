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

from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras.utils import plot_model

import sciann


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


def is_functional(f):
    """ Checks whether `f` is a functional object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if functional.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.Functional):
        return True

    else:
        return False


def validate_functional(f):
    """ if `f` is not a functional object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if functional, False otherwise.

    # Raises
        ValueError: if the object is not a Functional object.

    """
    if isinstance(f, sciann.Functional):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `functional` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors.'
        )


def is_pdemodel(f):
    """ Checks whether `f` is a `SciModel` object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if SciModel.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.SciModel):
        return True

    else:
        return False


def validate_pdemodel(f):
    """ if `f` is not a SciModel object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if SciModel, False otherwise.

    # Raises
        ValueError: if the object is not a SciModel object.

    """
    if isinstance(f, sciann.SciModel):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `SciModel` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors '
            'or layers. '
        )


def is_variable(f):
    """ Checks whether `f` is a `Variable` object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Variable.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.Variable):
        return True

    else:
        return False


def validate_variable(f):
    """ if `f` is not a Variable object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Variable, False otherwise.

    # Raises
        ValueError: if the object is not a Variable object.

    """
    if isinstance(f, sciann.Variable):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `Variable` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors '
            'or layers. '
        )

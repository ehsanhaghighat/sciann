from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import *

from keras.layers import Dense, TimeDistributed
from keras.activations import linear


class RNNField(TimeDistributed):
    """ Configures the `LSTMField` class for the model outputs.

    # Arguments
        units: Positive integer.
            Dimension of the output of the network.
        name: String.
            Assigns a layer name for the output.
        activation: Callable.
            A callable object for the activation.
        kernel_initializer: Initializer for the kernel.
            Defaulted to a normal distribution.
        bias_initializer: Initializer for the bias.
            Defaulted to a normal distribution.
        trainable: Boolean to activate parameters of the network.
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').

    # Raises

    """
    def __init__(self, units=1,
                 name=None,
                 activation=linear,
                 kernel_initializer=default_kernel_initializer(),
                 bias_initializer=default_bias_initializer(),
                 trainable=True,
                 dtype=None,):
        if not dtype:
            dtype = floatx()
        elif not dtype == floatx():
            set_floatx(dtype)

        assert isinstance(name, str), \
            "Please provide a string for field name. "
        assert callable(activation), \
            "Please provide a function handle for the activation. "

        super(RNNField, self).__init__(
            Dense(
                units=units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                use_bias=True,
                trainable=trainable,
                name=name,
                dtype=dtype,
            )
        )

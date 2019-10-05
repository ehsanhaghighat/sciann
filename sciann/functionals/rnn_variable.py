from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import *

from keras.layers import InputLayer
from .rnn_functional import RNNFunctional


class RNNVariable(RNNFunctional):
    """ Configures the `LSTMVariable` object for the network's input.

    # Arguments
        units: Int.
            number of time units in an recurrent architecture.
            A minimum of 2 is needed.
        name: String.
            Required as derivatives work only with layer names.
        tensor: Tensorflow `Tensor`.
            Can be pass as the input path.
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').

    # Raises

    """
    def __init__(self,
                 units,
                 name=None,
                 tensor=None,
                 dtype=None):

        if not dtype:
            dtype = floatx()
        elif not dtype == floatx():
            set_floatx(dtype)

        assert isinstance(units, int) and units>=2, \
            'RNN needs a minimum of 2 time units. '

        layer = InputLayer(
            batch_input_shape=(None, units, 1),
            input_tensor=tensor,
            name=name,
            dtype=dtype
        )

        super(RNNVariable, self).__init__(
            layers=to_list(layer),
            inputs=to_list(layer.input),
            outputs=to_list(layer.output),
        )

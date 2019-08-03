from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import *

from keras.layers import InputLayer
from .functional import Functional


class Variable(Functional):
    """ Configures the `Variable` object for the network's input.

    # Arguments
        name: String.
            Required as derivatives work only with layer names.
        tensor: Tensorflow `Tensor`.
            Can be pass as the input path.
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').

    # Raises

    """
    def __init__(self,
                 name=None,
                 tensor=None,
                 dtype=None):

        if not dtype:
            dtype = floatx()
        elif not dtype == floatx():
            set_floatx(dtype)

        layer = InputLayer(
            batch_input_shape=(None, 1),
            input_tensor=tensor,
            name=name,
            dtype=dtype
        )

        super(Variable, self).__init__(
            layers=to_list(layer),
            inputs=to_list(layer.input),
            outputs=to_list(layer.output),
        )

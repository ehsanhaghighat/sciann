from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import *

from keras.layers import InputLayer

from .functional import Functional
from .variable import Variable


class RadialBasis(Functional):
    """ Radial Basis functional class.
    """
    def __init__(self, vars, rb_vars, radii):
        vars = to_list(vars)
        if not all([isinstance(x, Variable) for x in vars]):
            raise TypeError

        rb_vars = to_list(rb_vars)
        if not all([isinstance(x, RadialBasisBase) for x in rb_vars]):
            try:
                for i, rbv in enumerate(rb_vars):
                    rb_vars[i] = RadialBasisBase(rbv)
            except (ValueError, TypeError):
                raise ValueError('Expected `str` or `RadialBasisBase` as rb_vars. ')

        if len(vars) != len(rb_vars):
            raise ValueError

        if radii <= 0.0:
            raise ValueError('Expecting a positive value for `radii`. ')

        inputs, layers = [], []
        for v in vars:
            inputs += v.outputs
            layers += v.layers
        for v in rb_vars:
            inputs += v.outputs
            layers += v.layers

        lmbd = [
            Lambda(lambda x: K.exp(-(x[1] - x[0])**2/radii**2))
            for i in range(len(vars))
        ]

        outputs = []
        for i, l in enumerate(lmbd):
            l.name = "{}/".format('RadialBasis') + l.name.split("_")[-1]
            assert len(vars[i].outputs) == 1
            assert len(rb_vars[i].outputs) == 1
            layers.append(l)
            outputs.append(l(vars[i].outputs + rb_vars[i].outputs))

        super(RadialBasis, self).__init__(
            layers=layers,
            inputs=inputs,
            outputs=outputs
        )


class RadialBasisBase(Functional):
    """ Configures the `RadialBasisBase` object for the network's input.

    # Arguments
        name: String.
            Required as derivatives work only with layer names.
        units (Int): Number of nodes to the network.
            Minimum number is 1.
        tensor: Tensorflow `Tensor`.
            Can be pass as the input path.
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').

    # Raises
        ValueError: Provide `units > 0`.
    """
    def __init__(self,
                 name=None,
                 units=1,
                 tensor=None,
                 dtype=None):

        if not dtype:
            dtype = K.floatx()
        elif not dtype == K.floatx():
            K.set_floatx(dtype)

        if units < 1:
            raise ValueError(
                'Expected at least one unit size - was provided `units`={:d}'.format(units)
            )

        layer = InputLayer(
            batch_input_shape=(None, units),
            input_tensor=tensor,
            name=name,
            dtype=dtype
        )

        super(RadialBasisBase, self).__init__(
            layers=to_list(layer),
            inputs=to_list(layer.input),
            outputs=to_list(layer.output),
        )
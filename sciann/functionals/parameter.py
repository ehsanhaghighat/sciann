""" Functional class for SciANN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import *

from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import InputSpec

from .functional import Functional
from .variable import Variable


class Parameter(Functional):
    def __init__(self,
                 val=1.0,
                 inputs=None,
                 name=None,
                 non_neg=None):

        inputs = to_list(inputs)
        if not all([isinstance(x, Variable) for x in inputs]):
            raise TypeError

        inputs_tensors, layers = [], []
        for v in inputs:
            inputs_tensors += v.outputs
            layers += v.layers

        if non_neg is None:
            non_neg = True
        layers.append(
            ParameterBase(val=val, non_neg=non_neg, name=name)
        )

        lay = Concatenate()
        lay.name = "conct_" + lay.name.split("_")[-1]
        lay_input = lay(inputs_tensors)
        outputs = layers[-1](lay_input)

        super(Parameter, self).__init__(
            inputs=to_list(inputs_tensors),
            outputs=to_list(outputs),
            layers=to_list(layers)
        )


class ParameterBase(Dense):
    """ Parameter class to be used for parameter inversion.
        Inherited from Dense layer.

    # Arguments
        val (float): Initial value for the parameter.
        non_neg (boolean): True (default) if only non-negative values are expected.
        **kwargs: keras.layer.Dense accepted arguments.
    """
    def __init__(self, val=1.0, non_neg=True, **kwargs):
        super(ParameterBase, self).__init__(
            units=1,
            use_bias=True,
            kernel_initializer='zeros',
            bias_initializer=k.initializers.constant(val),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=k.constraints.non_neg() if non_neg else None,
            **kwargs,
        )

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=False)

        self.bias = self.add_weight(shape=(self.units,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True



def is_variable(f):
    """ Checks whether `f` is a `Variable` object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Variable.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, Variable):
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
    if isinstance(f, Variable):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `Variable` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors '
            'or layers. '
        )

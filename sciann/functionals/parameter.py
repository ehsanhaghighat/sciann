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
from ..constraints import MinMax


class Parameter(Functional):
    """ Parameter functional to be used for parameter inversion.
        Inherited from Dense layer.

    # Arguments
        val: float.
            Initial value for the parameter.
        min_max: [MIN, MAX].
            A range to constrain the value of parameter.
            This constraint will overwrite non_neg constraint if both are chosen.
        inputs: Variables.
            List of `Variable`s to the parameters.
        name: str.
            A name for the Parameter layer.
        non_neg: boolean.
            True (default) if only non-negative values are expected.
        **kwargs: keras.layer.Dense accepted arguments.
        
    """

    def __init__(self,
                 val=1.0,
                 min_max=None,
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
            ParameterBase(val=val, min_max=min_max, non_neg=non_neg, name=name)
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
    """ Base Parameter class to be used for parameter inversion.
        Inherited from Dense layer.

    # Arguments
        val (float): Initial value for the parameter.
        min_max ([MIN, MAX]): A range to constrain the value of parameter.
            This constraint will overwrite non_neg constraint if both are chosen.
        non_neg (boolean): True (default) if only non-negative values are expected.
        **kwargs: keras.layer.Dense accepted arguments.
    """
    def __init__(self, val=1.0, min_max=None, non_neg=True, **kwargs):
        cst = None
        if min_max is not None:
            cst = MinMax(min_value=min_max[0], max_value=min_max[1])
            val = (min_max[0] + min_max[1])/2.0
        elif non_neg:
            cst = k.constraints.non_neg()
        super(ParameterBase, self).__init__(
            units=1,
            use_bias=True,
            kernel_initializer='zeros',
            bias_initializer=k.initializers.constant(val),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=cst,
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


from keras.constraints import Constraint
class MinMax(Constraint):
    """MinMax weight constraint.

    Constrains the weights incident to each hidden unit
    to have values between a lower bound and an upper bound.

    # Arguments
        min_value: the minimum norm for the incoming weights.
        max_value: the maximum norm for the incoming weights.
    """

    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        d = self.max_value - self.min_value
        w = K.square(K.relu(w - self.max_value)) + K.square(K.relu(self.min_value - w))
        w = K.square(K.clip(w - self.max_value, 0.0, 100*d)) + \
            K.square(K.clip(self.min_value - w, 0.0, 100*d))
        return w / d**2

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

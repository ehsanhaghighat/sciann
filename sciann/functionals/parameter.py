""" Functional class for SciANN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras as k
import keras.backend as K

from keras.layers import Dense
from keras.layers import Concatenate
from keras.layers import InputSpec
from keras.constraints import MinMaxNorm

from ..utils import to_list
from ..utils import default_constant_initializer

from .functional import Functional
from .variable import Variable


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
        if not all([isinstance(x, (Variable, Functional)) for x in inputs]):
            raise TypeError

        input_tensors, layers = [], []
        for v in inputs:
            input_tensors += v.outputs
            layers += v.layers

        if non_neg is None:
            non_neg = True
        layers.append(
            ParameterBase(val=val, min_max=min_max, non_neg=non_neg, name=name)
        )

        if len(input_tensors) > 1:
            lay = Concatenate()
            lay.name = "conct_" + lay.name.split("_")[-1]
            lay_input = lay(input_tensors)
        else:
            lay_input = input_tensors[0]
        outputs = layers[-1](lay_input)

        super(Parameter, self).__init__(
            inputs=to_list(input_tensors),
            outputs=to_list(outputs),
            layers=to_list(layers)
        )

    @classmethod
    def get_class(cls):
        return Functional


class ParameterBase(Dense):
    """ Base Parameter class to be used for parameter inversion.
        Inherited from Dense layer.

    # Arguments
        val (float): Initial value for the parameter.
        min_max ([MIN, MAX, [Penalty]]): A range to constrain the value of parameter.
            This constraint will overwrite non_neg constraint if both are chosen.
        non_neg (boolean): True (default) if only non-negative values are expected.
        **kwargs: keras.layer.Dense accepted arguments.
    """
    def __init__(self, val=None, min_max=None, non_neg=True, **kwargs):
        cst = None
        if min_max is not None:
            cst = MinMax(min_value=min_max[0], max_value=min_max[1], penalty=1.0 if len(min_max)==2 else min_max[2])
            val = (min_max[0] + min_max[1])/2.0 if val is None else val
        elif non_neg:
            cst = k.constraints.non_neg()
            
        # Default value for initial values. 
        val = 1.0 if val is None else val
        
        super(ParameterBase, self).__init__(
            units=1,
            use_bias=True,
            kernel_initializer='zeros',
            bias_initializer=default_constant_initializer(val),
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


from keras.constraints import Constraint
class MinMax(Constraint):
    """MinMax weight constraint.

    Constrains the weights incident to each hidden unit
    to have values between a lower bound and an upper bound.

    # Arguments
        min_value: the minimum norm for the incoming weights.
        max_value: the maximum norm for the incoming weights.
    """

    def __init__(self, min_value=0.0, max_value=1.0, penalty=1.0):
        self.min_value = min_value
        self.max_value = max_value
        self.penalty = penalty

    def __call__(self, w):
        return MinMaxNorm(self.min_value, self.max_value, self.penalty)(w)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}

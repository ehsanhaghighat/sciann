""" Functional class for SciANN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.keras as k

import tensorflow.python.keras.backend as K
graph_unique_name = K.get_graph().unique_name

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import InputSpec
from tensorflow.python.keras.constraints import MinMaxNorm
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from ..utils import to_list
from ..utils import default_constant_initializer
from ..utils import default_regularizer

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
            lay = Concatenate(name=graph_unique_name('conct'))
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


@keras_export('keras.layers.ParameterBase')
class ParameterBase(k.layers.Layer):
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
        super(ParameterBase, self).__init__(**kwargs)

        cst = None
        if min_max is not None:
            cst = MinMax(min_value=min_max[0], max_value=min_max[1], penalty=1.0 if len(min_max) == 2 else min_max[2])
            val = (min_max[0] + min_max[1]) / 2.0 if val is None else val
        elif non_neg:
            cst = k.constraints.non_neg()

        # Default value for initial values.
        val = 1.0 if val is None else val

        self.param_initializer = default_constant_initializer(val)
        self.param_regularizer = None
        self.param_constraint = None
        self.shared_axes = None
        self.param_constraint = cst

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = list(input_shape[1:])

        self.param = self.add_weight(
            shape=[1,],
            initializer=self.param_initializer,
            name='param',
            regularizer=self.param_regularizer,
            constraint=self.param_constraint
        )
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        return self.param #+ inputs*0.0

    def get_config(self):
        config = {
            'param_initializer': initializers.serialize(self.param_initializer),
            'param_regularizer': regularizers.serialize(self.param_regularizer),
            'param_constraint': constraints.serialize(self.param_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(ParameterBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.param.shape


from tensorflow.python.keras.constraints import Constraint

@keras_export('keras.constraints.MinMax')
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

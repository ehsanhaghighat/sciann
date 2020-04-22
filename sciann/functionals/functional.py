""" Functional class for SciANN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K

from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Concatenate
from keras.models import Model

from ..utils import to_list, unpack_singleton, is_same_tensor, unique_tensors
from ..utils import default_bias_initializer, default_kernel_initializer, default_constant_initializer
from ..utils import validations, get_activation, getitem
from ..utils import floatx, set_floatx
from ..utils import math

from .field import Field


class Functional(object):
    """ Configures the Functional object (Neural Network).

    # Arguments
        fields: String or Field.
            [Sub-]Network outputs.
            It can be of type `String` - Associated fields will be created internally.
            It can be of type `Field` or `Functional`
        variables: Variable.
            [Sub-]Network inputs.
            It can be of type `Variable` or other Functional objects.
        hidden_layers: A list indicating neurons in the hidden layers.
            e.g. [10, 100, 20] is a for hidden layers with 10, 100, 20, respectively.
        activation: defaulted to "tanh".
            Activation function for the hidden layers.
            Last layer will have a linear output.
        output_activation: defaulted to "linear".
            Activation function to be applied to the network output.
        kernel_initializer: Initializer of the `Kernel`, from `k.initializers`.
        bias_initializer: Initializer of the `Bias`, from `k.initializers`.
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').
            Note: Only network inputs should be set.
        trainable: Boolean.
            False if network is not trainable, True otherwise.
            Default value is True.

    # Raises
        ValueError:
        TypeError:
    """
    def __init__(self,
                 fields=None,
                 variables=None,
                 hidden_layers=None,
                 activation="tanh",
                 output_activation="linear",
                 kernel_initializer=default_kernel_initializer(),
                 bias_initializer=default_bias_initializer(),
                 dtype=None,
                 trainable=True,
                 **kwargs):
        # check data-type.
        if dtype is None:
            dtype = K.floatx()
        elif not K.floatx() == dtype:
            K.set_floatx(dtype)
        # check for copy constructor.
        if all([x in kwargs for x in ('inputs', 'outputs', 'layers')]):
            self._inputs = kwargs['inputs'].copy()
            self._outputs = kwargs['outputs'].copy()
            self._layers = kwargs['layers'].copy()
            self._set_model()
            return
        # prepare initializers. 
        if isinstance(kernel_initializer, (float, int)):
            kernel_initializer = default_constant_initializer(kernel_initializer)
        if isinstance(bias_initializer, (float, int)):
            bias_initializer = default_constant_initializer(bias_initializer)
        # prepares fields.
        fields = to_list(fields)
        if all([isinstance(fld, str) for fld in fields]):
            output_fileds = [
                Field(
                    name=fld,
                    dtype=dtype,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    trainable=trainable,
                )
                for fld in fields
            ]
        elif all([validations.is_field(fld) for fld in fields]):
            output_fileds = fields
        else:
            raise TypeError(
                'Please provide a "list" of field names of'
                + ' type "String" or "Field" objects.'
            )
        # prepare inputs/outputs/layers.
        inputs = []
        layers = []
        variables = to_list(variables)
        if all([isinstance(var, Functional) for var in variables]):
            for var in variables:
                inputs += var.outputs
            # for var in variables:
            #     for lay in var.layers:
            #         layers.append(lay)
        else:
            raise TypeError(
                "Input error: Please provide a `list` of `Functional`s. \n"
                "Provided - {}".format(variables)
            )
        # prepare hidden layers.
        if hidden_layers is None:
            hidden_layers = []
        else:
            hidden_layers = to_list(hidden_layers)
        # Check and convert activation functions to proper format.
        assert not isinstance(activation, list), \
            'Expected an activation function name not a "list". '
        afunc = get_activation(activation)

        # Input layers.
        if len(inputs) == 1:
            net_input = inputs[0]
        else:
            layer = Concatenate()
            layer.name = "conct_" + layer.name.split("_")[-1]
            net_input = layer(inputs)

        # Define the output network.
        net = [net_input]
        for nLay, nNeuron in enumerate(hidden_layers):
            # Add the layer.
            layer = Dense(
                nNeuron,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                trainable=trainable,
                dtype=dtype,
            )
            layer.name = "D{:d}b_".format(nNeuron) + layer.name.split("_")[-1]
            layers.append(layer)
            net[-1] = layer(net[-1])
            # Apply the activation.
            if afunc.__name__ != 'linear': #nLay<len(hidden_layers)-1 and
                layer = Activation(afunc)
                layer.name = "{}_".format(afunc.__name__) + layer.name.split("_")[-1]
                layers.append(layer)
                net[-1] = layer(net[-1])

        # store output layers.
        for out in output_fileds:
            layers.append(out)

        # Assign to the output variable
        if len(net) == 1:
            net_output = net[0]
        else:
            raise ValueError("Legacy for Enrichment: Must be updated. ")
            layer = Concatenate()
            layer.name = "conct_" + layer.name.split("_")[-1]
            net_output = layer(net)

        # check output activation functions.
        output_func = get_activation(output_activation)
        # Define the final outputs of each network
        outputs = []
        for out in output_fileds:
            # add the activation on the output.
            if output_func.__name__ != 'linear':
                layer = Activation(output_func)
                layer.name = "{}_".format(output_func.__name__) + layer.name.split("_")[-1]
                layers.append(layer)
                outputs.append(layer(out(net_output)))
            else:
                outputs.append(out(net_output))

        self._inputs = inputs
        self._outputs = outputs
        self._layers = layers
        self._set_model()

    def eval(self, *kwargs):
        """ Evaluates the functional object for a given input.

        # Arguments
            (SciModel, Xs): 
                Evalutes the functional object from the beginning 
                    of the graph defined with SciModel. 
                    The Xs should match those of SciModel. 
            
            (Xs):
                Evaluates the functional object from inputs of the functional. 
                    Xs should match those of inputs to the functional. 
                    
        # Returns
            Numpy array of dimensions of network outputs. 

        # Raises
            ValueError:
            TypeError:
        """
        if len(kwargs) == 1:
            model = self.model
            # read data.
            mesh = kwargs[0]
        elif len(kwargs) == 2:
            if validations.is_scimodel(kwargs[0]):
                model = K.function(kwargs[0].model.inputs, self.outputs)
            else:
                raise ValueError(
                    'Expected a SciModel object for the first arg. '
                )
            mesh = kwargs[1]
        else:
            raise NotImplemented()
        x_pred = to_list(mesh)
        # To have unified output for postprocessing - limitted support.
        shape_default = x_pred[0].shape if all([x.shape==x_pred[0].shape for x in x_pred]) else None
        # prepare X,Y data.
        for i, (x, xt) in enumerate(zip(x_pred, model.inputs)):
            x_shape = tuple(xt.get_shape().as_list())
            if x.shape != x_shape:
                try:
                    x_pred[i] = x.reshape((-1,) + x_shape[1:])
                except:
                    print(
                        'Could not automatically convert the inputs to be ' 
                        'of the same size as the expected input tensors. ' 
                        'Please provide inputs of the same dimension as the `Variables`. '
                    )
                    assert False

        y_pred = to_list(model(x_pred))

        if shape_default is not None:
            try:
                y_pred = [y.reshape(shape_default) for y in y_pred]
            except:
                print("Input and output dimensions need re-adjustment for post-processing.")

        return unpack_singleton(y_pred)

    @property
    def layers(self):
        return self._layers

    @layers.setter
    def layers(self, value):
        self._layers = value

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    @property
    def model(self):
        self._set_model()
        return self._model
    
    def _set_model(self):
        if hasattr(self, '_model'):
            if is_same_tensor(self._inputs, self._model.inputs) and \
               is_same_tensor(self._outputs, self._model.outputs):
               return
        self._model = K.function(
            unique_tensors(self._inputs),
            self._outputs
        )

    def get_weights(self, at_layer=None):
        return [l.get_weights() for l in self.layers]

    def set_weights(self, weights):
        try:
            for l, w in zip(self.layers, weights):
                l.set_weights(w)
        except:
            raise ValueError(
                'Provide data exactly the same as .get_weights() outputs. '
            )

    def count_params(self):
        return sum([l.count_params() for l in self.layers])

    def copy(self):
        return Functional(
            inputs=self.inputs,
            outputs=self.outputs,
            layers=self.layers
        )

    def append_to_layers(self, layers):
        if self.layers is not layers:
            cl = [x.name for x in self.layers]
            for x in layers:
                if not x.name in cl:
                    self.layers += [x]

    def append_to_inputs(self, inputs):
        if self.inputs is not inputs:
            cl = [x.name for x in self.inputs]
            for x in inputs:
                if not x.name in cl:
                    self.inputs.append(x)

    def append_to_outputs(self, outputs):
        self._outputs += to_list(outputs)

    def set_trainable(self, val):
        if isinstance(val, bool):
            for l in self._layers:
                l.trainable = val
        else:
            raise ValueError('Expected a boolean value: True or False')
        return self

    def reinitialize_weights(self):
        for lay in self.layers:
            if hasattr(lay, 'kernel_initializer') and lay.kernel is not None:
                K.set_value(lay.kernel, lay.kernel_initializer(lay.kernel.shape))
            if hasattr(lay, 'bias_initializer') and lay.bias is not None:
                K.set_value(lay.bias, lay.bias_initializer(lay.bias.shape))
        return self

    def split(self):
        """ In the case of `Functional` with multiple outputs,
            you can split the outputs and get an associated functional.

        # Returns
            (f1, f2, ...): Tuple of splitted `Functional` objects
                associated to each output.
        """
        if len(self._outputs)==1:
            return self
        fs = ()
        # total number of outputs to get splitted.
        nr = len(self._outputs)
        # associated to each output, there is a layer to be splitted.
        lays = self.layers[:-nr]
        for out, lay in zip(self._outputs, self._layers[-nr:]):
            # copy constructor for functional.
            f = Functional(
                inputs = to_list(self.inputs),
                outputs = to_list(out),
                layers = lays + to_list(lay)
            )
            fs += (f,)
        return fs

    def __call__(self):
        return self.outputs

    def __pos__(self):
        return self

    def __neg__(self):
        return self*-1.0

    def __add__(self, other):
        return math.add(self, other)

    def __radd__(self, other):
        return math.radd(self, other)

    def __sub__(self, other):
        return math.sub(self, other)

    def __rsub__(self, other):
        return math.rsub(self, other)

    def __mul__(self, other):
        return math.mul(self, other)

    def __rmul__(self, other):
        return math.rmul(self, other)

    def __truediv__(self, other):
        return math.div(self, other)

    def __rtruediv__(self, other):
        return math.rdiv(self, other)

    def __pow__(self, power):
        return math.pow(self, power)

    def __getitem__(self, item):
        return getitem(self, item)

    def diff(self, *args, **kwargs):
        return math.diff(self, *args, **kwargs)

    @classmethod
    def get_class(cls):
        return Functional

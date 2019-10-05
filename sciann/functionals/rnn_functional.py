""" Functional class for SciANN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import *

from keras.layers import Dense, LSTM, SimpleRNN
from keras.layers import Activation
from keras.layers import Concatenate

from .rnn_field import RNNField


class RNNFunctional(object):
    """ Configures the LSTMFunctional object (Recurrent Neural Network).

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
        rnn_type: currently, `SimpleRNN` and `LSTM` are accepted.
            Defaulted to `SimpleRNN`.
            Check `Keras` documentation for additional information.
        activation: Activation function for the hidden layers.
            Last layer will have a linear output.
        enrichment: Activation function to be applied to the network output.
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
                 rnn_type="SimpleRNN",
                 activation="tanh",
                 recurrent_activation=None,
                 enrichment="linear",
                 kernel_initializer=default_kernel_initializer(),
                 bias_initializer=default_bias_initializer(),
                 dtype=None,
                 trainable=True,
                 **kwargs,):
        # check data-type.
        if dtype is None:
            dtype = K.floatx()
        elif not K.floatx() == dtype:
            K.set_floatx(dtype)
        # check for copy constructor.
        if all([x in kwargs for x in ('inputs', 'outputs', 'layers')]):
            self._inputs = kwargs['inputs']
            self._outputs = kwargs['outputs']
            self._layers = kwargs['layers']
            return
        # prepares fields.
        fields = to_list(fields)
        if all([isinstance(fld, str) for fld in fields]):
            outputs = [
                RNNField(
                    name=fld,
                    dtype=dtype,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    trainable=trainable,
                )
                for fld in fields
            ]
        elif all([validations.is_field(fld) for fld in fields]):
            outputs = fields
        else:
            raise TypeError(
                'Please provide a "list" of field names of'
                + ' type "String" or "Field" objects.'
            )
        # prepare inputs/outputs/layers.
        inputs = []
        layers = []
        variables = to_list(variables)
        if all([isinstance(var, RNNFunctional) for var in variables]):
            for var in variables:
                inputs += var.outputs
            for var in variables:
                for lay in var.layers:
                    layers.append(lay)
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

        # check enrichment functions.
        enrichment = to_list(enrichment)
        efuncs = get_activation(enrichment)

        # Input layers.
        if len(inputs) == 1:
            net_input = inputs[0]
        else:
            layer = Concatenate()
            layer.name = "conct_" + layer.name.split("_")[-1]
            net_input = layer(inputs)

        # Define the output network.
        net = []
        for enrich in efuncs:
            net.append(net_input)
            for nLay, nNeuron in enumerate(hidden_layers):
                # Add the layer.
                if rnn_type=='LSTM':
                    layer = LSTM(
                        nNeuron,
                        return_sequences=True,
                        recurrent_activation=recurrent_activation,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        trainable=trainable,
                        dtype=dtype,
                        unroll=True,
                    )
                elif rnn_type=='SimpleRNN':
                    layer = SimpleRNN(
                        nNeuron,
                        return_sequences=True,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        trainable=trainable,
                        dtype=dtype,
                        unroll=True,
                    )
                else:
                    raise ValueError(
                        'Invalid entry for `rnn_type` -- '
                        'accepts from (`SimpleRNN`, `LSTM`).'
                    )
                layer.name = "D{:d}b_".format(nNeuron) + layer.name.split("_")[-1]
                layers.append(layer)
                net[-1] = layer(net[-1])
                # Apply the activation.
                if nLay<len(hidden_layers)-1 and afunc.__name__ != 'linear':
                    layer = Activation(afunc)
                    layer.name = "{}_".format(afunc.__name__) + layer.name.split("_")[-1]
                    layers.append(layer)
                    net[-1] = layer(net[-1])

            # add the activations.
            if enrich.__name__ != 'linear':
                layer = Activation(enrich)
                layer.name = "{}_".format(enrich.__name__) + layer.name.split("_")[-1]
                layers.append(layer)
                net[-1] = layer(net[-1])

        # store output layers.
        for out in outputs:
            layers.append(out)

        # Assign to the output variable
        if len(net) == 1:
            net_output = net[0]
        else:
            layer = Concatenate()
            layer.name = "conct_" + layer.name.split("_")[-1]
            net_output = layer(net)

        # Define the final outputs of each network
        outputs = [out(net_output) for out in outputs]

        self._inputs = inputs
        self._outputs = outputs
        self._layers = layers

    def eval(self, model, mesh):
        assert validations.is_scimodel(model), \
            'Expected a SciModel object. '
        return unpack_singleton(
            K.function(model.model.inputs, self._outputs)(mesh)
        )

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

    def copy(self):
        return RNNFunctional(
            inputs=self.inputs,
            outputs=self.outputs,
            layers=self.layers
        )

    def append_to_layers(self, layers):
        self.layers = self.layers + layers

    def append_to_inputs(self, inputs):
        self.inputs = self.inputs + inputs

    def append_to_outputs(self, outputs):
        self.outputs = self.outputs + outputs

    def split(self):
        """ In the case of `Functional` with multiple outputs,
            you can split the outputs and get an associated functional.

        # Returns
            (f1, f2, ...): Tuple of splitted `Functional` objects
                associated to cheach outputs.
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
            f = RNNFunctional(
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
        return add(self, other)

    def __radd__(self, other):
        return radd(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return rmul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, power):
        return pow(self, power)

    def diff(self, *args, **kwargs):
        return diff(self, *args, **kwargs)

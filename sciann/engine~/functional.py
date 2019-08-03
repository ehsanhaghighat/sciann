""" Functional class for SciANN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import *

from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import InputSpec

default_bias_initializer = k.initializers.random_uniform()
default_kernel_initializer = k.initializers.glorot_normal()


class Functional(object):
    """ Configures the Functional object (Neural Network).

    # Arguments
        fields (String or Field): [Sub-]Network outputs.
            It can be of type `String` - Associated fields will be created internally.
            It can be of type `Field` or `Functional`
        variables (Variable): [Sub-]Network inputs.
            It can be of type `Variable` or other Functional objects.
        hidden_layers: A list indicating neurons in the hidden layers.
            e.g. [10, 100, 20] is a for hidden layers with 10, 100, 20, respectively.
        activation: Activation function for the hidden layers.
            Last layer will have a linear output.
        enrichment: Activation function to be applied to the network output.
        kernel_initializer: Initializer of the `Kernel`, from `k.initializers`.
        bias_initializer: Initializer of the `Bias`, from `k.initializers`.
        dtype: data-type of the network parameters, can be
            ('float16', 'float32', 'float64').
            Note: Only network inputs should be set.
        trainable (Boolean): False if network is not trainable, True otherwise.
            Default value is True.

    # Raises
        ValueError:
        TypeError:
    """
    def __init__(self,
                 fields=None,
                 variables=None,
                 hidden_layers=None,
                 activation="linear",
                 enrichment="linear",
                 kernel_initializer=default_kernel_initializer,
                 bias_initializer=default_bias_initializer,
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
                Field(
                    name=fld,
                    dtype=dtype,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    trainable=trainable,
                )
                for fld in fields
            ]
        elif all([isinstance(fld, Field) for fld in fields]):
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
        if all([isinstance(var, Functional) for var in variables]):
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
        assert is_scimodel(model), \
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
        return Functional(
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
            dtype = K.floatx()
        elif not dtype == K.floatx():
            K.set_floatx(dtype)

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


class Field(Dense):
    """ Configures the `Field` class for the model outputs.

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
                 activation=k.activations.linear,
                 kernel_initializer=default_kernel_initializer,
                 bias_initializer=default_bias_initializer,
                 trainable=True,
                 dtype=None,):
        if not dtype:
            dtype = K.floatx()
        elif not dtype == K.floatx():
            K.set_floatx(dtype)

        assert isinstance(name, str), \
            "Please provide a string for field name. "
        assert callable(activation), \
            "Please provide a function handle for the activation. "

        super(Field, self).__init__(
            units=units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=True,
            trainable=trainable,
            name=name,
            dtype=dtype,
        )


class RadialBasisVariable(Functional):
    """ Configures the `RadialBasisVariable` object for the network's input.

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

        super(RadialBasisVariable, self).__init__(
            layers=to_list(layer),
            inputs=to_list(layer.input),
            outputs=to_list(layer.output),
        )


class RadialBasisLayer(Functional):
    """

    """
    def __init__(self, vars, rb_vars, radii):
        vars = to_list(vars)
        if not all([isinstance(x, Variable) for x in vars]):
            raise TypeError

        rb_vars = to_list(rb_vars)
        if not all([isinstance(x, RadialBasisVariable) for x in rb_vars]):
            raise TypeError

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

        super(RadialBasisLayer, self).__init__(
            layers=layers,
            inputs=inputs,
            outputs=outputs
        )


class Parameter(Dense):
    """ Parameter class to be used for parameter inversion.
        Inherited from Dense layer.

    # Arguments
        val (float): Initial value for the parameter.
        non_neg (boolean): True (default) if only non-negative values are expected.
        **kwargs: keras.layer.Dense accepted arguments.
    """
    def __init__(self, val=1.0, non_neg=True, **kwargs):
        super(Parameter, self).__init__(
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


class ToBeInferred(Functional):
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
            Parameter(val=val, non_neg=non_neg, name=name)
        )

        lay = Concatenate()
        lay.name = "conct_" + lay.name.split("_")[-1]
        lay_input = lay(inputs_tensors)
        outputs = layers[-1](lay_input)

        super(ToBeInferred, self).__init__(
            inputs=to_list(inputs_tensors),
            outputs=to_list(outputs),
            layers=to_list(layers)
        )

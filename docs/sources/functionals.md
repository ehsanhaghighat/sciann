# Intro

A combination of neural network layers form a `Functional`. 

Mathematically, a `functional` is a general mapping from input set \\(X\\) onto some output set \\(Y\\). Once the parameters of this transformation are found, this mapping is called a `function`. 

`Functional`s are needed to form `SciModels`. 

A `Functional` is a class to form complex architectures (mappings) from inputs (`Variables`) to the outputs. 


```python
from sciann import Variable, Functional

x = Variable('x')
y = Variable('y')

Fxy = Functional('Fxy', [x, y], 
                 hidden_layers=[10, 20, 10],
                 activation='tanh')
```

`Functionals` can be plotted when a `SciModel` is formed. A minimum of one `Constraint` is needed to form the SciModel

```python
from sciann.constraints import Data
from sciann import SciModel

model = SciModel(x, Data(Fxy), 
                 plot_to_file='output.png')
```

---

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/functional.py#L23)</span>
### Functional

```python
sciann.functionals.functional.Functional(fields=None, variables=None, hidden_layers=None, activation='tanh', output_activation='linear', kernel_initializer=<keras.initializers.VarianceScaling object at 0x7fdd1073b450>, bias_initializer=<keras.initializers.RandomUniform object at 0x7fdd1073b4d0>, dtype=None, trainable=True)
```

Configures the Functional object (Neural Network).

__Arguments__

- __fields__: String or Field.
    [Sub-]Network outputs.
    It can be of type `String` - Associated fields will be created internally.
    It can be of type `Field` or `Functional`
- __variables__: Variable.
    [Sub-]Network inputs.
    It can be of type `Variable` or other Functional objects.
- __hidden_layers__: A list indicating neurons in the hidden layers.
    e.g. [10, 100, 20] is a for hidden layers with 10, 100, 20, respectively.
- __activation__: defaulted to "tanh".
    Activation function for the hidden layers.
    Last layer will have a linear output.
- __output_activation__: defaulted to "linear".
    Activation function to be applied to the network output.
- __kernel_initializer__: Initializer of the `Kernel`, from `k.initializers`.
- __bias_initializer__: Initializer of the `Bias`, from `k.initializers`.
- __dtype__: data-type of the network parameters, can be
    ('float16', 'float32', 'float64').
    Note: Only network inputs should be set.
- __trainable__: Boolean.
    False if network is not trainable, True otherwise.
    Default value is True.

__Raises__

- __ValueError__:
- __TypeError__:
    
----

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/variable.py#L10)</span>
### Variable

```python
sciann.functionals.variable.Variable(name=None, units=1, tensor=None, dtype=None)
```

Configures the `Variable` object for the network's input.

__Arguments__

- __name__: String.
    Required as derivatives work only with layer names.
- __units__: Int.
    Number of feature of input var.
- __tensor__: Tensorflow `Tensor`.
    Can be pass as the input path.
- __dtype__: data-type of the network parameters, can be
    ('float16', 'float32', 'float64').

__Raises__


    
----

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/field.py#L12)</span>
### Field

```python
sciann.functionals.field.Field(name=None, units=1, activation=<function linear at 0x7fdd235c6710>, kernel_initializer=<keras.initializers.VarianceScaling object at 0x7fdd1072ec90>, bias_initializer=<keras.initializers.RandomUniform object at 0x7fdd1072ed50>, trainable=True, dtype=None)
```

Configures the `Field` class for the model outputs.

__Arguments__

- __name__: String.
    Assigns a layer name for the output.
- __units__: Positive integer.
    Dimension of the output of the network.
- __activation__: Callable.
    A callable object for the activation.
- __kernel_initializer__: Initializer for the kernel.
    Defaulted to a normal distribution.
- __bias_initializer__: Initializer for the bias.
    Defaulted to a normal distribution.
- __trainable__: Boolean to activate parameters of the network.
- __dtype__: data-type of the network parameters, can be
    ('float16', 'float32', 'float64').

__Raises__


    
----

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/parameter.py#L22)</span>
### Parameter

```python
sciann.functionals.parameter.Parameter(val=1.0, min_max=None, inputs=None, name=None, non_neg=None)
```

Parameter functional to be used for parameter inversion.
Inherited from Dense layer.

__Arguments__

- __val__: float.
    Initial value for the parameter.
- __min_max__: [MIN, MAX].
    A range to constrain the value of parameter.
    This constraint will overwrite non_neg constraint if both are chosen.
- __inputs__: Variables.
    List of `Variable`s to the parameters.
- __name__: str.
    A name for the Parameter layer.
- __non_neg__: boolean.
    True (default) if only non-negative values are expected.
- __**kwargs__: keras.layer.Dense accepted arguments.

    

# Intro

`Variable` is a way to to define inputs to the network, very much similar to the `Input` class in `Keras`. However, since we need to perform differentiation and other operations on the network, we cannot just use `Input`. Instead, we need to define the inputs of the network through `Variable`. 

For scientific computations, a `Variable` has only a dimension of 1. Therefore, if you need to have a three-dimensional coordinate inputs, you need to define three variables:

```python
from sciann import Variable

x = Variable('x')
y = Variable('y')
z = Variable('z')
```

This is precisely because we need to perform differentiation with respect to (x, y, z). 


---

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/variable.py#L11)</span>
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


    

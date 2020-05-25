# Intro

`Field` is a layer to define outputs of each Functional. It is very much similar to Keras' `Dense` layer. 

It is not necessary to be defined explicitly, however, if you are expecting multiple outputs, it is better to be defined using `Field`.  

```python
from sciann import Field

Fx = Field(name='Fx', units=10)
```

---

<span style="float:right;">[[source]](https://github.com/sciann/sciann/tree/master/sciann/functionals/field.py#L13)</span>
### Field

```python
sciann.functionals.field.Field(name=None, units=1, activation=<function linear at 0x7f7ee822f170>, kernel_initializer=<keras.initializers.VarianceScaling object at 0x7f7e789ff210>, bias_initializer=<keras.initializers.RandomUniform object at 0x7f7e789ff310>, kernel_regularizer=None, bias_regularizer=None, trainable=True, dtype=None)
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
- __kernel_regularizer__: Regularizer for the kernel.
    By default, it uses l1=0.001 and l2=0.001 regularizations.
    To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
- __bias_regularizer__: Regularizer for the bias.
    By default, it uses l1=0.001 and l2=0.001 regularizations.
    To set l1 and l2 to custom values, pass [l1, l2] or {'l1':l1, 'l2':l2}.
- __trainable__: Boolean to activate parameters of the network.
- __dtype__: data-type of the network parameters, can be
    ('float16', 'float32', 'float64').

__Raises__


    

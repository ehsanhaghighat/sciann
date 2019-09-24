'''
# Curve fitting in 1D

Here, a 1D curve fitting example is explored. Imagine, a synthetic data
generated from \\\( \sin(x) \\\) over the range of \\\( [0, 2\pi] \\\).

To train a neural network model on this curve, you should first define a `Variable`.

A neural network with three layers, each containing 10 neurons, and with `tanh` activation function is then generated
using the `Functional` class.

The target is imposed on the output using the `Data` class from `Constraint`, and passed to the `SciModel` to form a
Sciann model.
'''

import numpy as np
from sciann.functionals.rnn_variable import RNNVariable
from sciann.functionals.rnn_functional import RNNFunctional
from sciann.functionals.rnn_field import RNNField
from sciann import SciModel
from sciann.utils import diff, gradients
from sciann.constraints import Data, Tie


# Synthetic data generated from sin function over [0, 2pi]
x_true = np.linspace(0, np.pi*2, 10000)
y_true = np.sin(x_true)
dy_true = np.cos(x_true)

# The network inputs should be defined with Variable.
tunits = 2
x = RNNVariable(tunits, name='x', dtype='float32')

# Each network is defined by Functional.
y = RNNFunctional('y', x, [5])

# Define the target (output) of your model.
c1 = Data(y)
c2 = Data(diff(y, x))

# The model is formed with input `x` and condition `c1`.
model = SciModel(x, [c1, c2])

# Training: .train runs the optimization and finds the parameters.
model.train(x_true.reshape(-1, tunits, 1), [y_true.reshape(-1, tunits, 1), dy_true.reshape(-1, tunits, 1)], batch_size=32, epochs=100)

# used to evaluate the model after the training.
x_pred = np.linspace(0, np.pi*4, 10000)
y_pred, dy_pred = model.predict(x_pred.reshape(-1, tunits, 1))

y_star = np.sin(x_pred)
dy_star = np.cos(x_pred)

import matplotlib.pyplot as plt
plt.plot(x_pred, y_pred.reshape(-1), x_pred, y_star)
plt.plot(x_pred, dy_pred.reshape(-1), x_pred, dy_star)
plt.show()

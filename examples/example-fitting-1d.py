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
from sciann import Variable, Functional, SciModel
from sciann.constraints import Data


# Synthetic data generated from sin function over [0, 2pi]
x_true = np.linspace(0, np.pi*2, 10000)
y_true = np.sin(x_true)

# The network inputs should be defined with Variable.
x = Variable('x', dtype='float32')

# Each network is defined by Functional.
y = Functional('y', x, [10, 10, 10], activation='tanh')

# Define the target (output) of your model.
c1 = Data(y)

# The model is formed with input `x` and condition `c1`.
model = SciModel(x, c1)

# Training: .train runs the optimization and finds the parameters.
model.train(x_true, y_true, batch_size=32, epochs=100)

# used to evaluate the model after the training.
y_pred = model.predict(x_true)


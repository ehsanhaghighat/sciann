'''
# Use SciANN to perform neural network training on synthetic data.

TODO: Add proper document.
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

# The training data is a condition (constraint) on the model.
c1 = Data(y, y_true)

# The model is formed with input `x` and condition `c1`.
model = SciModel(x, c1)

# Training: .solve runs the optimization and finds the parameters. 
model.solve(x_true, batch_size=32, epochs=100)

# used to evaluate the model after the training. 
y_pred = model.predict(x_true)
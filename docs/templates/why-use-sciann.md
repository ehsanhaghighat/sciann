# Why use SciANN among all other codes?

The main purpose of SciANN is a platform for people with Scientific Computations backgrounds in mind. 

You will find this code very useful for: 

- Solving ODEs and PDEs using densely connect, complex networks, recurrent networks are on the way. 

- This platform is ready to use for Curve Fitting, Differentiations, Integration, etc. 

- If you have other scientific computations in mind that are not implemented yet, [`contact us`](mailto:info@sciann.com). 

As an example, let's fit a neural network with three-hidden layers, each with 10 neurons and \\( \tanh \\) activation function, on data generated from \\( sin(x) \\): 

```python
import numpy as np
from sciann import Variable, Functional, SciModel
from sciann.conditions import Data

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
```

As you may find, this code takes advantage of [`Keras`](http://keras.io) great design and takes it to the next level for scientific computations. 


w
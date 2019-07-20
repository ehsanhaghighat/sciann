# Using Functional to form complex network architectures 

The `Functional` class is designed to allow users to design complex networks with a few lines of code. 

To use Functional, you can follow the exmaple bellow: 

```python
import numpy as np
from sciann Variable, Functional, SciModel
from sciann.conditions Data

# Synthetic data to be fitted. 
x_true = np.linspace(0.0, 2*np.pi, 10000)
y_true = np.sin(x_true)

# Functional requires input features to be defined through Variable. 
x = Variable("x", dtype='float32')

# A complex network with 5 hidden layers ([5, 10, 20, 10, 5]), 
# and feature aumentation [x, x**2, x**3, sin(x), cos(x), sinh(x)].
y = Functional(
    "y", 
    [x, x**2, x**3, sin(x), cos(x), sinh(x)],
    hidden_layers = [5, 10, 20, 10, 5],
    activations = 'tanh',
)

# Define the SciModel. 
model = SciModel(x, Data(y, y_true))

# Solve the neural network model.
model.solve(x_true, epochs=32, batches=10)

# Find model's prediciton. 
x_pred = model.predict(x_true)
```


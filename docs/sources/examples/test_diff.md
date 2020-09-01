

```python
from sciann import Variable, Field, Functional, SciModel
from sciann.utils.math import diag_grad, div_grad, grad, dot
from sciann.utils.math import diag_part, diag
import keras.backend as K

num_node = 1000
famil = np.arange(0, num_node).reshape(-1, 1) + np.array([-1, 0, 1])
famil[famil>999] = num_node - famil[famil>999]

xtrain = np.linspace(-2*np.pi, 2*np.pi, num_node)
gf00 = np.concatenate([np.ones((num_node,1))*0.2, np.ones((num_node,1))*0.6, np.ones((num_node,1))*0.2], axis=-1)

x = Variable("x", units=3, dtype='float64')
GF00 = Variable("GF00", units=3, dtype='float64')
y = Field("y", units=3, dtype='float64')
fx = Functional(y, x, 4*[10], 'tanh')

yf = dot(fx, GF00)
gf = grad(fx, x)
gf1 = diag_grad(fx, x)
gf2 = div_grad(fx, x)

dotgf = dot(gf, diag(GF00))

raise ValueError


model = SciModel([x, GF00], gf)

model.train([xtrain[famil], gf00], np.cos(xtrain), epochs=1000)

xtest = np.linspace(-4*np.pi, 4*np.pi, num_node)
ypred = yf.eval(model, [xtest[famil], gf00])
dypred = gf.eval(model, [xtest[famil], gf00])

import matplotlib.pyplot as plt
plt.plot(xtest, ypred, 'b', xtest, np.sin(xtest), '--g')
plt.plot(xtest, dypred, 'r', xtest, np.cos(xtest), '--k')
plt.show()
```
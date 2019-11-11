# Getting started with the SciANN model or `SciModel`

The `SciModel` is the relation between network inputs, i.e. `Variable` and network outputs, i.e. `Conditions`. 

You can set up a `SciModel` as simple as the code bellow:

```python
from sciann import Variable, Functional, SciModel
from sciann.conditions import Data

x = Variable("x")
y = Functional("y", x)
cy = Data(y)
model = SciModel(cy)
```


----
# -*- coding: utf-8 -*-
'''
General documentation architecture:

Home
Index

- Getting started
    Guide to SciANN model
    Guide to Functional
    FAQ

- SciModels
    About SciANN models
        explain when one should use Sequential or functional API
        explain compilation step
        explain weight saving, weight loading
        explain serialization, deserialization
    Sequential
    Model (functional API)

- Functionals
    About SciANN functionals
        explain common functional operations: get_weights, set_weights, get_config
        explain input_shape
        explain usage on non-Keras tensors
    Variables
    Fields
    Functionals 

Operations
Constraints 

'''
from sciann import utils
from sciann import Functional
from sciann import Variable
from sciann import Field
from sciann import SciModel
from sciann import constraints
from sciann import utils
from sciann import Parameter


EXCLUDE = {
    'Constraint'
}

PAGES = [
    {
        'page': 'scimodels.md',
        'classes': [
            SciModel
        ],
        'methods': [
            SciModel.train,
            SciModel.solve,
            SciModel.predict,
            SciModel.loss_functions,
        ]
    },
    {
        'page': 'functionals.md',
        'classes': [
            Functional,
            Variable,
            Field,
            Parameter,
        ]
    },
    {
        'page': 'variables.md',
        'classes': [
            Variable
        ]
    },
    {
        'page': 'fields.md',
        'classes': [
            Field
        ]
    },
    {
        'page': 'constraints.md',
        'classes': [
            constraints.Data,
            constraints.PDE,
            constraints.Tie
        ]
    },
    {
        'page': 'utils.md',
        'methods': [
            utils.math.gradients,
            utils.math.lambda_gradient,
            utils.math.diff,
            utils.math.radial_basis,
            utils.math.sin,
            utils.math.asin,
            utils.math.cos,
            utils.math.acos,
            utils.math.tan,
            utils.math.atan,
            utils.math.tanh,
            utils.math.exp,
            utils.math.pow,
            utils.math.add,
            utils.math.sub,
            utils.math.mul,
            utils.math.div,
        ],
    },
]

ROOT = 'https://sciann.com/'

template_np_implementation = """# Numpy implementation

    ```python
{{code}}
    ```
"""

template_hidden_np_implementation = """# Numpy implementation

    <details>
    <summary>Show the Numpy implementation</summary>

    ```python
{{code}}
    ```

    </details>
"""

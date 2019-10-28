""" Constraint class to condition on the training.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from ..utils import *


class Constraint(object):
    """ Configures the condition to impose constraint.

    # Arguments
        var: The layer name to impose the constraint.
        cond: A callable handle to function that imposes the condition.
        name: A `str` to be associated to the constraint.

    # Returns

    # Raises
        ValueError if `model` is not of class `SciModel`.
        ValueError for unrecognized inputs.
    """
    def __init__(self,
                 var=None,
                 cond=None,
                 name=None,
                 **kwargs):
        # check the inputs.
        assert isinstance(var, (str, type(None))), \
            "Expected a Layer Name of type str. "
        assert (cond is None or callable(cond) or is_tensor(cond)), \
            "Expected a function or a Tensor as input. "
        if 'ids' in kwargs or 'sol' in kwargs:
            raise ValueError(
                'Legacy interface: please use `SciModel` and `SciModel.train` to impose on specific ids. '
            )
        elif len(kwargs)>0:
            raise ValueError(
                'Unrecognized input variable: {}'.format(kwargs.keys())
            )
        assert isinstance(name, (str, type(None))), \
            "Expected a string input for the name. "
        self.var = var
        self.cond = cond
        self.name = name

    def __call__(self):
        return self.cond

    def eval(self, xs):
        return self.cond.eval(xs)

""" PDE class to impose pde constraint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constraint import *


class MinMax(Constraint):
    """MinMax weight constraint.

    Constrains the weights incident to each hidden unit
    to have values between a lower bound and an upper bound.

    # Arguments
        min_value: the minimum norm for the incoming weights.
        max_value: the maximum norm for the incoming weights.

    # Raises
        ValueError: 'cond' should be a functional object.
    """
    def __init__(self, cond, min_value=None, max_value=None, name="minmax"):
        if not is_functional(cond):
            raise ValueError(
                "Expected a Functional object, received a "
                "{} - {}".format(type(cond), cond)
            )
        if min_value > max_value:
            raise ValueError(
                "Check inputs: `min_value` should be smaller than `max_value`. "
            )
        try:
            delta = max_value - min_value
            cond = (relu(cond - max_value)**2 + relu(min_value - cond)**2) / delta**2
        except (ValueError, TypeError):
            assert False, 'Unexpected error - cannot evaluate the regularization. '

        super(MinMax, self).__init__(
            cond=cond,
            name=name
        )

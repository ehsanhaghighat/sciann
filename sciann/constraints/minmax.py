""" PDE class to impose pde constraint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constraint import Constraint
from ..utils import is_functional
from ..utils import relu, sign, abs, tanh


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
    def __init__(self, cond, min_value=None, max_value=None, penalty=1.0, name="minmax"):
        if not is_functional(cond):
            raise ValueError(
                "Expected a Functional object, received a "
                "{} - {}".format(type(cond), cond)
            )
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError(
                "Check inputs: `min_value` should be smaller than `max_value`. "
            )
        try:
            delta = max_value - min_value
            const = 0.0
            if min_value is not None:
                const += (1.0 - sign(cond - min_value)) * abs(cond - min_value)
            if max_value is not None:
                const += (1.0 + sign(cond - max_value)) * abs(cond - max_value)
            const *= penalty
        except (ValueError, TypeError):
            assert False, 'Unexpected error - cannot evaluate the regularization. '

        super(MinMax, self).__init__(
            cond=const,
            name=name
        )

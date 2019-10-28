""" PDE class to impose pde constraint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constraint import *


class PDE(Constraint):
    """ PDE class to impose to the system.

    # Arguments
        pde: Functional.
            The `Functional` object that pde if formed on.
        name: String.
            A `str` for name of the pde.

    # Returns

    # Raises
        ValueError: 'pde' should be a functional object.
    """
    def __init__(self, pde, name="pde", **kwargs):
        if not is_functional(pde):
            raise ValueError(
                "Expected a Functional object as the pde, received a "
                "{} - {}".format(type(pde), pde)
            )
        # Check inputs.
        if 'mesh_ids' in kwargs or 'sol' in kwargs:
            raise ValueError(
                'Legacy interface: please use `SciModel` and `SciModel.train` to impose on specific ids. '
            )
        elif len(kwargs)>0:
            raise ValueError(
                'Unrecognized input variable: {}'.format(kwargs.keys())
            )
        super(PDE, self).__init__(
            cond=pde,
            name=name
        )

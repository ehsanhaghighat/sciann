""" Tie constraint to tie different outputs of the network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constraint import *


class Tie(Constraint):
    """ Tie class to constrain network outputs.
        constraint: `cond1 - cond2 == sol`.

    # Arguments
        cond1: Functional.
            A `Functional` object to be tied to cond2.
        cond2: Functional.
            A 'Functional' object to be tied to cond1.
        name: String.
            A `str` for name of the pde.

    # Returns

    # Raises
        ValueError: 'pde' should be a functional object.
    """
    def __init__(self, cond1, cond2, name="tie", **kwargs):
        # prepare cond.
        if not is_functional(cond1):
            raise ValueError(
                "Expected a Functional object as the cond1, received a "
                "{} - {}".format(type(cond1), cond1)
            )
        if not is_functional(cond2):
            raise ValueError(
                "Expected a Functional object as the cond2, received a "
                "{} - {}".format(type(cond2), cond2)
            )
        # Form the constraint.
        try:
            cond = cond1-cond2
        except (ValueError, TypeError):
            print(
                'Unexpected ValueError/TypeError - ',
                'make sure `cond1` and `cond2` are functional objects. \n',
                'cond1 - {} \n'.format(cond1),
                'cond2 - {} \n'.format(cond2)
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
        super(Tie, self).__init__(
            cond=cond,
            name=name
        )

""" Data class to impose data constraint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constraint import *


class Data(Constraint):
    """ Data class to impose to the system.

    # Arguments
        cond: Functional.
            The `Functional` object that Data condition
            will be imposed on.
        name: String.
            A `str` for name of the pde.

    # Returns

    # Raises
        ValueError: 'cond' should be a functional object.
                    'mesh' should be a list of numpy arrays.
    """
    def __init__(self, cond, name="data", **kwargs):
        if not is_functional(cond):
            raise ValueError(
                "Expected a Functional object as the `cond`, received a "
                "{} - {}".format(type(cond), cond)
            )
        if 'x_true_ids' in kwargs or 'y_true' in kwargs:
            raise ValueError(
                "Legacy inputs: please check `SciModel` and `SciModel.train` on how to impose partial data. "
            )
        elif len(kwargs)>0:
            raise ValueError(
                'Unrecognized input variable: {}'.format(kwargs.keys())
            )
        super(Data, self).__init__(
            cond=cond,
            name=name
        )

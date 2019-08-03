""" Data class to impose data constraint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constraint import *


class Data(Constraint):
    """ Data class to impose to the system.

    # Arguments
        cond (Functional): The `Functional` object that Data condition
            will be imposed on.
        y_true (np.ndarray): Expected output to set the `pde` to.
            If not provided, will be set to `zero`.
        x_true_ids (np.ndarray): A 1D numpy arrays consists of node-ids to impose the condition.
        name (String): A `str` for name of the pde.

    # Returns

    # Raises
        ValueError: 'cond' should be a functional object.
                    'mesh' should be a list of numpy arrays.
    """
    def __init__(self, cond, y_true=None, x_true_ids=None, name="data"):
        if not is_functional(cond):
            raise ValueError(
                "Expected a Functional object as the `cond`, received a "
                "{} - {}".format(type(cond), cond)
            )
        if x_true_ids is not None:
            if not all([isinstance(x_true_ids, np.ndarray), x_true_ids.ndim==1]):
                raise ValueError(
                    "Expected a 1D-numpy arrays for `x_true_ids`, received a "
                    "{} - {}".format(type(x_true_ids), x_true_ids)
                )
        # prepare expected output.
        if y_true is not None:
            y_true = to_list(y_true)
            if not all([isinstance(x, np.ndarray) for x in y_true]):
                raise ValueError(
                    "Expected a list of numpy arrays for `sol`, received a "
                    "{} - {}".format(type(y_true), y_true)
                )
            if len(y_true) != len(cond.outputs):
                raise ValueError(
                    "Number of expected outputs in `sol` does not match "
                    "number of outputs from the constraint. \n "
                    "Provided {} \nExpected {} ".format(y_true, cond.outputs)
                )

        super(Data, self).__init__(
            cond=cond,
            ids=x_true_ids,
            sol=y_true,
            name=name
        )

""" Dirichlet class to impose dirichlet constraint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constraint import *


class Dirichlet(Constraint):
    """ Dirichlet class to impose to the system.

    # Arguments
        cond (Functional): The `Functional` object that Dirichlet condition
            will be imposed on.
        sol (np.ndarray): Expected output to set the `pde` to.
            If not provided, will be set to `zero`.
        mesh_ids (np.ndarray): A 1D numpy arrays consists of node-ids to impose the condition.
        name (String): A `str` for name of the pde.

    # Returns

    # Raises
        ValueError: 'cond' should be a functional object.
                    'mesh' should be a list of numpy arrays.
    """
    def __init__(self, cond, sol=None, mesh_ids=None, name="dirichlet"):
        if not is_functional(cond):
            raise ValueError(
                "Expected a Functional object as the `cond`, received a "
                "{} - {}".format(type(cond), cond)
            )
        # prepare mesh.
        if mesh_ids is not None:
            if not all([isinstance(mesh_ids, np.ndarray), mesh_ids.ndim==1]):
                raise ValueError(
                    "Expected a 1d numpy arrays of mesh ids, received a "
                    "{} - {}".format(type(mesh_ids), mesh_ids)
                )
        # prepare expected output.
        if sol is not None:
            sol = to_list(sol)
            if not all([isinstance(x, np.ndarray) for x in sol]):
                raise ValueError(
                    "Expected a list of numpy arrays for `sol`, received a "
                    "{} - {}".format(type(sol), sol)
                )
            if len(sol) != len(cond.outputs):
                raise ValueError(
                    "Number of expected outputs in `sol` does not match "
                    "number of outputs from the constraint. \n "
                    "Provided {} \nExpected {} ".format(sol, cond.outputs)
                )

        super(Dirichlet, self).__init__(
            cond=cond,
            ids=mesh_ids,
            sol=sol,
            name=name
        )

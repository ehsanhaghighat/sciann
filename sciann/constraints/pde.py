""" PDE class to impose pde constraint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .constraint import *


class PDE(Constraint):
    """ PDE class to impose to the system.

    # Arguments
        pde (Functional): The `Functional` object that pde if formed on.
        sol (np.ndarray): Expected output to set the `pde` to.
            If not provided, will be set to `zero`.
        mesh_ids (np.ndarray): A 1D numpy arrays consists of node-ids to impose the condition.
        name (String): A `str` for name of the pde.

    # Returns

    # Raises
        ValueError: 'pde' should be a functional object.
                    'mesh' should be a list of numpy arrays.
    """
    def __init__(self, pde, sol=None, mesh_ids=None, name="pde"):
        if not is_functional(pde):
            raise ValueError(
                "Expected a Functional object as the pde, received a "
                "{} - {}".format(type(pde), pde)
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
            if len(sol) != len(pde.outputs):
                raise ValueError(
                    "Number of expected outputs in `sol` does not match "
                    "number of outputs from the constraint. \n "
                    "Provided {} \nExpected {} ".format(sol, pde.outputs)
                )

        super(PDE, self).__init__(
            cond=pde,
            ids=mesh_ids,
            sol=sol,
            name=name
        )

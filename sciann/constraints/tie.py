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
        cond1 (Functional): A `Functional` object to be tied to cond2.
        cond2 (Functional): A 'Functional' object to be tied to cond1.
        sol (np.ndarray): Expected output to set the `pde` to.
            If not provided, will be set to `zero`.
        mesh_ids (np.ndarray): A 1D numpy arrays consists of node-ids to impose the condition.
        name (String): A `str` for name of the pde.

    # Returns

    # Raises
        ValueError: 'pde' should be a functional object.
                    'mesh' should be a list of numpy arrays.
    """
    def __init__(self, cond1, cond2, sol=None, mesh_ids=None, name="tie"):
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

        super(Tie, self).__init__(
            cond=cond,
            ids=mesh_ids,
            sol=sol,
            name=name
        )

""" Process/Prepare standard data for the `Radial Basis Networks`.
"""

import numpy as np
from sklearn.neighbors import BallTree


def find_neighbors(*args, **kwargs):
    """ Find closest point to each node in xs.

    # Arguments
        xs: A list of input vectors of dimension (N,1) with N as number of data-points.
        ys: A list of output vectors of dimension (N,1) with N as number of data-points.
        size_rb: Number of closest points.
        return_ids: Boolean.

    # Returns
        xrb: A list of N-`neighbors` of dimension (N,size_rb).
    """
    if len(args)==2:
        xs = args[0]
        ys = None
        size_rb = args[1]
    elif len(args)==3:
        xs = args[0]
        ys = args[1]
        size_rb = args[2]
    else:
        raise ValueError

    # prepare data.
    if all([x.shape[-1]!=1 for x in xs]):
        for i, x in enumerate(xs):
            xs[i] = x.reshape(x.shape + (1,))
    xsc = np.concatenate(xs, axis=-1)

    tree = BallTree(xsc, size_rb)
    ids = tree.query(xsc, size_rb, return_distance=False)

    if 'return_ids' in kwargs and kwargs['return_ids']:
        return ids
    else:
        xrbs = [x[ids].reshape(-1, size_rb) for x in xs]
        if ys is None:
            return xrbs
        else:
            yrbs = [y[ids].reshape(-1, size_rb) for y in ys]
            return xrbs, yrbs

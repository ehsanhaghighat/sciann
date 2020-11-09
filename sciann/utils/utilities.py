""" Built-in utilities to process inputs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import pi

import tensorflow as tf
import tensorflow.python.keras as k
import tensorflow.python.keras.backend as K

# interface for some keras features to be acessible across sciann.
from tensorflow.python.keras.backend import is_keras_tensor as is_tensor
from tensorflow.python.keras.backend import floatx
from tensorflow.python.keras.backend import set_floatx
from tensorflow.python.keras.utils.vis_utils import model_to_dot
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.initializers import RandomUniform as default_bias_initializer
from tensorflow.python.keras.initializers import GlorotNormal as default_kernel_initializer
from tensorflow.python.keras.initializers import Constant as default_constant_initializer
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.regularizers import l1_l2

from .initializers import SciKernelInitializer as KInitializer
from .initializers import SciBiasInitializer as BInitializer
from .activations import get_activation, SciActivation, SciActivationLayer


def _is_tf_1():
    return tf.__version__.startswith('1.')


def set_random_seed(val=1234):
    """ Set random seed for reproducibility.

    # Arguments
        val: A seed value..

    """
    np.random.seed(val)
    if _is_tf_1():
        tf.set_random_seed(val)
    else:
        tf.random.set_seed(val)


def reset_session():
    """ Clear keras and tensorflow sessions.
    """
    if _is_tf_1():
        K.clear_session()
    else:
        tf.keras.backend.clear_session()


clear_session = reset_session


def is_same_tensor(x, y):
    if len(to_list(x)) != len(to_list(y)):
        return False
    else:
        res = []
        for xi, yi in zip(to_list(x), to_list(y)):
            res.append(xi.name == yi.name)
        return all(res)


def unique_tensors(Xs):
    if len(Xs) > 1:
        ux, uids = np.unique([x.name for x in Xs], return_index=True)
        uids = sorted(uids)
        return [Xs[i] for i in uids]
    else:
        return Xs


def default_regularizer(*args, **kwargs):
    l1, l2 = 0.0, 0.0
    if (len(args) == 0 and len(kwargs) == 0) or args[0] is None:
        return None
    elif len(args) == 1:
        if isinstance(args[0], (float, int)):
            l1 = 0.0
            l2 = args[0]
        elif isinstance(args[0], list):
            l1 = args[0][0]
            l2 = args[0][1]
        elif isinstance(args[0], dict):
            l1 = 0.0 if 'l1' not in args[0] else args[0]['l1']
            l2 = 0.0 if 'l2' not in args[0] else args[0]['l2']
    elif len(args) == 2:
        l1 = args[0]
        l2 = args[1]
    elif len(kwargs) > 0:
        l1 = 0.0 if 'l1' not in kwargs else kwargs['l1']
        l2 = 0.0 if 'l2' not in kwargs else kwargs['l2']
    else:
        raise ValueError('Unrecognized entry - input regularization values for l1 and l2.')
    # print("regularization is used with l1={} and l2={}".format(l1, l2))
    return l1_l2(l1=l1, l2=l2)


def default_weight_initializer(actf='linear', distribution='uniform', mode='fan_in', scale=None):
    inz = []
    for i, af in enumerate(to_list(actf)):
        if distribution in ('uniform', 'normal'):
            tp = VarianceScaling(
                scale=eval_default_scale_factor(af, i) if scale is None else scale,
                mode=mode, distribution=distribution
            )
        elif distribution in ('constant',):
            tp = default_constant_initializer(0.0 if scale is None else scale)
        else:
            raise ValueError('Undefined distribution: pick from ("uniform", "normal", "constant").')
        inz.append(tp)
    return inz


def eval_default_scale_factor(actf, lay):
    if actf in ('linear', 'relu'):
        return 2.0
    elif actf in ('tanh', 'sigmoid'):
        return 1.0 if lay > 0 else 1.0
    elif actf in ('sin', 'cos'):
        return 2.0 if lay > 0 else 2.0 #*30.0
    else:
        return 1.0


def prepare_default_activations_and_initializers(actfs, seed=None):
    activations = []
    bias_initializer = []
    kernel_initializer = []
    for lay, actf in enumerate(to_list(actfs)):
        if isinstance(actf, str):
            lay_actf = actf.lower().split('l-')
        else:
            raise ValueError('expected a string for actf.')
        bias_initializer.append(BInitializer(lay_actf[-1], lay, seed))
        kernel_initializer.append(KInitializer(lay_actf[-1], lay, seed))
        f = get_activation(lay_actf[-1])
        w = kernel_initializer[-1].w0
        if len(lay_actf) == 1:
            activations.append(SciActivation(w, f))
        else:
            activations.append(SciActivationLayer(w, f))

    return activations, bias_initializer, kernel_initializer


def unpack_singleton(x):
    """Gets the first element if the iterable has only one value.

    Otherwise return the iterable.

    # Argument
        x: A list or tuple.

    # Returns
        The same iterable or the first element.
    """
    if len(x) == 1:
        return x[0]
    return x


def to_list(x, allow_tuple=False):
    """Normalizes a list/tensor into a list.

    If a tensor is passed, we return
    a list of size 1 containing the tensor.

    # Arguments
        x: target object to be normalized.
        allow_tuple: If False and x is a tuple,
            it will be converted into a list
            with a single element (the tuple).
            Else converts the tuple to a list.

    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    if allow_tuple and isinstance(x, tuple):
        return list(x)
    return [x]


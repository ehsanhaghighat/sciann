""" Utilities to process functionals.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Lambda
from keras.layers import Dot
from keras.layers import Input
from keras.models import Model

from .utilities import *
from .validations import *


def pow(f, a):
    """Element-wise exponentiation applied to the `Functional` object.

    # Arguments
        f: Functional object.
        a: Python integer.

    # Returns
        A Functional.
    """
    validate_functional(f)

    res = f.copy()
    lmbd = [Lambda(lambda x: x**a) for X in f.outputs]
    for l in lmbd:
        if isinstance(a, int):
            l.name = "pow{:d}/".format(a) + l.name.split("_")[-1]
        elif isinstance(a, float):
            l.name = "pow{:.3f}/".format(a) + l.name.split("_")[-1]
        else:
            raise ValueError(
                'Expected an `int` or `float` for a in x^a. '
            )
    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, res)
    return res


def add(f, other):
    """Element-wise addition applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    res = f.copy()
    if is_functional(other):
        res.append_to_inputs(other.inputs)
        res.append_to_layers(other.layers)
        lmbd = [Lambda(lambda x: x[0]+x[1]) for X in f.outputs]
    else:
        lmbd = [Lambda(lambda x: x+other) for X in f.outputs]

    for l in lmbd:
        l.name = "add/" + l.name.split("_")[-1]
    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, res, other)
    return res


def radd(f, other):
    """Element-wise right-addition applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    return add(f, other)


def sub(f, other):
    """Element-wise subtraction applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    res = f.copy()
    if is_functional(other):
        res.append_to_inputs(other.inputs)
        res.append_to_layers(other.layers)
        lmbd = [Lambda(lambda x: x[0]-x[1]) for X in f.outputs]
    else:
        lmbd = [Lambda(lambda x: x-other) for X in f.outputs]

    for l in lmbd:
        l.name = "sub/" + l.name.split("_")[-1]
    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, res, other)
    return res


def rsub(f, other):
    """Element-wise right-subtraction applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    res = f.copy()
    if is_functional(other):
        res.append_to_inputs(other.inputs)
        res.append_to_layers(other.layers)
        lmbd = [Lambda(lambda x: x[1]-x[0]) for X in f.outputs]
    else:
        lmbd = [Lambda(lambda x: other-x) for X in f.outputs]

    for l in lmbd:
        l.name = "rsub/" + l.name.split("_")[-1]
    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, res, other)
    return res


def mul(f, other):
    """Element-wise multiplication applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    res = f.copy()
    if is_functional(other):
        res.append_to_inputs(other.inputs)
        res.append_to_layers(other.layers)
        lmbd = [Lambda(lambda x: x[0]*x[1]) for X in f.outputs]
    else:
        lmbd = [Lambda(lambda x: x*other) for X in f.outputs]

    for l in lmbd:
        l.name = "mul/" + l.name.split("_")[-1]

    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, res, other)
    return res


def rmul(f, other):
    """Element-wise right-multiplication applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    return mul(f, other)


def div(f, other):
    """Element-wise division applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    res = f.copy()
    if is_functional(other):
        res.append_to_inputs(other.inputs)
        res.append_to_layers(other.layers)
        lmbd = [Lambda(lambda x: x[0]/x[1]) for X in f.outputs]
    else:
        lmbd = [Lambda(lambda x: x/other) for X in f.outputs]

    for l in lmbd:
        l.name = "div/" + l.name.split("_")[-1]
    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, res, other)
    return res


def rdiv(f, other):
    """Element-wise right-division applied to the `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    res = f.copy()
    if is_functional(other):
        res.append_to_inputs(other.inputs)
        res.append_to_layers(other.layers)
        lmbd = [Lambda(lambda x: x[1]/x[0]) for X in f.outputs]
    else:
        lmbd = [Lambda(lambda x: other/x) for X in f.outputs]

    for l in lmbd:
        l.name = "rdiv/" + l.name.split("_")[-1]
    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, res, other)
    return res


def dot(f, other):
    """Dot product of two `Functional` objects.

    # Arguments
        f: Functional object.
        other: A python number or a tensor or a functional object.

    # Returns
        A Functional.
    """
    validate_functional(f)

    res = f.copy()
    if is_functional(other):
        res.append_to_inputs(other.inputs)
        res.append_to_layers(other.layers)
        lmbd = [k.layers.Dot(axes=-1) for X in f.outputs]
    else:
        lmbd = [Lambda(lambda x: x*other) for X in f.outputs]

    for l in lmbd:
        l.name = "dot/" + l.name.split("_")[-1]

    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, res, other)
    return res


def _apply_operation(lambda_layer, lhs, rhs=None):
    """Element-wise mathematical operation applied on the `Functional` objects.

    # Arguments
        lambda_layer: the layers to perform the operation.
        lhs: left hand side objects.
        rhs: right hand side objects.

    # Returns
        output tensors.
    """
    validate_functional(lhs)

    if is_functional(rhs):
        outputs = [l([x, y]) for l, x, y in zip(lambda_layer, lhs.outputs, rhs.outputs)]
    else:
        try:
            outputs = [l(x) for l, x in zip(lambda_layer, lhs.outputs)]
        except (ValueError, TypeError):
            print(
                'Unsupported operation with an object of type {}. '.format(type(lhs))
            )
            outputs = None

    return outputs


def sin(x):
    """Computes sin of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'sin')


def asin(x):
    """Computes asin of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'asin')


def cos(x):
    """Computes cos of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'cos')


def acos(x):
    """Computes acos of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'acos')


def tan(x):
    """Computes tan of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'tan')


def atan(x):
    """Computes atan of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'atan')


def cot(x):
    """Computes cot of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'cot')


def acot(x):
    """Computes acot of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'acot')


def sinh(x):
    """Computes sinh of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'sinh')


def cosh(x):
    """Computes cosh of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'cosh')


def tanh(x):
    """Computes tanh of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'tanh')


def coth(x):
    """Computes coth of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'coth')


def abs(x):
    """Computes abs of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'abs')


def log(x):
    """Computes log of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'log')


def exp(x):
    """Computes exp of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'exp')


def sqrt(x):
    """Computes sqrt of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'sqrt')


def square(x):
    """Computes square of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'square')


def relu(x):
    """Computes relu of x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    return _apply_function(x, 'relu')


def mean(x, **kwargs):
    """Apply mean to the `Functional` objects on far-right axis.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    if "axis" not in kwargs:
        kwargs["axis"] = -1
    if "keepdims" not in kwargs:
        kwargs["keepdims"] = True
    return _apply_function(x, 'mean', **kwargs)


def _apply_function(x, fname, **kwargs):
    """Apply `fname` function to x element-wise.

    # Arguments
        x: Functional object.

    # Returns
        A new functional object.
    """
    validate_functional(x)
    res = x.copy()

    fun = get_activation(fname)
    lmbd = [Lambda(lambda x: fun(x, **kwargs)) for X in x.outputs]
    for l in lmbd:
        l.name = "{}/".format(fname) + l.name.split("_")[-1]
    # res.append_to_layers(lmbd)
    res.outputs = _apply_operation(lmbd, x)
    return res


def gradients(ys, xs, order=1):
    """Returns the gradients of y in `ys` w.r.t. x in `xs`.
    
    `ys` and `xs` are each a Tensor or a list of tensors.

    # Arguments
        ys: A tensor or list of tesnors to be differentiated.
        xs: A tensor or list of tensors to be used for differentiation.
        order: Order of differentiation. 

    # Returns
        A list of `D^n y / Dx^n` for each y and x in `ys` and `xs`.
    """
    ds = ys
    for i in range(order):
        ds = unpack_singleton(
            tf.gradients(
                ds, xs,
                unconnected_gradients='zero',
                # colocate_gradients_with_ops=True, TF: V1.14.0
            )
        )
    return ds


def getitem(x, item):
    """returns specific item of a tensor (Functional).

    # Arguments
        item: Item list.

    # Returns
        A new functional object.
    """
    validate_functional(x)
    res = x.copy()

    ys = []
    lmbd = [Lambda(lambda xx: xx.__getitem__(item)) for xx in x.outputs]
    for l, y in zip(lmbd, x.outputs):
        # l.name = "slice/" + l.name.split("_")[-1]
        ys.append(l(y))

    res.outputs = ys
    return res


def lambda_gradient(ys, xs, order=1, name=''):
    """Returns the gradients of y in `ys` w.r.t. x in `xs` using Lambda layers.
    
    `ys` and `xs` are each a Tensor or a list of tensors.

    # Arguments
        ys: A tensor or list of tesnors to be differentiated.
        xs: A tensor or list of tensors to be used for differentiation.
        name: A str name for the Lambda layer. 

    # Returns
        A tuple, `(layers, grads)`.
        layers: A Lambda layer or list of Lambda layers where the gradient operator is applied.
        grads: A gradient tensor or list of gradient tensors. 
    """
    name_prefix = 'Grad_' if order==1 else 'Grad{:d}_'.format(order)
    
    grads, layers = [], []
    for y in to_list(ys):
        lay = Lambda(lambda y: gradients(y, xs, order))
        lay.name = name_prefix + name + '/' + lay.name
        layers += to_list(lay)
        grads += to_list(lay(y))

    return (unpack_singleton(layers), unpack_singleton(grads))


def diff(f, *args, **kwargs):
    """Computes diff of functional object f.

    # Arguments
        f: Functional object.
        ys: layer name for `ys` to differentiate.
        xs: layer name for `xs` to be differentiated w.r.t.
        order: order of differentiation w.r.t. xs - defaulted to 1.

    # Returns
        A new functional object.
    """

    assert is_functional(f), \
        'Please provide a proper functional object. '
    assert(len(args) in (1,2)), \
        'Expected (`ys`, `xs`) or `xs` as the input, '\
        'was provided {:d} inputs'.format(len(args))
    if not all([isinstance(v, str) or is_functional(v) for v in args]):
        raise ValueError(
            'Expected a `Layer` name for a `Functional` to perform differentitation.'
        )

    try:
        if len(args) == 0:
            x = unpack_singleton(f.inputs)
            assert is_tensor(x), \
                'multiple inputs detected - please provide an `x` name. '
            x_name = x.name.split('/')[0]
        else:
            x_id = 0 if len(args)==1 else 1
            if isinstance(args[x_id], str):
                x_name = args[x_id]
                x = next(l for l in f.layers if l.name == x_name).output
            elif is_functional(args[x_id]):
                x_lay = args[x_id].layers[-1]
                x_name = x_lay.name
                x = x_lay.output
            else:
                raise TypeError('Unsupported `x` entry. ')
            
        if len(args) <= 1:
            y = unpack_singleton(f.outputs)
            assert is_tensor(y), \
                'multiple outputs detected - please provide a `y` name. '
            y_name = y.name.split('/')[0]
        else:
            y_id = 0
            if isinstance(args[y_id], str):
                y_name = args[y_id]
                y = next(l for l in f.layers if l.name == y_name).output
            elif is_functional(args[y_id]):
                y_lay = args[y_id].layers[-1]
                y_name = y_lay.name
                y = y_lay.output
            else:
                raise TypeError('Unsupported `y` entry. ')
        
    except (StopIteration, ValueError):
        print("Did not find the layer {}. ".format(args))

    # check order of differentiation.
    order = 1
    if 'order' in kwargs.keys():
        order = kwargs['order']

    res = f.copy()
    
    lay, tens = lambda_gradient(
        y, x, order, "{}_{}".format(y_name, x_name)
    )

    res.append_to_layers(to_list(lay))
    res.outputs = to_list(tens)

    return res


def radial_basis(xs, ci, radii):
    """Apply `radial_basis` function to x element-wise.

    # Arguments
        xs: List of functional objects.
        ci: Center of basis functional (same length as xs).
        radii: standard deviation or radius from the center.

    # Returns
        A new functional object.
    """
    assert len(xs) == len(ci)
    assert radii > 0.0
    assert all([is_variable(x) for x in xs])
    assert isinstance(xs, list) and isinstance(ci, list)

    for x in xs:
        validate_variable(x)

    return exp(-sum([(x - c)**2 for x, c in zip(xs, ci)])/radii**2)


def radial_basis2(xs, ci, radii):
    """Apply `radial_basis` function to x element-wise.

    # Arguments
        xs: List of functional objects.
        ci: Center of basis functional (same length as xs).
        radii: standard deviation or radius from the center.

    # Returns
        A new functional object.
    """
    assert len(xs) == len(ci)
    assert radii > 0.0
    assert all([is_variable(x) for x in xs])
    assert isinstance(xs, list) and isinstance(ci, list)

    for x in xs:
        validate_variable(x)

    d = xs[0] - ci[0]
    for i in range(1, len(xs)):
        d += xs[i] - ci[i]
    d /= radii
    
    return exp(-sum([(x - c)**2 for x, c in zip(xs, ci)])/radii**2)

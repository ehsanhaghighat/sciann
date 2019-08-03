""" Utilities to process functionals.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sciann


def is_functional(f):
    """ Checks whether `f` is a functional object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if functional.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.Functional):
        return True

    else:
        return False


def validate_functional(f):
    """ if `f` is not a functional object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if functional, False otherwise.

    # Raises
        ValueError: if the object is not a Functional object.

    """
    if isinstance(f, sciann.Functional):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `functional` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors.'
        )


def is_constraint(f):
    """ Checks whether `f` is a `Constraint` object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Constraint.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.Constraint):
        return True

    else:
        return False


def validate_constraint(f):
    """ if `f` is not a Constraint object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Constraint, False otherwise.

    # Raises
        ValueError: if the object is not a Constraint object.

    """
    if isinstance(f, sciann.Constraint):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `Constraint` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors '
            'or layers. '
        )


def is_parameter(f):
    """ Checks whether `f` is a parameter object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if a parameter.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.Parameter):
        return True

    else:
        return False


def validate_parameter(f):
    """ if `f` is not a parameter object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if parameter, False otherwise.

    # Raises
        ValueError: if the object is not a Parameter object.

    """
    if isinstance(f, sciann.Parameter):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `parameter` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors.'
        )


def is_field(f):
    """ Checks whether `f` is a `Field` object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Field.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.Field):
        return True

    else:
        return False


def validate_field(f):
    """ if `f` is not a Field object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Field, False otherwise.

    # Raises
        ValueError: if the object is not a Field object.

    """
    if isinstance(f, sciann.Field):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `Field` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors '
            'or layers. '
        )


def is_variable(f):
    """ Checks whether `f` is a `Variable` object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Variable.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.Variable):
        return True

    else:
        return False


def validate_variable(f):
    """ if `f` is not a Variable object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if Variable, False otherwise.

    # Raises
        ValueError: if the object is not a Variable object.

    """
    if isinstance(f, sciann.Variable):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `Variable` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors '
            'or layers. '
        )


def is_scimodel(f):
    """ Checks whether `f` is a `SciModel` object.

    # Arguments
        f: an object to be tested.

    # Returns
        True if SciModel.

    # Raises
        ValueError: if the object cannot be tested with `isinstance`.

    """
    if isinstance(f, sciann.SciModel):
        return True

    else:
        return False


def validate_scimodel(f):
    """ if `f` is not a SciModel object, raises value error.

    # Arguments
        f: an object to be tested.

    # Returns
        True if SciModel, False otherwise.

    # Raises
        ValueError: if the object is not a SciModel object.

    """
    if isinstance(f, sciann.SciModel):
        return True

    else:
        raise ValueError(
            'These operations can only be applied to the `SciModel` object. '
            'Use `Keras` or `TensorFlow` functions when applying to tensors '
            'or layers. '
        )

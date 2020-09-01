
import tensorflow.python.keras as k
import tensorflow as tf

class SciActivation(k.layers.Activation):
    """Applies an activation function to an output.

    # Arguments
        w0: The factor to be applied to initialized weights.
            e.g. sin(w0 * input)
        activation: name of activation function to use
            (see: [activations](../activations.md)),
            or alternatively, a TensorFlow operation.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, w0=1.0, activation='linear', **kwargs):
        super(SciActivation, self).__init__(activation, **kwargs)
        self.w0 = w0

    def call(self, inputs):
        return self.activation(self.w0 * inputs)

    def get_config(self):
        config = {'w0': self.w0}
        base_config = super(SciActivation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_activation(activation):
    """ Evaluates the activation function from a string or list of string inputs.

    # Arguments
        activation: A string pointing to the function name.

    # Returns:
        A function handle.
    """

    if isinstance(activation, list):
        return [get_activation(act) for act in activation]

    elif isinstance(activation, str):
        if hasattr(k.activations, activation):
            return getattr(k.activations, activation)
        elif hasattr(k.backend, activation):
            return getattr(k.backend, activation)
        elif hasattr(tf.math, activation):
            return getattr(tf.math, activation)
        else:
            raise ValueError(
                'Not a valid function name: ' + activation +
                ' - Please provide a valid activation '  
                'function name from tensorflow.python.keras or Tensorflow. '
            )

    elif callable(activation):
        return activation

    else:
        raise TypeError(
            'Please provide a valid input: ' + type(activation) +
            ' - Expecting a function name or function handle. '
        )


k.utils.generic_utils.get_custom_objects().update({
    'SciActivation': SciActivation
})

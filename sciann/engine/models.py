""" SciModel class to define and train the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..utils import *

from keras.models import Model
from keras.utils import plot_model

from .functional import Variable, RadialBasisVariable
from .constraint import Constraint


class SciModel(object):
    """Configures the model for training.

    # Arguments
        inputs: Main variables of the network, also known as `xs`,
            should be of type `Variable`.
        constraints: list all conditions to be imposed on the training;
            should be of type `Constraint`.
        plot_to_file: A string fine name to output the network architecture.

    # Returns

    # Raises
        ValueError: `inputs` must be of type Variable.
                    `constraints` must be of type Functional.
    """
    def __init__(self,
                 inputs=None,
                 constraints=None,
                 loss_func="mse",
                 plot_to_file=None,
                 **kwargs):
        # strictly check for inputs to be of type variable.
        inputs = to_list(inputs)
        if not all([isinstance(x, (Variable, RadialBasisVariable)) for x in inputs]):
            raise ValueError(
                'Please provide a `list` of `Variable` or `RadialBasisVariable` objects for inputs. '
            )
        # prepare input tensors.
        input_vars = []
        for var in inputs:
            input_vars += var.inputs
        # check outputs if of correct type.
        constraints = to_list(constraints)
        if not all([isinstance(y, Constraint) for y in constraints]):
            raise ValueError('Please provide a "list" of "Constraint"s.')
        # prepare network outputs.
        output_vars = []
        for cond in constraints:
            output_vars += cond().outputs
        # prepare loss_functions.
        if isinstance(loss_func, str) and loss_func in ["mse", "mae"]:
            loss_func = SciModel.loss_functions(loss_func)
        elif not callable(loss_func):
            raise TypeError(
                'Please provide a valid loss function from ("mse", "mae") '
                + "or a callable function for input of tensor types. "
            )
        # Initialize the Model form super class.
        model = Model(
            inputs=input_vars,
            outputs=output_vars,
            **kwargs
        )
        # compile the model.
        # multiple optimizers were tested, "ADAM" worked best on my tests.
        model.compile(
            loss=loss_func,
            optimizer="adam",
            # optimizer=k.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0),
            # optimizer=k.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False),
            # optimizer = k.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
        )

        # Set the variables.
        self._model = model
        self._inputs = inputs
        self._constraints = constraints
        self._loss_func = loss_func

        # Plot to file if requested.
        if plot_to_file is not None:
            plot_model(self._model, to_file=plot_to_file)

    @property
    def model(self):
        return self._model

    @property
    def constraints(self):
        return self._constraints

    @property
    def inputs(self):
        return self._inputs

    def verify_update_constraints(self, constraints):
        ver = []
        for old, new in zip(self._constraints, constraints):
            if old==new and old.sol==new.sol:
                ver.append(True)
            else:
                ver.append(False)
        return all(ver)

    def __call__(self, *args, **kwargs):
        output = self._model.__call__(*args, **kwargs)
        return output if isinstance(output, list) else [output]

    def save(self, filepath, *args, **kwargs):
        return self._model.save(filepath, *args, **kwargs)

    def summary(self, *args, **kwargs):
        return self._model.summary(*args, **kwargs)

    def solve(self,
              x_true,
              weights=None,
              epochs=10,
              batch_size=2**8,
              shuffle=True,
              callbacks=None,
              stop_after=100,
              default_zero_weight=1.0e-10,
              **kwargs,):
        """Performs the training on the model.

        # Arguments
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
            batch_size: Integer or 'None'.
                Number of samples per gradient update.
                If unspecified, 'batch_size' will default to 128.
            shuffle: Boolean (whether to shuffle the training data).
                Default value is True.
            callbacks: List of `keras.callbacks.Callback` instances.

        # Returns
            A 'History' object after performing fitting.
        """
        if callbacks is None:
            callbacks = [
                k.callbacks.EarlyStopping(
                    monitor="loss", mode="min", verbose=1, patience=stop_after
                ),
                k.callbacks.TerminateOnNaN(),
            ]

        # prepare X,Y data.
        x_true = [x.reshape(-1, 1) if len(x.shape)==1 else x for x in to_list(x_true)]
        num_sample = x_true[0].shape[0]
        assert all([x.shape[0]==num_sample for x in x_true[1:]])
        ids_all = np.arange(0, num_sample)

        if weights is None:
            weights = np.ones(num_sample)
        else:
            if len(weights.shape)!=1 or \
                weights.shape[0] != num_sample:
                raise ValueError(
                    'Input error: `weights` should have dimension 1 with '
                    'the same sample length as `Xs. '
                )
                
        y_true, sample_weights = [], []
        for c in self._constraints:
            # prepare sample weight.
            if c.ids is None:
                ids = ids_all
                wei = [weights for yi in c.cond.outputs]
            else:
                ids = c.ids
                wei = [np.zeros(num_sample)+default_zero_weight for yi in c.cond.outputs]
                for w in wei:
                    w[ids] = weights[ids]
                    w[ids] *= sum(weights)/sum(w[ids])
            # prepare y_true.
            sol = [np.zeros(((num_sample,) + k.backend.int_shape(yi)[1:])) for yi in c.cond.outputs]
            if c.sol is not None:
                for yi, soli in zip(sol, c.sol):
                    yi[ids, :] = soli
            # add to the list.
            y_true += sol
            sample_weights += wei

        # perform the training.
        history = self._model.fit(
            x_true, y_true,
            sample_weight=sample_weights,  #sums to number of samples.
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=callbacks,
            **kwargs,
        )
        return history

    def predict(self, x,
                batch_size=None,
                verbose=0,
                steps=None):
        """ Predict output from network.

        # Arguments
            x:
            batch_size:
            verbose:
            steps:

        # Returns
            List of numpy array of the size of network outputs.

        # Raises

        """
        return self._model.predict(x, batch_size, verbose, steps)

    def eval(self, *args):
        if len(args) == 1:
            x_data = to_list(args[0])
            if not all([isinstance(xi, np.ndarray) for xi in x_data]):
                raise ValueError("Please provide input data to the network. ")
            return unpack_singleton(self.predict(x_data))

        elif len(args) == 2:
            var_name = args[0]
            if not isinstance(var_name, str):
                raise ValueError("Value Error: Expected a LayerName as the input. ")
            x_data = to_list(args[1])
            new_model = Model(self.inputs, self.get_layer(var_name).output)
            if not all([isinstance(xi, np.ndarray) for xi in x_data]):
                raise ValueError("Please provide input data to the network. ")
            return unpack_singleton(new_model.predict(x_data))

    def plot_model(self, *args, **kwargs):
        """ Keras plot_model functionality.
            Refer to Keras documentation for help.
        """
        plot_model(self._model, *args, **kwargs)

    @staticmethod
    def loss_functions(method="mse"):
        """ loss_function returns the callable object to evaluate the loss.

        # Arguments
            method: String.
                "mse" for `Mean Squared Error` or
                "mae" for 'Mean Absolute Error'.
        # Returns
            Callable function that gets (y_true, y_pred) as the input and
                returns the loss value as the output.
        # Raises
            ValueError if anything other than "mse" or "mae" is passed.
        """
        if method == "mse":
            return lambda y_true, y_pred: K.mean(K.square(y_true - y_pred))
        elif method == "mae":
            return lambda y_true, y_pred: K.mean(K.abs(y_true - y_pred))
        else:
            raise ValueError

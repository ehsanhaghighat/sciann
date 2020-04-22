""" SciModel class to define and train the model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import keras.backend as K
import keras as k
import numpy as np

from keras.models import Model
from keras.utils import plot_model

from ..utils import unpack_singleton, to_list
from ..utils import is_variable, is_constraint, is_functional

from ..functionals import Variable
from ..functionals import RadialBasis
from ..constraints import Data, Tie


class SciModel(object):
    """Configures the model for training.
    Example:
    # Arguments
        inputs: Main variables (also called inputs, or independent variables) of the network, `xs`.
            They all should be of type `Variable`.
        targets: list all targets (also called outputs, or dependent variables)
            to be satisfied during the training. Expected list members are:
            - Entries of type `Constraint`, such as Data, Tie, etc.
            - Entries of type `Functional` can be:
                . A single `Functional`: will be treated as a Data constraint.
                    The object can be just a `Functional` or any derivatives of `Functional`s.
                    An example is a PDE that is supposed to be zero.
                . A tuple of (`Functional`, `Functional`): will be treated as a `Constraint` of type `Tie`.
            - If you need to impose more complex types of constraints or
                to impose a constraint partially in a specific part of region,
                use `Data` or `Tie` classes from `Constraint`.
        loss_func: defaulted to "mse" or "mean_squared_error".
            It can be an string from supported loss functions, i.e. ("mse" or "mae").
            Alternatively, you can create your own loss function and
            pass the function handle (check Keras for more information).
        optimizer: defaulted to "adam" optimizer.
            It can be one of Keras accepted optimizers, e.g. "adam".
            You can also pass more details on the optimizer:
            - `optimizer = k.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)`
            - `optimizer = k.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)`
            - `optimizer = k.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)`
            Check our Keras documentation for further details. We have found
        load_weights_from: (file_path) Instantiate state of the model from a previously saved state.
        plot_to_file: A string file name to output the network architecture.

    # Raises
        ValueError: `inputs` must be of type Variable.
                    `targets` must be of types `Functional`, or (`Functional`, data), or (`Functional`, `Functional`).
    """
    def __init__(self,
                 inputs=None,
                 targets=None,
                 loss_func="mse",
                 optimizer="adam",
                 load_weights_from=None,
                 plot_to_file=None,
                 **kwargs):
        # strictly check for inputs to be of type variable.
        inputs = to_list(inputs)
        if not all([is_variable(x) for x in inputs]):
            raise ValueError(
                'Please provide a `list` of `Variable` or `RadialBasis` objects for inputs. '
            )
        # prepare input tensors.
        input_vars = []
        for var in inputs:
            input_vars += var.inputs
        # check outputs if of correct type.
        if targets is None:
            if 'constraints' in kwargs:
                targets = kwargs.get('constraints')
            elif 'conditions' in kwargs:
                targets = kwargs.get('conditions')
        else:
            if 'conditions' in kwargs or 'constraints' in kwargs:
                raise TypeError(
                    'Inconsistent inputs: `constraints`, `conditions`, and `targets` are all equivalent keywords '
                    '- pass all targets as a list to `SciModel`. '
                )
        # setup constraints.
        targets = to_list(targets)
        for i, y in enumerate(targets):
            if not is_constraint(y):
                if is_functional(y):
                    # Case of Data-type constraint.
                    targets[i] = Data(y)
                elif isinstance(y, tuple) and \
                        len(y) == 2 and \
                        is_functional(y[0]) and is_functional(y[1]):
                    # Case of Tie-type constraint.
                    targets[i] = Tie(y[0], y[1])
                else:
                    # Not recognised.
                    raise ValueError(
                        'The {}th target entry is not of type `Constraint` or `Functional` - '
                        'received \n ++++++ {} '.format(i, y)
                    )
        # prepare network outputs.
        output_vars = []
        for cond in targets:
            output_vars += cond().outputs
        # prepare loss_functions.
        if isinstance(loss_func, str):
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
        model.compile(
            loss=loss_func,
            optimizer=optimizer,
        )
        # model.train_function = True

        # set initial state of the model.
        if load_weights_from is not None:
            if os.path.exists(load_weights_from): 
                model.load_weights(load_weights_from)
            else:
                raise Warning("File not found - load_weights_from: {}".format(load_weights_from))
        # Set the variables.
        self._model = model
        self._inputs = inputs
        self._constraints = targets
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
                if old.sol is None:
                    ver.append(True)
                else:
                    if all([all(xo==xn) for xo, xn in zip(old.sol, new.sol)]):
                        ver.append(True)
                    else:
                        ver.append(False)
            else:
                ver.append(False)
        return all(ver)

    def __call__(self, *args, **kwargs):
        output = self._model.__call__(*args, **kwargs)
        return output if isinstance(output, list) else [output]

    def save(self, filepath, *args, **kwargs):
        return self._model.save(filepath, *args, **kwargs)

    def save_weights(self, filepath, *args, **kwargs):
        return self._model.save_weights(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        return self._model.load_weights(filepath, *args, **kwargs)

    def summary(self, *args, **kwargs):
        return self._model.summary(*args, **kwargs)

    def train(self,
              x_true,
              y_true,
              weights=None,
              target_weights=None,
              batch_size=2**6,
              epochs=100,
              learning_rate=0.001,
              shuffle=True,
              callbacks=None,
              stop_after=None,
              save_weights_to=None,
              save_weights_freq=0,
              default_zero_weight=0.0,
              **kwargs):
        """Performs the training on the model.

        # Arguments
            x_true: list of `Xs` associated to targets of `Y`.
                Expecting a list of np.ndarray of size (N,1) each,
                with N as the sample size.
            y_true: list of true `Ys` associated to the targets defined during model setup.
                Expecting the same size as list of targets defined in `SciModel`.
                - To impose the targets at specific `Xs` only, pass a tuple of `(ids, y_true)` for that target.
            weights: (np.ndarray) A global sample weight to be applied to samples.
                Expecting an array of shape (N,1), with N as the sample size.
                Default value is `one` to consider all samples equally important.
            target_weights: (list) A weight for each target defined in `y_true`.
            batch_size: (Integer) or 'None'.
                Number of samples per gradient update.
                If unspecified, 'batch_size' will default to 2^6=64.
            epochs: (Integer) Number of epochs to train the model.
                Defaulted to 100.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
            learning_rate: (Tuple/List) (epochs, lrs).
                Expects a list/tuple with a list of epochs and a list or learning rates.
                It linearly interpolates between entries.
                Defaulted to 0.001 with no decay.
                Example:
                    learning_rate = ([0, 100, 1000], [0.001, 0.0005, 0.00001])
            shuffle: Boolean (whether to shuffle the training data).
                Default value is True.
            callbacks: List of `keras.callbacks.Callback` instances.
            stop_after: To stop after certain missed epochs.
                Defaulted to epochs.
            save_weights_to: (file_path) If you want to save the state of the model (at the end of the training).
            save_weights_freq: (Integer) Save weights every N epcohs.
                Defaulted to 0.
            default_zero_weight: a small number for zero sample-weight.

        # Returns
            A Keras 'History' object after performing fitting.
        """
        if callbacks is None:
            # default_lr = learning_rate/0.9900
            # f0 = np.log(1.0/0.9900 - 1.0)
            # f1 = np.log(1.0/decay_max - 1.0)
            # decay_epochs = epochs if decay_epochs is None else decay_epochs
            # a0 = decay_epochs / (f1-f0)
            # n0 = -a0*f0
            callbacks = []
            if isinstance(learning_rate, (type(None), float, int)):
                lr_rates = 0.001 if learning_rate is None else learning_rate
                K.set_value(self.model.optimizer.lr, lr_rates)
                callbacks.append(
                    k.callbacks.callbacks.ReduceLROnPlateau(
                        monitor='loss', factor=0.5,
                        patience=epochs/10, cooldown=epochs/10,
                        verbose=1, mode='auto', min_delta=0.001,
                        min_lr=1e-3*lr_rates
                    )
                )
            elif isinstance(learning_rate, (tuple, list)):
                lr_epochs = learning_rate[0]
                lr_rates = learning_rate[1]
                callbacks.append(
                    # k.callbacks.LearningRateScheduler(lambda n: default_lr/(1.0 + np.exp((n-n0)/a0))),
                    k.callbacks.LearningRateScheduler(lambda n: np.interp(n, lr_epochs, lr_rates))
                )
            else:
                raise ValueError(
                    "learning rate: expecting a `float` or a tuple/list of two arrays"
                    " with `epochs` and `learning rates`"
                )
            if stop_after is None:
                stop_after = max([10, epochs/10])
            callbacks += [
                k.callbacks.EarlyStopping(monitor="loss", mode='auto', verbose=1,
                                          patience=stop_after, min_delta=1.0e-12),
                k.callbacks.TerminateOnNaN()
            ]
        # prepare X,Y data.
        x_true = to_list(x_true)
        for i, (x, xt) in enumerate(zip(x_true, self._model.inputs)):
            x_shape = tuple(xt.get_shape().as_list())
            if x.shape != x_shape:
                try:
                    x_true[i] = x.reshape((-1,) + x_shape[1:])
                except:
                    print(
                        'Could not automatically convert the inputs to be ' 
                        'of the same size as the expected input tensors. ' 
                        'Please provide inputs of the same dimension as the `Variables`. '
                    )
                    assert False

        num_sample = x_true[0].shape[0]
        assert all([x.shape[0]==num_sample for x in x_true[1:]]), \
            'Inconsistent sample size among `Xs`. '
        ids_all = np.arange(0, num_sample)

        if weights is None:
            weights = np.ones(num_sample)
        else:
            if len(weights.shape)!=1 or \
                    weights.shape[0] != num_sample:
                try:
                    weights = weights.reshape(num_sample)
                except:
                    raise ValueError(
                        'Input error: `weights` should have dimension 1 with '
                        'the same sample length as `Xs. '
                    )

        y_true = to_list(y_true)
        assert len(y_true)==len(self._constraints), \
            'Miss-match between expected targets (constraints) defined in `SciModel` and ' \
            'the provided `y_true`s - expecting the same number of data points. '

        sample_weights, y_star = [], []
        for i, yt in enumerate(y_true):
            c = self._constraints[i]
            # verify entry.
            ys, wei = SciModel._prepare_data(
                c.cond.outputs, to_list(yt),
                weights, num_sample, default_zero_weight
            )
            # add to the list.
            y_star += ys
            sample_weights += wei

        if target_weights is not None:
            if not(isinstance(target_weights, list) and
                   len(target_weights) == len(y_true)):
                raise ValueError(
                    'Expected a list of weights for the same size as the targets '
                    '- was provided {}'.format(target_weights)
                )
            else:
                for i, cw in enumerate(target_weights):
                    sample_weights[i] *= cw
        # save model.
        model_file_path = None
        if save_weights_to is not None:
            try:
                self._model.save_weights("{}-start.hdf5".format(save_weights_to))
                model_file_path = save_weights_to + "-{epoch:05d}-{loss:.3e}.hdf5"
                model_check_point = k.callbacks.callbacks.ModelCheckpoint(
                    model_file_path, monitor='loss', save_weights_only=True, mode='auto',
                    period=10 if save_weights_freq==0 else save_weights_freq,
                    save_best_only=True if save_weights_freq==0 else False
                )
            except:
                print("\nWARNING: Failed to save model.weights to the provided path: {}\n".format(save_weights_to))
        if model_file_path is not None:
            callbacks.append(model_check_point)
        # perform the training.
        history = self._model.fit(
            x_true, y_star,
            sample_weight=sample_weights,  #sums to number of samples.
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            callbacks=callbacks,
            **kwargs
        )
        if save_weights_to is not None:
            try:
                self._model.save_weights("{}-end.hdf5".format(save_weights_to))
            except:
                print("\nWARNING: Failed to save model.weights to the provided path: {}\n".format(save_weights_to))
        # return the history.
        return history

    def predict(self, xs,
                batch_size=None,
                verbose=0,
                steps=None):
        """ Predict output from network.

        # Arguments
            xs: list of `Xs` associated model.
                Expecting a list of np.ndarray of size (N,1) each,
                with N as the sample size.
            batch_size: defaulted to None.
                Check Keras documentation for more information.
            verbose: defaulted to 0 (None).
                Check Keras documentation for more information.
            steps: defaulted to 0 (None).
                Check Keras documentation for more information.

        # Returns
            List of numpy array of the size of network outputs.

        # Raises
            ValueError if number of `xs`s is different from number of `inputs`.
        """
        xs = to_list(xs)
        if len(xs) != len(self._inputs):
            raise ValueError(
                "Please provide consistent number of inputs as the model is defined: "
                "Expected {} - provided {}".format(len(self._inputs), len(to_list(xs)))
            )
        # To have unified output for postprocessing - limitted support.
        shape_default = xs[0].shape if all([x.shape==xs[0].shape for x in xs]) else None
        # prepare X,Y data.
        for i, (x, xt) in enumerate(zip(xs, self._model.inputs)):
            x_shape = tuple(xt.get_shape().as_list())
            if x.shape != x_shape:
                try:
                    xs[i] = x.reshape((-1,) + x_shape[1:])
                except:
                    print(
                        'Could not automatically convert the inputs to be ' 
                        'of the same size as the expected input tensors. ' 
                        'Please provide inputs of the same dimension as the `Variables`. '
                    )
                    assert False

        y_pred = self._model.predict(xs, batch_size, verbose, steps)

        if shape_default is not None:
            try:
                y_pred = [y.reshape(shape_default) for y in y_pred]
            except:
                print("Input and output dimensions need re-adjustment for post-processing.")

        return unpack_singleton(y_pred)

    def eval(self, *args):
        if len(args) == 1:
            x_data = to_list(args[0])
            if len(x_data) != len(self._inputs):
                raise ValueError(
                    "Please provide consistent number of inputs as the model is defined: "
                    "Expected {} - provided {}".format(len(self._inputs), len(x_data))
                )
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
            - "mse" for `Mean Squared Error` or
            - "mae" for `Mean Absolute Error` or
            - "se" for `Squared Error` or
            - "ae" for `Absolute Error`.

        # Returns
            Callable function that gets (y_true, y_pred) as the input and
                returns the loss value as the output.

        # Raises
            ValueError if anything other than "mse" or "mae" is passed.
        """
        if method in ("mse", "mean_squared_error"):
            return lambda y_true, y_pred: K.mean(K.square(y_true - y_pred), axis=-1)
        elif method in ("mae", "mean_absolute_error"):
            return lambda y_true, y_pred: K.mean(K.abs(y_true - y_pred), axis=-1)
        elif method in ("se", "squared_error"):
            return lambda y_true, y_pred: K.sum(K.square(y_true - y_pred), axis=-1)
        elif method in ("ae", "absolute_error"):
            return lambda y_true, y_pred: K.sum(K.abs(y_true - y_pred), axis=-1)
        elif hasattr(k.losses, method):
            return getattr(k.losses, method)
        else:
            raise ValueError(
                'Supported losses: Keras loss function or (mse, mae, se, ae)'
            )


    @staticmethod
    def _prepare_data(cond_outputs, y_true, global_weights, num_sample, default_zero_weight):
        ys, weis = [], []
        ids_all = np.arange(0, num_sample)
        # prepare sample weight.
        for i, yt in enumerate(to_list(y_true)):
            ids = None
            yc = cond_outputs[i]
            if isinstance(yt, tuple) and len(yt) == 2:
                ids = yt[0].flatten()
                if isinstance(yt[1], np.ndarray):
                    adjusted_yt = yt[1]
                    if ids.size == yt[1].shape[0] and ids.size < num_sample:
                        adjusted_yt = np.zeros((num_sample,)+yt[1].shape[1:])
                        adjusted_yt[ids, :] = yt[1]
                    elif yt[1].shape[0] != num_sample:
                        raise ValueError(
                            'Error in size of the target {}.'.format(i)
                        )
                else:
                    adjusted_yt = yt[1]
                ys.append(adjusted_yt)
            elif isinstance(yt, (np.ndarray, str, float, int, type(None))):
                ys.append(yt)
            else:
                raise ValueError(
                    'Unrecognized entry - please provide a list of `data` or tuples of `(ids, data)`'
                    ' for each target defined in `SciModel`. '
                )
            # Define weights of samples.
            if ids is None:
                ids = ids_all
                wei = global_weights
            else:
                wei = np.zeros(num_sample) + default_zero_weight
                wei[ids] = global_weights[ids]
                wei[ids] *= sum(global_weights)/sum(wei[ids])
            weis.append(wei)
            # preparing targets.
            if isinstance(ys[-1], np.ndarray):
                if not (ys[-1].shape[1:] == k.backend.int_shape(yc)[1:]):
                    try:
                        ys[-1] = ys[-1].reshape((-1,) + k.backend.int_shape(yc)[1:])
                    except (ValueError, TypeError):
                        raise ValueError(
                            'Dimension of expected `y_true` does not match with defined `Constraint`'
                        )
            elif isinstance(ys[-1], str):
                if ys[-1] == 'zero' or ys[-1] == 'zeros':
                    ys[-1] = np.zeros((num_sample, ) + k.backend.int_shape(yc)[1:])
                elif ys[-1] == 'one' or ys[-1] == 'ones':
                    ys[-1] = np.ones((num_sample, ) + k.backend.int_shape(yc)[1:])
                else:
                    raise ValueError(
                        'Unexpected `str` entry - only accepts `zeros` or `ones`.'
                    )
            elif isinstance(ys[-1], (int, float)):
                ys[-1] = np.ones((num_sample, ) + k.backend.int_shape(yc)[1:]) * float(ys[-1])
            elif isinstance(ys[-1], type(None)):
                ys[-1] = np.zeros((num_sample, ) + k.backend.int_shape(yc)[1:])
            else:
                raise ValueError(
                    'Unsupported entry - {} '.format(ys[-1])
                )
            # set undefined ids to zeros.
            if ids.size != num_sample:
                adjusted_ids = np.ones(num_sample, dtype=bool)
                adjusted_ids[ids] = False
                ys[-1][adjusted_ids, :] = 0.0

        return ys, weis

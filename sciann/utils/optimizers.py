# Scipy wrapper for keras
# originally developed by:
#   https://github.com/rohitgr7/keras-opt
#   addopted by Ehsan Haghighat

import numpy as np
from scipy.optimize import minimize
from tensorflow import keras
from tensorflow.keras import backend as K  # pylint: disable=import-error
from tensorflow.python.keras.callbacks import BaseLogger, CallbackList, History  # pylint: disable=no-name-in-module
from tensorflow.python.keras.optimizer_v2 import optimizer_v2   # pylint: disable=no-name-in-module
from tensorflow.python.ops import variables as tf_variable

from tensorflow.python.keras.utils.generic_utils import Progbar

# from tqdm import trange, tqdm_notebook


class GradientObserver(optimizer_v2.OptimizerV2):
    """
    Implements the Keras Optimizer interface in order to accumulate gradients for
    each mini batch. Gradients are then read at the end of the epoch by the ScipyOptimizer. 
    """

    def __init__(self, learning_rate=0.001, method='L-BFGS-B', **kwargs):
        super(GradientObserver, self).__init__('GradientObserver')
        self._learning_rate = tf_variable.Variable(kwargs.get('lr', learning_rate), 'float32')
        self._method = method.lower().split("scipy-")[-1]
        self._vars = []
        self._grads = {}

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def lr(self):
        return self._learning_rate

    @property
    def method(self):
        return self._method

    def get_updates(self, loss, params):
        """
        Build the graph nodes that accumulate gradients.
        """
        self.updates = []
        grads = self.get_gradients(loss, params)
        for param, grad in zip(params, grads):
            shape = K.int_shape(param)
            var = K.zeros(shape)
            self._vars.append(var)
            self.updates.append(K.update_add(var, grad))
        return self.updates

    def get_gradient_values(self):
        """
        Read gradient values (at epoch end).
        """
        values = []
        for g in self._grads.values():
            values.append(K.eval(g))
        for v in self._vars:
            values.append(K.eval(v))
        return values

    def clear(self):
        """
        Clear gradient values (used at epoch start)
        """
        self._grads = {}
        for var in self._vars:
            K.set_value(var, np.zeros(var.shape))

    def _create_slots(self, var_list):
        pass

    def _resource_apply_dense(self, grad, var, **apply_kwargs):
        self._grads[var.name] = grad

    def _resource_apply_sparse(self, grad, handle, indices, apply_state):
        if handle.name in self._grads:
            dense_grad = self._grads[handle.name]
        else:
            dense_grad = np.zeros(handle.shape.as_list())
            self._grads[handle.name] = dense_grad

        for i, idx in enumerate(indices.numpy()):
            dense_grad[idx] += grad[i]

    def get_config(self):
        config = super(GradientObserver, self).get_config()
        return config


class GeneratorWrapper(keras.utils.Sequence):
    """
    Converts fit() into fit_generator() interface.
    """

    def __init__(self, inputs, outputs, sample_weights, batch_size, shuffle):
        self._inputs = inputs
        self._outputs = outputs
        self._sample_weights = sample_weights
        self._size = inputs[0].shape[0]
        self._batch_size = batch_size
        self._num_batches = int((self._size-1)/batch_size) + 1
        self._shuffle = shuffle
        self._ids = np.arange(0, self._size)
        print("\nTotal samples: {} ".format(self._size))
        print("Batch size: {} ".format(min(self._batch_size, self._size)))
        print("Total batches: {} \n".format(self._num_batches))

    def __len__(self):
        return self._num_batches

    def __getitem__(self, index):
        if self._num_batches > 1 and index == 0 and self._shuffle:
            self._ids = np.random.choice(self._size, self._size, replace=False)
        start = index * self._batch_size
        end = min(start + self._batch_size, self._size)
        ids = self._ids[start: end]
        inputs = [v[ids, :] for v in self._inputs]
        outputs = [v[ids, :] for v in self._outputs]
        sample_weights = [v[ids] for v in self._sample_weights]
        return inputs, outputs, sample_weights


class ScipyOptimizer(object):
    """
    Invokes the underlying model in order to obtain the cost and gradients for the function
    being optimized.
    """

    def __init__(self, model):
        self._model = model
        self._layers = [layer for layer in model._layers if layer.weights]
        self._weights_size = 0
        for layer in self._layers:
            for w in layer.weights:
                if not w.trainable:
                    continue
                self._weights_size += np.prod(w.shape)

    def _update_weights(self, x):
        x_offset = 0
        for layer in self._layers:
            w_list = []
            w_trainable = [w.trainable for w in layer.weights]
            batch_update = False not in w_trainable
            for w in layer.weights:
                if not w.trainable:
                    continue
                shape = w.get_shape()
                w_size = np.prod(shape)
                value = np.array(x[x_offset:x_offset+w_size]).reshape(shape)
                if batch_update:
                    w_list.append(value)
                else:
                    K.set_value(w, value)
                x_offset += w_size
            if batch_update:
                layer.set_weights(w_list)
        assert x_offset == self._weights_size

    def _collect_weights(self):
        x_values = np.empty(self._weights_size)
        x_offset = 0
        for layer in self._layers:
            w_trainable = [w.trainable for w in layer.weights]
            for var, trainable in zip(layer.get_weights(), w_trainable):
                if not trainable:
                    continue
                w_size = var.size
                x_values[x_offset:x_offset+w_size] = var.reshape(-1)
                x_offset += w_size
        assert x_offset == self._weights_size
        return x_values

    def _fun_generator(self, x, generator, state):
        self._model.optimizer.clear()
        self._update_weights(x)
        callbacks = state['callbacks']

        if not state['in_epoch']:
            callbacks.on_epoch_begin(state['epoch'])
            state['in_epoch'] = True

        cost_sum = 0
        iterator = range(len(generator))
        if state['verbose'] == 1 and len(generator) > 1:
            # iterator = trange(len(generator))
            progress_bar = Progbar(len(generator))
        else:
            # iterator = range(len(generator))
            progress_bar = False

        state['epoch_logs'] = {}
        epoch_logs = state['epoch_logs']

        for batch_index in iterator:
            inputs, outputs, sample_weights = generator[batch_index]
            if isinstance(inputs, list):
                isize = inputs[0].shape[0]
            else:
                isize = inputs.shape[0]
            batch_logs = {'batch': batch_index, 'size': isize}
            callbacks.on_batch_begin(batch_index, batch_logs)
            outs = self._model.train_on_batch(inputs, outputs, sample_weights)
            if not isinstance(outs, list):
                outs = [outs]
            for lbl, v in zip(self._model.metrics_names, outs):
                batch_logs[lbl] = v
                epoch_logs[lbl] = epoch_logs.get(lbl, 0.0) + v
            callbacks.on_batch_end(batch_index, batch_logs)
            batch_cost = batch_logs['loss']
            if progress_bar:
                progress_bar.update(batch_index + 1)
                # iterator.set_postfix(cost=batch_cost)

            cost_sum += batch_cost

        generator.on_epoch_end()

        epoch_logs = state['epoch_logs']
        if state['verbose'] > 0:
            print('itr:', state['epoch'],
                  ', '.join(['{0}: {1:.4e}'.format(k, v) for k, v in epoch_logs.items()]))
        elif state['verbose'] < 0:
            print('itr:', state['epoch'], ', loss: {0:.6e}'.format(epoch_logs['loss']))

        # average the metrics
        for lbl in self._model.metrics_names:
            epoch_logs[lbl] = epoch_logs.get(lbl) / len(iterator)

        cost = cost_sum / len(iterator)

        gradients = self._model.optimizer.get_gradient_values()
        x_grad = np.empty(x.shape)
        x_offset = 0
        for grad in gradients:
            w_size = grad.size
            x_grad[x_offset:x_offset + w_size] = grad.reshape(-1)
            x_offset += w_size
        assert x_offset == self._weights_size
        self._cost = cost
        self._gradients = x_grad
        return cost, x_grad

    def _validate(self, x, val_generator, state):
        # TODO: weight update should be optimized in the most common case
        self._update_weights(x)
        # test callback are in the newer version of the CallbackList API.
        # callbacks = state['callbacks']
        epoch_logs = state['epoch_logs']

        # callbacks.on_test_begin()

        val_outs = [0] * len(self._model.metrics_names)
        n_steps = len(val_generator)
        for batch_index in range(n_steps):
            inputs, outputs = val_generator[batch_index]
            batch_logs = {'batch': batch_index, 'size': inputs.shape[0]}

            # callbacks.on_test_batch_begin(batch_index, batch_logs)
            batch_outs = self._model.test_on_batch(inputs, outputs)
            if not isinstance(batch_outs, list):
                batch_outs = [batch_outs]
            for l, o in zip(self._model.metrics_names, batch_outs):
                batch_logs[l] = o
            # callbacks.on_test_batch_end(batch_index, batch_logs)
            for i, batch_out in enumerate(batch_outs):
                val_outs[i] += batch_out

        for l, o in zip(self._model.metrics_names, val_outs):
            o /= n_steps
            epoch_logs['val_' + l] = o

        # callbacks.on_test_end()

    def fit(self, inputs, outputs, sample_weight, batch_size=32, shuffle=True, **kwargs):
        return self.fit_generator(GeneratorWrapper(inputs, outputs, sample_weight, batch_size, shuffle), **kwargs)

    def fit_generator(self, generator, epochs=1,
                      validation_data=None,
                      callbacks=None,
                      verbose=True):
        method = self._model.optimizer.method
        x0 = self._collect_weights()
        history = History()
        _callbacks = [BaseLogger(stateful_metrics=self._model.metrics_names)]
        _callbacks += (callbacks or []) + [history]
        callback_list = CallbackList(_callbacks)
        callback_list.set_model(self._model)
        callback_list.set_params({
            'epochs': epochs,
            'verbose': False,
            'metrics': list(self._model.metrics_names),
        })
        state = {
            'epoch': 0,
            'verbose': verbose,
            'callbacks': callback_list,
            'in_epoch': False,
            'epoch_logs': {},
        }
        min_options = {
            'maxiter': epochs,
        }

        val_generator = None
        if validation_data is not None:
            if isinstance(validation_data, keras.utils.Sequence):
                val_generator = validation_data
            elif isinstance(validation_data, tuple) and len(validation_data) == 2:
                val_generator = GeneratorWrapper(*validation_data)

        def on_iteration_end(xk):
            cb = state['callbacks']
            if val_generator is not None:
                self._validate(xk, val_generator, state)
            cb.on_epoch_end(state['epoch'], state['epoch_logs'])
            # if state['verbose']:
            #     epoch_logs = state['epoch_logs']
            #     print('epoch: ', state['epoch'],
            #           ', '.join([' {0}: {1:.3e}'.format(k, v) for k, v in epoch_logs.items()]))
            state['epoch'] += 1
            state['in_epoch'] = False
            state['epoch_logs'] = {}

        callback_list.on_train_begin()
        result = minimize(
            self._fun_generator, x0, method=method, jac=True, options=min_options,
            callback=on_iteration_end, args=(generator, state))
        self._update_weights(result['x'])
        callback_list.on_train_end()
        return history

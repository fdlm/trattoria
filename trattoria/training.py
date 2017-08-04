from __future__ import print_function

import operator
import time
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor
from tqdm import tqdm

from trattoria.outputs import ConsoleLog


class Status:
    FINISHED = 0
    ERROR = 1
    STOPPED_EARLY = 2


class StopTraining(Exception):

    def __init__(self, status, message):
        self.status = status
        self.message = message


class MonitoredUpdater(object):

    def __init__(self, updater, **updater_params):
        self.updater = updater
        self.updater_params = updater_params
        self.updates = None
        self.init_status = None

    @property
    def status(self):
        if self.updates is None:
            raise ValueError('Updater not created yet!')
        return [shared_var.get_value() for shared_var in self.updates]

    @status.setter
    def status(self, status):
        if self.updates is None:
            raise ValueError('Updater not created yet!')
        if len(status) != len(self.updates):
            raise ValueError('Incompatible status!')
        for shared_var, value in zip(self.updates, status):
            shared_var.set_value(value)

    def __call__(self, loss_or_grads, params):
        self.updates = self.updater(loss_or_grads, params,
                                    **self.updater_params)
        if self.init_status:
            self.status = self.init_status
        return self.updates


class ImprovementTrigger(object):

    def __init__(self, callbacks, observed, compare=operator.lt,
                 no_trigger_callbacks=None):
        self.observed = observed
        self.cmp = compare
        self.callbacks = callbacks
        self.no_trigger_callbacks = no_trigger_callbacks or []
        self.best_value = None

    def __call__(self, epoch, epoch_results):
        first_call = self.best_value is None
        improvement = self.cmp(epoch_results[self.observed], self.best_value)
        if first_call or improvement:
            self.best_value = epoch_results[self.observed]
            for cb in self.callbacks:
                cb(epoch, epoch_results)
        else:
            for cb in self.no_trigger_callbacks:
                cb(epoch, epoch_results)


class CountdownTrigger(object):

    def __init__(self, callbacks, patience):
        self.callbacks = callbacks
        self.patience = patience
        self.wait = 0

    def reset(self, *args, **kwargs):
        self.wait = 0

    def __call__(self, epoch, epoch_results):
        self.wait += 1
        if self.wait == self.patience:
            for cb in self.callbacks:
                cb(epoch, epoch_results)


class NanLossTrigger(object):

    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, epoch, epoch_results):
        if np.isnan(epoch_results['loss']):
            for cb in self.callbacks:
                cb(epoch, epoch_results)


def _stop_training_early(*args, **kwargs):
    raise StopTraining(
        Status.STOPPED_EARLY,
        message='Early Stopping.'
    )


def early_stopping(patience, observed, compare=operator.lt):
    countdown = CountdownTrigger(
        callbacks=[_stop_training_early],
        patience=patience
    )
    return ImprovementTrigger(
        callbacks=[countdown.reset],
        observed=observed,
        compare=compare,
        no_trigger_callbacks=[countdown]
    )


def _raise_nan_loss(*args, **kwargs):
    raise StopTraining(
        Status.ERROR,
        message='Encountered NaN Loss.'
    )


def stop_on_nan():
    return NanLossTrigger([_raise_nan_loss])


def iterate(batch_iterator, func, observables):
    vals = OrderedDict((name, 0.0) for name in observables)
    n_iter = 0

    batches = tqdm(batch_iterator, leave=False, ncols=80)

    batch_time = time.time()
    for batch in batches:
        theano_time = time.time()
        batch_objectives = func(*batch)
        theano_time = time.time() - theano_time

        for name, obj_val in batch_objectives.items():
            vals[name] += obj_val
        n_iter += 1

        batch_time = time.time() - batch_time
        perc_theano = int((theano_time / batch_time) * 100)

        if 'loss' in vals:
            batches.set_description(
                'Loss: {:6.4f} (%th: {:3d})'.format(vals['loss'] / n_iter,
                                                    perc_theano)
            )
        batch_time = time.time()

    for name, val in vals.items():
        vals[name] = val / n_iter

    return vals


class Validator(object):

    def __init__(self, net, batches, observables):
        if not isinstance(observables, dict):
            observables = {'loss': observables}
        self.observables = OrderedDict(('val_' + name, obj)
                                       for name, obj in observables.items())
        self.batches = batches

        y_hat_test = net.get_output(deterministic=True)
        y = net.get_output_tensor()
        self.test_fn = theano.function(
            inputs=net.get_inputs() + [y],
            outputs={name: obj(y_hat_test, y)
                     for name, obj in self.observables.items()}
        )

    def __call__(self):
        return iterate(self.batches, self.test_fn, self.observables)


def train(net, train_batches, num_epochs, observables,
          updater, regularizers=None, validator=None, logs=None,
          callbacks=None, init_epoch=0, **tags):

    if not isinstance(observables, dict):
        observables = {'loss': observables}

    if 'loss' not in observables:
        raise ValueError('Need definition of loss in objectives for training!')

    if regularizers is None:
        regularizers = []

    if callbacks is None:
        callbacks = []

    if logs is None:
        logs = [ConsoleLog()]

    y = net.get_output_tensor()
    y_hat = net.get_output()
    loss = observables['loss'](y_hat, y)
    params = net.get_params(trainable=True, **tags)
    updates = updater(sum(regularizers, loss), params)

    train_fn = theano.function(
        inputs=net.get_inputs() + [y],
        outputs={name: calc(y_hat, y) for name, calc in observables.items()},
        updates=updates
    )

    for log in logs:
        log.start(observables, validator.observables if validator else None)

    try:
        epochs = range(init_epoch, init_epoch + num_epochs)
        for epoch in tqdm(epochs, ncols=80, leave=False):
            observed = iterate(train_batches, train_fn, observables)
            if validator:
                observed.update(validator())

            for log in logs:
                log.add(epoch, observed)

            for callback in callbacks:
                callback(epoch, observed)

    except StopTraining as st:
        return st.status

    return Status.FINISHED


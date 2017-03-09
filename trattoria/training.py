from __future__ import print_function

from collections import OrderedDict

import theano
import theano.tensor
from tqdm import tqdm

from trattoria.outputs import ConsoleLog


class StopTraining(Exception):
    pass


def iterate(batch_iterator, func, observables):
    vals = OrderedDict((name, 0.0) for name in observables)
    n_iter = 0

    batches = tqdm(batch_iterator, leave=False)
    for batch in batches:
        batch_objectives = func(*batch)
        for name, obj_val in batch_objectives.items():
            vals[name] += obj_val
        n_iter += 1

        if 'loss' in vals:
            batches.set_description(
                'Loss: {:6.4f}'.format(vals['loss'] / n_iter))

    for name, val in vals.items():
        vals[name] = val / n_iter

    return vals


def _tensor(shape, dtype, name):
    return theano.tensor.TensorType(
        dtype, broadcastable=[False] * (len(shape) + 1))(name)


class Validator(object):

    def __init__(self, net, batches, observables):
        self.observables = OrderedDict(('val_' + name, obj)
                                       for name, obj in observables.items())
        self.batches = batches

        y_hat_test = net.get_outputs(deterministic=True)
        y = _tensor(batches.tshape, batches.ttype, 'y')
        self.test_fn = theano.function(
            inputs=net.get_inputs() + [y],
            outputs={name: obj(y_hat_test, y)
                     for name, obj in self.observables.items()}
        )

    def __call__(self):
        return iterate(self.batches, self.test_fn, self.observables)


def train(net, train_batches, num_epochs, observables,
          updater, validator=None, logs=None, callbacks=None, **tags):

    if not isinstance(observables, dict):
        observables = {'loss': observables}

    if 'loss' not in observables:
        raise ValueError('Need definition of loss in objectives for training!')

    if callbacks is None:
        callbacks = []

    if logs is None:
        logs = [ConsoleLog()]

    y = _tensor(train_batches.tshape, train_batches.ttype, 'y')
    y_hat = net.get_outputs()
    loss = observables['loss'](y_hat, y)
    params = net.get_params(**tags)
    updates = updater(loss, params)

    train_fn = theano.function(
        inputs=net.get_inputs() + [y],
        outputs={name: calc(y_hat, y) for name, calc in observables.items()},
        updates=updates
    )

    for log in logs:
        log.start(observables, validator.observables if validator else None)

    try:
        for epoch in tqdm(range(num_epochs), leave=False):
            observed = iterate(train_batches, train_fn, observables)
            if validator:
                observed.update(validator())

            for log in logs:
                log.add(epoch, observed)

            for callback in callbacks:
                callback(epoch, observed)

    except StopTraining:
        pass


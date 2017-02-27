from __future__ import print_function

from collections import OrderedDict

import lasagne as lnn
import theano
from tqdm import tqdm

from trattoria.outputs import ConsoleLog


class StopTraining(Exception):
    pass


def iterate(batch_iterator, func, objectives):
    vals = OrderedDict((name, 0.0) for name in objectives)
    n_iter = 0

    batches = tqdm(batch_iterator, leave=False)
    for batch in batches:
        batch_objectives = func(*batch)
        for name, obj_val in batch_objectives:
            vals[name] += obj_val
        n_iter += 1

        if 'loss' in vals:
            batches.set_description(
                'Loss: {:6.4f}'.format(vals['loss'] / n_iter))

    for name, val in vals:
        vals[name] = val / n_iter

    return vals


class Validator(object):

    def __init__(self, net, batches, objectives):
        self.objectives = OrderedDict(('val.' + name, obj)
                                      for name, obj in objectives.items())
        self.batches = batches

        y_hat_test = net.get_outputs(deterministic=True)
        y = batches.target_tensor_type('y')
        self.test_fn = theano.function(
            inputs=net.get_inputs(),
            outputs={name: obj(y_hat_test, y)
                     for name, obj in objectives.items()}
        )

    def __call__(self):
        return iterate(self.batches, self.test_fn, self.objectives)


def train(net, train_batches, num_epochs, objectives,
          updater, validator=None, callbacks=None):

    if callbacks is None:
        callbacks = []

    if not isinstance(objectives, dict):
        objectives = {'loss': objectives}

    if 'loss' not in objectives:
        raise ValueError('Need definition of loss in objectives for training!')

    y = train_batches.target_tensor_type('y')
    y_hat = net.get_outputs()
    loss = objectives['loss'](y_hat, y)
    params = lnn.layers.get_all_params(net)
    updates = updater(params, loss)

    train_fn = theano.function(
        inputs=net.get_inputs(),
        outputs={name: obj(y_hat, y) for name, obj in objectives.items()},
        updates=updates
    )

    console_log = ConsoleLog(
        objectives, validator.objectives if validator else None
    )
    console_log.write_header()

    try:
        for epoch in tqdm(range(num_epochs), leave=False):
            epoch_results = iterate(train_batches, train_fn, objectives)
            if validator:
                epoch_results.update(validator())
            console_log.write_row(epoch, epoch_results)

            for callback in callbacks:
                callback(epoch, epoch_results)
    except StopTraining:
        pass


from __future__ import print_function

import lasagne as lnn
import numpy as np
import tqdm
import theano
from collections import OrderedDict


class ConsoleLog(object):

    def __init__(self, train_objectives, val_objectives=None):
        if val_objectives is None:
            val_objectives = []

        self.log_val_names = ['loss']
        train_row_fmts = ['{loss:>15.6f}']
        header_fmts = ['{:>15s}']

        for name in train_objectives:
            if name == 'loss':
                continue
            self.log_val_names.append(name)
            name_len = max(15, len(name))
            header_fmts.append('{:>%ds}' % name_len)
            train_row_fmts.append('{%s:>%d.6f}' % (name, name_len))

        val_row_fmts = []
        if 'loss' in val_objectives:
            self.log_val_names.append('val_loss')
            val_row_fmts.append('{loss:>15.6f}')
            header_fmts.append('{:>15s}')

        for name in val_objectives:
            if name == 'loss':
                continue
            caption = 'val_' + name
            self.log_val_names.append(caption)
            cap_len = max(15, len(caption))
            header_fmts.append('{:>%ds}' % cap_len)
            val_row_fmts.append('{%s:>%d.6f}' % (name, cap_len))

        self.header_fmt = ''.join(header_fmts)
        self.train_row_fmt = ''.join(train_row_fmts)
        self.val_row_fmt = ''.join(val_row_fmts)

    def header(self):
        return self.header_fmt.format(*self.log_val_names)

    def row(self, train_objectives, val_objectives=None):
        train_row = self.train_row_fmt.format(**train_objectives)
        if val_objectives:
            val_row = self.val_row_fmt.format(**val_objectives)
        else:
            val_row = ''
        return train_row + val_row


def train(net, train_batches, num_epochs, objectives, l1, l2,
          updater, val_batches=None, val_objectives=None, patience=20):

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

    if val_batches is not None:
        if not val_objectives:
            raise ValueError('Need validation objectives for validation')

        y_hat_test = net.get_outputs(deterministic=True)
        test_fn = theano.function(
            inputs=net.get_inputs(),
            outputs={name: obj(y_hat_test, y)
                     for name, obj in val_objectives.items()}
        )
    else:
        test_fn = None

    def iterate(batch_iterator, func, objectives):
        vals = OrderedDict((name, 0.0) for name in objectives)
        n_iter = 0

        batches = tqdm.tqdm(batch_iterator, leave=False)
        for batch in batches:
            batch_objectives = func(*batch)
            for name, obj_val in batch_objectives:
                vals[name] += obj_val
            n_iter += 1

            if 'loss' in vals:
                batches.set_description(
                    'Loss: {:6.4f}'.format(vals['loss'] / n_iter))

        return tuple(v / n_iter for v in vals)

    console_log = ConsoleLog(objectives, val_objectives)
    print(console_log.header())

    for epoch in tqdm.tqdm(range(num_epochs), leave=False):
        train_vals = iterate(train_batches, train_fn, objectives)
        if val_batches:
            val_vals = iterate(val_batches, test_fn, val_objectives)
        else:
            val_vals = None
        tqdm.tqdm.write(console_log.row(train_vals, val_vals))

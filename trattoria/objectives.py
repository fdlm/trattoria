import lasagne
import theano.tensor as tt
from functools import partial


def average(function, mask=None):
    if mask is None:
        return lambda pred, targ: function(pred, targ).mean()
    else:
        return lambda pred, targ: lasagne.objectives.aggregate(
            function(pred, targ), mask, mode='normalized_sum'
        )


def average_categorical_crossentropy(predictions, targets, mask=None, eta=1e-7):
    func = average(lasagne.objectives.categorical_crossentropy, mask)
    return func(tt.clip(predictions, eta, 1 - eta), targets)


def average_categorical_accuracy(predictions, targets, mask=None):
    def acc(pred, targ):
        return tt.cast(
            lasagne.objectives.categorical_accuracy(pred, targ),
            'floatX'
        )
    func = average(acc, mask)
    return func(predictions, targets)


def average_binary_crossentropy(predictions, targets, mask=None, eta=1e-7):
    func = average(lasagne.objectives.binary_crossentropy, mask)
    return func(tt.clip(predictions, eta, 1 - eta), targets)


def average_binary_accuracy(predictions, targets, mask=None):
    def acc(pred, targ):
        return tt.cast(
            lasagne.objectives.binary_accuracy(pred, targ),
            'floatX'
        )
    func = average(acc, mask)
    return func(predictions, targets)


def average_squared_error(predictions, targets, mask=None):
    func = average(lasagne.objectives.squared_error, mask)
    return func(predictions, targets)


def masked(loss_func, mask):
    return partial(loss_func, mask=mask)

import lasagne
from functools import partial


def average(function, mask=None):
    if mask is None:
        return lambda pred, targ: function(pred, targ).mean()
    else:
        return lambda pred, targ: lasagne.objectives.aggregate(
            function(pred, targ), mask, mode='normalized_sum'
        )


def average_categorical_crossentropy(predictions, targets, mask=None):
    func = average(lasagne.objectives.categorical_crossentropy, mask)
    return func(predictions, targets)


def average_categorical_accuracy(predictions, targets, mask=None):
    func = average(lasagne.objectives.categorical_accuracy, mask)
    return func(predictions, targets)


def masked(loss_func, mask):
    return partial(loss_func, mask=mask)

import lasagne


def average(function):
    return lambda predictions, targets: function(predictions, targets).mean()


def average_categorical_crossentropy(predictions, targets):
    func = average(lasagne.objectives.categorical_crossentropy)
    return func(predictions, targets)


def average_categorical_accuracy(predictions, targets):
    func = average(lasagne.objectives.categorical_accuracy)
    return func(predictions, targets)

import operator


class ValueScheduler(object):
    """
    Changes to a shared variable according to a pre-defined schedule.

    Parameters
    ----------
    variable : theano shared variable
        Variable to be changed.
    schedule : dict
        Schedule dictionary. Keys are epochs after which changes should
        be applied, values are values to be set.
    """

    def __init__(self, variable, schedule):
        self.schedule = schedule
        self.variable = variable

    def __call__(self, epoch, observed):
        """
        Update the shared variable if an update is scheduled at the
        given epoch.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        observed : dict, unused
            Dictionary of observed values.

        """
        if epoch in self.schedule:
            self.variable.set_value(self.schedule[epoch])


class Linear(object):

    def __init__(self, variable, start_epoch, end_epoch, target_value):
        self.variable = variable
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.target_value = target_value
        self.shift = None

    def __call__(self, epoch, observed):
        if self.start_epoch <= epoch < self.end_epoch:
            if self.shift is None:
                start_value = self.variable.get_value()
                self.shift = ((self.target_value - start_value) /
                              (self.end_epoch - self.start_epoch))
            self.variable.set_value(self.variable.get_value() + self.shift)


class PatienceMult(object):

    def __init__(self, variable, factor, observe, patience,
                 compare=operator.lt):
        self.variable = variable
        self.factor = factor
        self.observe = observe
        self.patience = patience
        self.cmp = compare
        self.wait = 0
        self.best_value = None

    def __call__(self, epoch, observed):
        first_call = self.best_value is None
        if first_call or self.cmp(observed[self.observe], self.best_value):
            self.wait = 0
            self.best_value = observed[self.observe]
        elif self.wait >= self.patience - 1:
            self.wait = 0
            self.best_value = None
            self.variable.set_value(self.variable.get_value() * self.factor)
        else:
            self.wait += 1


class WarmRestart(object):

    def __init__(self, net, observe, patience, compare=operator.lt):
        self.net = net
        self.observe = observe
        self.cmp = compare
        self.patience = patience
        self.wait = 0
        self.best_value = None
        self.best_params = None

    def __call__(self, epoch, observed):
        first_call = self.best_value is None

        if first_call or self.cmp(observed[self.observe], self.best_value):
            self.wait = 0
            self.best_value = observed[self.observe]
            self.best_params = self.net.get_param_values()
        elif self.wait >= self.patience:
            self.wait = 0
            self.best_value = None
            self.net.set_param_values(self.best_params)
        else:
            self.wait += 1


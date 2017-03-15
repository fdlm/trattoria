import operator


class ValueScheduler(object):

    def __init__(self, variable, schedule):
        self.schedule = schedule
        self.variable = variable

    def __call__(self, epoch, observed):
        if epoch in self.schedule:
            self.variable.set_value(self.schedule[epoch])


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
        elif self.wait >= self.patience:
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


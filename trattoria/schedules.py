class ValueScheduler(object):

    def __init__(self, variable, schedule):
        self.schedule = schedule
        self.variable = variable

    def __call__(self, epoch, observed):
        if epoch in self.schedule:
            self.variable.set_value(self.schedule[epoch])

import pickle
import lasagne as lnn
import theano


class NeuralNetwork(object):

    def __init__(self, net):
        self.net = net
        self._process = None

    def get_outputs(self, **kwargs):
        return lnn.layers.get_output(self.net, **kwargs)

    def get_inputs(self):
        return [l.input_var
                for l in lnn.layers.get_all_layers(self.net)
                if isinstance(l, lnn.layers.InputLayer)]

    def get_params(self, **tags):
        return lnn.layers.get_all_params(self.net, **tags)

    def compile_process_function(self):
        self._process = theano.function(
            inputs=self.get_inputs(),
            outputs=self.get_outputs(deterministic=True),
            name='process'
        )

    def process(self, *args):
        if self._process is None:
            self.compile_process_function()
        return self._process(*args)

    def save(self, filename):
        pickle.dump(
            lnn.layers.get_all_param_values(self.net),
            open(filename, 'wb')
        )

    def load(self, filename):
        lnn.layers.set_all_param_values(
            self.net,
            pickle.load(open(filename, 'rb'))
        )

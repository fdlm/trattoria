import pickle
import lasagne as lnn
import theano


class NeuralNetwork(object):
    """
    Standard neural network class.

    Parameters
    ----------
    net : lasagne layer
        Output layer handle that represents the neural network
    """

    def __init__(self, net, output_tensor):
        self.net = net
        self._process = None
        self._output_tensor = output_tensor

    def get_output(self, **kwargs):
        """
        Get network output.

        Parameters
        ----------
        **kwargs (optional)
            Parameters to pass to lasagne.layers.get_output

        Returns
        -------
        Network output

        """
        return lnn.layers.get_output(self.net, **kwargs)

    def get_output_tensor(self):
        """
        Get symbolic output tensor.

        Returns
        -------
        Theano tensor variable

        """
        return self._output_tensor

    def get_inputs(self):
        """
        Get input variables of the network

        Returns
        -------
        list of theano input variables

        """
        return [l.input_var
                for l in lnn.layers.get_all_layers(self.net)
                if isinstance(l, lnn.layers.InputLayer)]

    def get_params(self, **tags):
        """
        Get network parameters.

        Parameters
        ----------
        **tags (optional)
            tags to filter which network parameters to return.

        Returns
        -------
        list of network parameters

        """
        return lnn.layers.get_all_params(self.net, **tags)

    def get_param_values(self):
        """
        Get network parameter values.

        Returns
        -------
        list of numpy arrays containing network parameters

        """
        return lnn.layers.get_all_param_values(self.net)

    def set_param_values(self, params):
        """
        Set network parameter values.

        Parameters
        ----------
        params : list of numpy arrays
            parameters to set

        Returns
        -------
        None

        """
        lnn.layers.set_all_param_values(self.net, params)

    def compile_process_function(self):
        """
        Compile process function.
        """
        self._process = theano.function(
            inputs=self.get_inputs(),
            outputs=self.get_output(deterministic=True),
            name='process'
        )

    def process(self, *args):
        """
        Compute network output for input data. Compiles the process function
        if it has not been compiled yet.

        Parameters
        ----------
        args : numpy arrays
            Data to be processed by the network.

        Returns
        -------
        Network output for the input data.

        """
        if self._process is None:
            self.compile_process_function()
        return self._process(*args)

    def save(self, filename):
        """
        Pickle network parameters to a file.

        Parameters
        ----------
        filename : str
            Filename where to pickle network parameters to.
        """
        pickle.dump(
            lnn.layers.get_all_param_values(self.net),
            open(filename, 'wb')
        )

    def load(self, filename):
        """
        Load network parameters from a pickle file.

        Parameters
        ----------
        filename : str
            Filename from which to load network parameters.
        """
        lnn.layers.set_all_param_values(
            self.net,
            pickle.load(open(filename, 'rb'))
        )

    def __str__(self):
        repr_str = ''
        for layer in lnn.layers.get_all_layers(self.net):
            if isinstance(layer, lnn.layers.DropoutLayer):
                repr_str += '\t -> dropout p = {:g}\n'.format(layer.p)
                continue

            repr_str += '\t{:20s} {:>20s}'.format(
                type(layer).__name__, layer.output_shape)
            if layer.name:
                repr_str += ' - {}'.format(layer.name)
            if isinstance(layer, lnn.layers.DenseLayer):
                repr_str += '  W ({}x{})'.format(*layer.W.shape.eval())
            repr_str += '\n'

        # return everything except the last newline
        return repr_str[:-1]

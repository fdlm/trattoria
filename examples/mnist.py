from __future__ import print_function
import sys
import os
import numpy as np
import lasagne
import trattoria
import theano
from functools import partial


def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    training_set = trattoria.data.DataSource(X_train[:1000], y_train[:1000].astype(np.int))
    validation_set = trattoria.data.DataSource(X_val, y_val.astype(np.int))
    test_set = trattoria.data.DataSource(X_test, y_test)

    print(training_set)

    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28))
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    net = trattoria.nets.NeuralNetwork(l_out)
    train_batches = trattoria.iterators.BatchIterator(
        datasource=training_set,
        batch_size=500,
        shuffle=True
    )
    val_batches = trattoria.iterators.BatchIterator(
        datasource=validation_set,
        batch_size=500,
        fill_last=False
    )
    val = trattoria.training.Validator(
        net=net, batches=val_batches,
        observables={
            'loss': trattoria.objectives.average_categorical_crossentropy,
            'acc': trattoria.objectives.average_categorical_accuracy
        }
    )

    lr = theano.shared(np.float32(0.001), allow_downcast=True)
    updater = partial(lasagne.updates.adam, learning_rate=lr)
    # vs = trattoria.schedules.ValueScheduler(
    #     lr, {10: 0.01, 50: 0.001, 100: 0.0001}
    # )
    vs = trattoria.schedules.PatienceMult(lr, 0.1, 'loss', 2)

    trattoria.training.train(
        net=net,
        train_batches=train_batches,
        num_epochs=500,
        observables={
            'loss': trattoria.objectives.average_categorical_crossentropy,
            'lr': lambda *args: lr
        },
        updater=updater,
        validator=val,
        callbacks=[vs]
    )


if __name__ == "__main__":
    main()


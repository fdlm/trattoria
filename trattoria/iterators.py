import random


def iterate_batches(data, batch_size, shuffle=False, fill_last=True):
    """
    Generates mini-batches from data.

    Parameters
    ----------
    data : Indexable
        Data to generate mini-batches from. Needs to be indexable and return
        a tuple (data, target) for an index, and provide its length with len()
    batch_size : int
        Number of data points and targets in each mini-batch
    shuffle : bool
        Indicates whether to randomize the items in each mini-batch
        or not.
    fill_last : bool
        Indicates whether to fill up the last mini-batch with
        random data points if there is not enough data available.

    Yields
    ------
    tuple of numpy arrays
        mini-batch of data and targets

    """

    idxs = range(len(data))
    if shuffle:
        random.shuffle(idxs)

    # last batch could be too small
    if fill_last and len(data) % batch_size != 0:
        # fill up with random indices
        idxs += random.sample(idxs, batch_size - len(data) % batch_size)

    while len(idxs) > 0:
        batch_idxs, idxs = idxs[:batch_size], idxs[batch_size:]
        yield data[batch_idxs]


class BatchIterator:
    """
    Iterates over mini batches of data.

    Parameters
    ----------
    datasource : Indexable
        Data to generate mini-batches from. Needs to be indexable and return
        a tuple (data, target) for an index, and provide its length with len()
    batch_size : int
        Number of data points and targets in each mini-batch
    shuffle : bool
        Indicates whether to randomize the items in each mini-batch
        or not.
    fill_last : bool
        Indicates whether to fill up the last mini-batch with
        random data points if there is not enough data available.

    Yields
    ------
    tuple of numpy arrays
        mini-batch of data and targets
    """

    def __init__(self, datasource, batch_size, shuffle=False, fill_last=True):
        self.datasource = datasource
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fill_last = fill_last

    def __iter__(self):
        """Returns the mini batch generator."""
        return iterate_batches(self.datasource, self.batch_size,
                               self.shuffle, self.fill_last)

    @property
    def tshape(self):
        return self.datasource.tshape

    @property
    def ttype(self):
        return self.datasource.ttype

    def __len__(self):
        return len(self.datasource) // self.batch_size + 1

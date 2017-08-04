import random
import numpy as np
import functools


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

    i = 0

    while i < len(idxs):
        batch_idxs = idxs[i:i + batch_size]
        yield data[batch_idxs]
        i += batch_size


def _chunks_to_arrays(data_chunks, target_chunks, max_len, return_mask):
    """
    Concatenates chunks of data and targets into a single array.

    This array has a pre-defined "length". If a chunk is shorter than this
    length, it is padded with the last valid value and corresponding elements
    are masked in the corresponding mask array.

    Parameters
    ----------
    data_chunks : list of numpy arrays
        Data chunks to concatenate
    target_chunks : list of numpy arrays
        Target chunks to concatenate
    max_len : int
        Length if the concatenated array

    Returns
    -------
    tuple of numpy arrays
        Concatenated data, target, and mask arrays

    """
    # create the arrays to store data, targets, and mask
    feature_shape = data_chunks[0].shape[1:]
    target_shape = target_chunks[0].shape[1:]
    data = np.zeros(
        (len(data_chunks), max_len) + feature_shape,
        dtype=data_chunks[0].dtype)
    targets = np.zeros(
        (len(target_chunks), max_len) + target_shape,
        dtype=target_chunks[0].dtype)
    mask = np.zeros(
        (len(data_chunks), max_len),
        dtype=np.float32
    )

    for i in range(len(data_chunks)):
        dlen = len(data_chunks[i])
        data[i, :dlen] = data_chunks[i]
        targets[i, :dlen] = target_chunks[i]
        mask[i, :dlen] = 1.
        # Repeat last valid value of data and targets throughout the whole
        # masked area. This is consistent with the semantics of Lasagne's RNN
        # implementation, which repeats the previous output value at every
        # masked element. Also, Spaghetti (CRF Library) requires it to be this
        # way.
        data[i, dlen:] = data[i, dlen - 1]
        targets[i, dlen:] = targets[i, dlen - 1]

    if return_mask:
        return data, mask, targets
    else:
        return data, targets


def _chunks_to_arrays_cls(data_chunks, target_chunks, max_len, return_mask):
    # create the arrays to store data, targets, and mask
    max_len = max(dc.shape[1] for dc in data_chunks)
    feature_shape = data_chunks[0].shape[2:]
    target_shape = target_chunks[0].shape[1:]
    data = np.zeros(
        (len(data_chunks), max_len) + feature_shape,
        dtype=data_chunks[0].dtype)
    targets = np.zeros(
        (len(target_chunks),) + target_shape,
        dtype=target_chunks[0].dtype)
    mask = np.zeros(
        (len(data_chunks), max_len),
        dtype=np.float32
    )

    for i in range(len(data_chunks)):
        dlen = data_chunks[i].shape[1]
        data[i, :dlen] = data_chunks[i][0]
        targets[i] = target_chunks[i][0]
        mask[i, :dlen] = 1.
        # Repeat last valid value of data and targets throughout the whole
        # masked area. This is consistent with the semantics of Lasagne's RNN
        # implementation, which repeats the previous output value at every
        # masked element. Also, Spaghetti (CRF Library) requires it to be this
        # way.
        data[i, dlen:] = data[i, dlen - 1]

    if return_mask:
        return data, mask, targets
    else:
        return data, targets


def iterate_sequences(datasources, batch_size, shuffle=False,
                      fill_last=True, max_seq_len=None, mask=True,
                      compile_chunk_fn=_chunks_to_arrays):
    """
    Generates mini batches of sequences by iterating over a list of
    data sources.

    This generator generates mini batches of :param:batch_size sub-sequences
    of length :param:max_seq_len. Each :class:DataSource contained in the
    list of data sources is considered a sequence. If too long, sequences
    are broken into several sub-sequences in a mini batch.

    Parameters
    ----------
    datasources : list of data sources
        Data sources to generate mini-batches from
    batch_size : int
        Number of (sub-)sequences per mini batch
    shuffle : bool
        Indicates whether to randomise the order of data sources
    fill_last : bool
        Indicates whether to fill the last mini batch with sequences from a
        random data source if there is not enough data available
        Maximum length of each sequence in a data source
    max_seq_len : int or None
        Maximum sequence length of each sub-sequence in the mini batch. If
        None, the maximum length is determined to be the longest data source
        in the mini-batch. Note that this might result in different
        sequence lengths in each mini batch.

    Yields
    ------
    tuple of numpy arrays
        mini batch of sub-sequences with data, target, and mask arrays

    """

    ds_idxs = range(len(datasources))
    if shuffle:
        random.shuffle(ds_idxs)

    data_chunks = []
    target_chunks = []

    max_len = max_seq_len or 0
    for ds_idx in ds_idxs:
        ds = datasources[ds_idx]
        # we chunk the data according to sequence_length
        for d, t in iterate_batches(ds, max_seq_len or len(ds),
                                    shuffle=False, fill_last=False):
            data_chunks.append(d)
            target_chunks.append(t)
            max_len = max(max_len, len(d))

            if len(data_chunks) == batch_size:
                yield compile_chunk_fn(data_chunks, target_chunks, max_len,
                                       mask)
                data_chunks = []
                target_chunks = []
                max_len = max_seq_len or 0

    # after we processed all data sources, there might be some chunks left.
    while fill_last and len(data_chunks) < batch_size:
        # add more sequences until we fill it up
        # get a random data source
        ds_idx = random.sample(ds_idxs, 1)[0]
        ds = datasources[ds_idx]
        for d, t in iterate_batches(ds, max_seq_len or len(ds),
                                    shuffle=False, fill_last=False):
            data_chunks.append(d)
            target_chunks.append(t)
            max_len = max(max_len, len(d))

            if len(data_chunks) == batch_size:
                # we filled it!
                break

    if len(data_chunks) > 0:
        yield compile_chunk_fn(data_chunks, target_chunks, max_len, mask)


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

    def __len__(self):
        return len(self.datasource) // self.batch_size + 1


class SequenceIterator:
    """
    Iterates over mini batches of sequences from an aggregated data source.

    Each mini batch contains :param:batch_size sequences of length
    :param:max_seq_len. Each :class:DataSource contained in the
    data source list is considered a sequence. If too long, it is
    broken into several sub-sequences in a mini batch.

    Parameters
    ----------
    datasources : list of data sources
        List of Data source to generate mini-batches from
    batch_size : int
        Number of (sub-)sequences per mini batch
    randomise : bool
        Indicates whether to randomise the order of data sources
    expand : bool
        Indicates whether to fill the last mini batch with sequences from a
        random data source if there is not enough data available
        Maximum length of each sequence in a data source
    max_seq_len : int or None
        Maximum sequence length of each sub-sequence in the mini batch. If
        None, the maximum length is determined to be the longest data source
        in the mini-batch. Note that this might result in different
        sequence lengths in each mini batch.

    Yields
    ------
    tuple of numpy arrays
        mini batch of sub-sequences with data, target, and mask arrays
    """

    def __init__(self, datasources, batch_size, shuffle=False, fill_last=True,
                 max_seq_len=None, mask=True):
        if len(datasources) == 0:
            raise ValueError('Need at least one data source!')
        self.datasources = datasources
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fill_last = fill_last
        self.max_seq_len = max_seq_len
        self.mask = mask

    def __iter__(self):
        """Returns the sequence mini batch generator."""
        return iterate_sequences(self.datasources, self.batch_size,
                                 self.shuffle, self.fill_last, self.max_seq_len,
                                 self.mask)

    def __len__(self):
        return len(self.datasources) // self.batch_size + 1


class SequenceClassificationIterator:

    def __init__(self, datasources, batch_size, shuffle=False, fill_last=True):
        self.datasources = datasources
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fill_last = fill_last

    def __iter__(self):
        return iterate_sequences(
            self.datasources,
            self.batch_size,
            self.shuffle,
            self.fill_last,
            compile_chunk_fn=_chunks_to_arrays_cls)

    def __len__(self):
        return len(self.datasources) // self.batch_size + 1



class SubsetIterator:

    def __init__(self, batch_iterator, percentage=1.0):
        self.percentage = percentage
        self.batch_iterator = batch_iterator

    def __iter__(self):
        for i, batch in enumerate(self.batch_iterator):
            yield batch
            if i > len(self):
                break

    @property
    def tshape(self):
        return self.batch_iterator.tshape

    @property
    def ttype(self):
        return self.batch_iterator.ttype

    def __len__(self):
        return int(len(self.batch_iterator) * self.percentage)


def compose(*functions):
    """Compose a list of function to one."""
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions,
                            lambda x: x)


class AugmentedIterator:

    def __init__(self, batch_iterator, *augment_fns):
        self.batch_iterator = batch_iterator
        self.augment = compose(*augment_fns)

    def __iter__(self):
        return self.augment(self.batch_iterator.__iter__())

    @property
    def tshape(self):
        return self.batch_iterator.tshape

    @property
    def ttype(self):
        return self.batch_iterator.ttype

    def __len__(self):
        return len(self.batch_iterator)

class DataSource(object):

    def __init__(self, *data):
        if len(data) < 2:
            raise ValueError('Need at least input and targets!')

        n_data = data[0].shape[0]
        for i, d in enumerate(data):
            if d.shape[0] != n_data:
                raise ValueError(
                    'Number of data at idx {} ({}) is not equals number of '
                    'data at idx 0 ({})'.format(i, d.shape[0], n_data)
                )
        self.data = data

    def __getitem__(self, idx):
        return tuple([d[idx] for d in self.data])

    def __len__(self):
        return len(self.data[0])

    def shape(self, idx):
        if idx >= len(self.data):
            raise ValueError()
        # return shape of data, skip number of data
        return self.data[idx].shape[1:]

    @property
    def dshape(self):
        return self.shape(0)

    @property
    def tshape(self):
        return self.shape(-1)

    def type(self, idx):
        if idx >= len(self.data):
            raise ValueError()
        return self.data[idx].dtype

    @property
    def dtype(self):
        return self.type(0)

    @property
    def ttype(self):
        return self.type(-1)

    def __str__(self):
        return '{}: N={}  dshape={}  tshape={}'.format(
            self.__class__.__name__, len(self),
            self.dshape, self.tshape)

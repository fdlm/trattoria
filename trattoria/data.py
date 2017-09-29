import numpy as np


class DataSource(object):

    def __init__(self, data, name=''):
        n_data = data[0].shape[0]
        for i, d in enumerate(data):
            if d.shape[0] != n_data:
                raise ValueError(
                    'Number of data at idx {} ({}) is not equals number of '
                    'data at idx 0 ({})'.format(i, d.shape[0], n_data)
                )
        self._data = data
        self.name = name

    def __getitem__(self, idx):
        return tuple([d[idx] for d in self._data])

    def __len__(self):
        return len(self._data[0])

    def shape(self, idx):
        if idx >= len(self._data):
            raise ValueError()
        # return shape of data, skip number of data
        return self._data[idx].shape[1:]

    @property
    def dshape(self):
        return self.shape(0)

    @property
    def tshape(self):
        return self.shape(-1)

    def type(self, idx):
        if idx >= len(self._data):
            raise ValueError()
        return self._data[idx].dtype

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


class AggregatedDataSource(object):

    def __init__(self, datasources):
        if len(datasources) == 0:
            raise ValueError('Need at least one data source')
        if not all(x.dtype == datasources[0].dtype for x in datasources):
            raise ValueError('Data sources data type has to be equal')
        if not all(x.ttype == datasources[0].ttype for x in datasources):
            raise ValueError('Data sources target type has to be equal')

        self._datasources = datasources
        self._ds_idxs = np.array(
            sum([[i] * len(ds) for i, ds in enumerate(datasources)], []))
        self._ds_ends = np.array([0] + [len(d) for d in datasources]).cumsum()

    def _get_ds_idx(self, idx):
        datasource_idx = self._ds_idxs[idx]
        data_idx = idx - self._ds_ends[datasource_idx]
        return datasource_idx, data_idx

    def __getitem__(self, idx):
        if isinstance(idx, int):
            ds_idx, d_idx = self._get_ds_idx(idx)
            return self._datasources[ds_idx][d_idx]
        elif isinstance(idx, np.ndarray):
            ds_idxs, d_idxs = self._get_ds_idx(idx)
            data = []
            for ds_idx, d_idx in zip(ds_idxs, d_idxs):
                data.append(self._datasources[ds_idx][d_idx])
            return zip(*data)
        elif isinstance(idx, slice):
            return self[np.arange(idx.start or 0, idx.stop or len(self),
                                  idx.step or 1)]
        elif isinstance(idx, list):
            return self[np.array(idx)]
        else:
            raise TypeError('Index type {} not supported!'.format(type(idx)))

    def __len__(self):
        return sum(len(ds) for ds in self._datasources)

    def shape(self, idx):
        total_shape = None
        for shp in (ds.shape(idx) for ds in self._datasources):
            if total_shape is None:
                total_shape = list(shp)
            else:
                for i in range(len(shp)):
                    if total_shape[i] != shp[i]:
                        total_shape[i] = None
        return tuple(total_shape)

    @property
    def datasources(self):
        return list(self._datasources)

    @property
    def dshape(self):
        return self.shape(0)

    @property
    def tshape(self):
        return self.shape(-1)

    @property
    def dtype(self):
        return self._datasources[0].type(0)

    @property
    def ttype(self):
        return self._datasources[0].type(-1)

    def __str__(self):
        return '{}: N_ds={} N={}  dshape={}  tshape={}'.format(
            self.__class__.__name__, len(self.datasources), len(self),
            self.dshape, self.tshape)

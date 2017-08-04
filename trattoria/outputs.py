import os
import yaml
from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle


class ConsoleLog(object):

    def __init__(self, out=tqdm):
        self.out = out
        self.log_val_names = None
        self.header_fmt = None
        self.row_fmt = None

    def start(self, train_objectives, val_objectives=None):
        if val_objectives is None:
            val_objectives = []

        self.log_val_names = ['epoch', 'loss']
        row_fmts = ['{loss:>15.6f}']
        header_fmts = ['{:>5s}', '{:>15s}']

        for name in train_objectives:
            if name == 'loss':
                continue
            self.log_val_names.append(name)
            name_len = max(15, len(name))
            header_fmts.append('{:>%ds}' % name_len)
            row_fmts.append('{%s:>%d.6f}' % (name, name_len))

        if 'val_loss' in val_objectives:
            self.log_val_names.append('val_loss')
            row_fmts.append('{val_loss:>15.6f}')
            header_fmts.append('{:>15s}')

        for name in val_objectives:
            if name == 'val_loss':
                continue
            self.log_val_names.append(name)
            cap_len = max(15, len(name))
            header_fmts.append('{:>%ds}' % cap_len)
            row_fmts.append('{%s:>%d.6f}' % (name, cap_len))

        self.header_fmt = ''.join(header_fmts)
        self.row_fmt = ''.join(row_fmts)
        header = self.header_fmt.format(*self.log_val_names)
        self.out.write(header)
        self.out.write('-' * len(header))

    def add(self, epoch, epoch_results):
        if self.row_fmt is None:
            raise ValueError('Use start() before add()!')
        row = self.row_fmt.format(**epoch_results)
        self.out.write('{:>5d}'.format(epoch) + row)


class YamlLog(object):

    def __init__(self, filename, load=False):
        self.filename = filename
        self.load = load
        if not self.load or not os.path.exists(filename):
            self.loaded = False
            self.log = {}
        else:
            self.loaded = True
            self.log = yaml.load(open(filename))

    def start(self, train_objectives, val_objectives):
        if val_objectives is None:
            val_objectives = []
        for o in train_objectives:
            if self.loaded and o not in self.log:
                raise ValueError('Training objective "{}" '
                                 'not in YAML log.'.format(o))
            self.log[o] = self.log.get(o, [])
        for o in val_objectives:
            if self.loaded and o not in self.log:
                raise ValueError('Validation objective "{}" '
                                 'not in YAML log.'.format(o))
            self.log[o] = self.log.get(o, [])

    def add(self, epoch, epoch_results):
        import yaml
        for name, value in epoch_results.items():
            self.log[name].append(value)
        with open(self.filename, 'w') as f:
            yaml.dump(self.log, f)


class ModelCheckpoint(object):

    def __init__(self, net, file_fmt, max_history=0):
        self.net = net
        self.history = []
        self.max_history = max_history
        self.file_fmt = file_fmt

    def __call__(self, epoch, epoch_results):
        filename = self.file_fmt.format(epoch=epoch, **epoch_results)
        with open(filename, 'wb') as f:
            pickle.dump(self.net.get_param_values(), f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            self.history.append(filename)
        if self.max_history and len(self.history) > self.max_history:
            os.remove(self.history.pop(0))
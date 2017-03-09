from tqdm import tqdm


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

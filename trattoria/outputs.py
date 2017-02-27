from tqdm import tqdm


class ConsoleLog(object):

    def __init__(self, train_objectives, val_objectives=None, out=tqdm):
        self.out = out
        if val_objectives is None:
            val_objectives = []

        self.log_val_names = ['epoch', 'loss']
        train_row_fmts = ['{loss:>15.6f}']
        header_fmts = ['{:>5s}', '{:>15s}']

        for name in train_objectives:
            if name == 'loss':
                continue
            self.log_val_names.append(name)
            name_len = max(15, len(name))
            header_fmts.append('{:>%ds}' % name_len)
            train_row_fmts.append('{%s:>%d.6f}' % (name, name_len))

        val_row_fmts = []
        if 'val.loss' in val_objectives:
            self.log_val_names.append('val.loss')
            val_row_fmts.append('{val.loss:>15.6f}')
            header_fmts.append('{:>15s}')

        for name in val_objectives:
            if name == 'val.loss':
                continue
            self.log_val_names.append(name)
            cap_len = max(15, len(name))
            header_fmts.append('{:>%ds}' % cap_len)
            val_row_fmts.append('{%s:>%d.6f}' % (name, cap_len))

        self.header_fmt = ''.join(header_fmts)
        self.train_row_fmt = ''.join(train_row_fmts)
        self.val_row_fmt = ''.join(val_row_fmts)

    def write_header(self):
        self.out.write(self.header_fmt.format(*self.log_val_names))

    def write_row(self, epoch, train_objectives, val_objectives=None):
        train_row = self.train_row_fmt.format(**train_objectives)
        if val_objectives:
            val_row = self.val_row_fmt.format(**val_objectives)
        else:
            val_row = ''
        self.out.write('{:>5d}'.format(epoch) + train_row + val_row)
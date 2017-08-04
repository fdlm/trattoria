from . import data
from . import iterators
from . import nets
from . import objectives
from . import outputs
from . import schedules
from . import training

__version__ = "0.1.dev0"


import yaml
import numpy as np


def _yaml_rep_npfloat(self, val):
    return self.represent_float(val)


def _yaml_rep_npint(self, val):
    return self.represent_int(val)


yaml.add_representer(np.float, _yaml_rep_npfloat)
yaml.add_representer(np.float16, _yaml_rep_npfloat)
yaml.add_representer(np.float32, _yaml_rep_npfloat)
yaml.add_representer(np.float64, _yaml_rep_npfloat)
yaml.add_representer(np.int, _yaml_rep_npint)
yaml.add_representer(np.int16, _yaml_rep_npint)
yaml.add_representer(np.int32, _yaml_rep_npint)
yaml.add_representer(np.int64, _yaml_rep_npint)


del yaml
del np

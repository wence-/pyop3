from collections import namedtuple

import numpy

from pyop3.obj.args import AbstractArg, ArgType
from pyop3.utils import cached_property, check_args


class GlobalArg(AbstractArg):
    lgmaps = None
    unroll_map = False
    map_tuple = ()
    Info = namedtuple("Info", ["packer", "shape", "dtype"])
    Info.argtype = ArgType.GLOBAL

    def validator(self, obj):
        assert isinstance(obj, Global)

    @check_args(validator)
    def __init__(self, obj):
        self.obj = obj
        self._parloop_args_ = obj._parloop_args_

    def validate(self, iterset):
        return True

    @cached_property
    def _codegen_info_(self):
        return GlobalArg.Info(packer=None, shape=self.obj.shape, dtype=self.obj.dtype)


class Global(object):
    def __init__(self, dataset, dtype, data=None):
        shape = dataset.shape
        if data is None:
            self._data = numpy.zeros(shape, dtype=dtype)
        else:
            assert data.dtype == dtype
            self._data = data.reshape(shape)
        self.dataset = dataset
        self.comm = dataset.comm
        self.shape = shape
        self.dtype = dtype

    @cached_property
    def _parloop_args_(self):
        return (self._data.ctypes.data, )

    def arg(self, map_tuple=None):
        return GlobalArg(self)

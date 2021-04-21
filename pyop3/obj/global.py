import ctypes
from operator import itemgetter

import numpy

from pyop3.obj.args import AbstractArg, ArgType
from pyop3.utils import cached_property, debug_check_args


class GlobalArg(AbstractArg):
    argtype = ArgType.GLOBAL
    lgmaps = None
    unroll_map = False
    map_tuple = ()

    class Info(tuple):
        argtype = ArgType.GLOBAL
        packer = property(itemgetter(0))
        shape = property(itemgetter(1))
        dtype = property(itemgetter(2))

    def validator(self, obj):
        assert isinstance(obj, Global)

    @debug_check_args(validator)
    def __init__(self, obj):
        self.obj = obj
        self._parloop_args_ = obj._parloop_args_
        self._parloop_argtypes_ = obj._parloop_argtypes_

    def validate(self, iterset, toset):
        return True

    @cached_property
    def _codegen_info_(self):
        return GlobalArg.Info((None, self.obj.shape, self.obj.dtype))


class Global(object):
    _parloop_argtypes_ = (ctypes.c_voidp, )

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

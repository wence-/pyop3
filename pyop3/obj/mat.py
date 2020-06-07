from collections import namedtuple

from pyop3.datatypes import ScalarType
from pyop3.obj.args import AbstractArg, ArgType
from pyop3.utils import cached_property, debug_check_args


class MatArg(AbstractArg):
    Info = namedtuple("Info", ["packer", "dtype", "mapinfo", "unroll", "shape"])
    Info.argtype = ArgType.MAT

    def validator(self, obj, map_tuple, *, lgmaps=None, unroll=None):
        for m, d in zip(map_tuple, obj.datasets):
            assert m.toset == d.iterset

    @debug_check_args(validator)
    def __init__(self, obj, map_tuple, *, lgmaps=None, unroll=False):
        self.obj = obj
        self.map_tuple = tuple(map_tuple)
        self.lgmaps = lgmaps
        self.unroll = unroll
        self._parloop_args_ = obj._parloop_args_

    def validate(self, iterset):
        for m in self.map_tuple:
            if m.iterset != iterset:
                return False
        return True

    @cached_property
    def _codegen_info_(self):
        return MatArg.Info(packer=None, dtype=self.obj.dtype,
                           mapinfo=tuple(m._codegen_info_ for m in self.map_tuple),
                           unroll=self.unroll,
                           shape=self.obj.shape)


class Mat(object):
    def __init__(self, sparsity):
        self.sparsity = sparsity
        self.dtype = ScalarType
        self.shape = (1, 1)     # FIXME

    def arg(self, map_tuple, *, lgmaps=None, unroll_map=False):
        return MatArg(self, map_tuple, lgmaps=lgmaps, unroll=unroll_map)


class MixedMat(object):
    def arg(self):
        raise NotImplementedError

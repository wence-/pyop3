from operator import itemgetter

from pyop3.datatypes import ScalarType
from pyop3.obj.args import AbstractArg, ArgType
from pyop3.utils import cached_property, debug_check_args


class MatArg(AbstractArg):
    argtype = ArgType.MAT

    class Info(tuple):
        argtype = ArgType.MAT
        packer = property(itemgetter(0))
        shape = property(itemgetter(1))
        dtype = property(itemgetter(2))
        unroll = property(itemgetter(3))
        mapinfo = property(itemgetter(4))

    def validator(self, obj, map_tuple, *, lgmaps=None, unroll=None):
        for m, d in zip(map_tuple, obj.datasets):
            assert m.toset == d.iterset

    @debug_check_args(validator)
    def __init__(self, obj, map_tuple, *, lgmaps=None, unroll=False):
        self.obj = obj
        self._parloop_args_ = obj._parloop_args_
        self._parloop_argtypes_ = obj._parloop_argtypes_
        self.map_tuple = map_tuple
        self.lgmaps = lgmaps
        self.unroll = unroll

    def validate(self, iterset):
        for m in self.map_tuple:
            if m.iterset != iterset:
                return False
        return True

    @cached_property
    def _codegen_info_(self):
        return MatArg.Info((None, self.obj.shape, self.obj.dtype,
                            self.unroll,
                            tuple(m._codegen_info_ for m in self.map_tuple)))


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

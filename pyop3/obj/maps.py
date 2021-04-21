import ctypes
from collections import namedtuple
from itertools import chain

import numpy
from petsc4py.PETSc import IntType
from pyop3.obj.sets import DataSet, ExtrudedSet, Set
from pyop3.utils import cached_property, check_args


class IdentityMap(object):
    """Singleton marker for the identity map (direct args)"""
    _codegen_info_ = None
    _parloop_argtypes_ = ()
    _parloop_args_ = ()

    @staticmethod
    def validate(iterset, toset):
        return iterset == toset


IdentityMap = IdentityMap()


class Map(object):
    dtype = IntType
    offset = None
    Info = namedtuple("Info", ["shape", "offset", "dtype", "setinfo"])
    _parloop_argtypes_ = (ctypes.c_voidp, )

    def validator(self, iterset, toset, arity, values, offset=None):
        assert isinstance(iterset, (Set, ExtrudedSet))
        assert isinstance(toset, (Set, ExtrudedSet))
        values = numpy.asarray(values).astype(IntType)
        assert max(values) < toset.total_size
        if offset is not None:
            assert len(offset) == arity
            assert all(o >= 0 for o in offset)

    @check_args(validator)
    def __init__(self, iterset, toset, arity, values, offset=None):
        self.iterset = iterset
        self.toset = toset
        self.comm = toset.comm  # FIXME
        values = numpy.asarray(values).astype(IntType)
        self.values = values.reshape(iterset.total_size, arity)
        self.arity = arity
        self.shape = self.values.shape
        if offset is not None:
            self.offset = offset.reshape(self.arity)

    def validate(self, iterset, toset):
        return iterset == self.iterset and toset == self.toset

    @cached_property
    def _codegen_info_(self):
        return Map.Info(shape=self.shape, offset=self.offset, dtype=self.dtype,
                        setinfo=self.iterset._codegen_info_)

    @cached_property
    def _parloop_args_(self):
        return (self.values.ctypes.data, )


class MixedMap(object):

    def validator(self, maps):
        assert all(isinstance(m, (Map, type(None))) for m in maps)

    @check_args(validator)
    def __init__(self, maps):
        self.maps = tuple(maps)
        self.comm = maps[0].comm  # FIXME

    @cached_property
    def _parloop_argtypes_(self):
        return tuple(chain(*(m._parloop_argtypes_ for m in self.maps)))

    @cached_property
    def _parloop_args_(self):
        return tuple(chain(*(m._parloop_args_ for m in self.maps)))


class Sparsity(object):
    def validator(self, datasets, maps, *, block_sparse=True):
        assert all(isinstance(d, DataSet) for d in datasets)
        assert all(isinstance(m, Map) for m in chain(*maps))
        rset, cset = datasets
        for rmap, cmap in maps:
            assert rmap.iterset == cmap.iterset
            assert rmap.toset == rset.iterset
            assert cmap.toset == cset.iterset

    @check_args(validator)
    def __init__(self, datasets, maps, *, block_sparse=True):
        self.datasets = tuple(datasets)
        rowmaps, colmaps = zip(*maps)
        self.rowmaps = tuple(rowmaps)
        self.colmaps = tuple(colmaps)
        self.block_sparse = block_sparse
        self.shape = tuple(int(numpy.prod(d.shape, dtype=int)) for d in self.datasets)


class MixedSparsity(object):
    def validator(self, sparsities, shape):
        assert all(isinstance(s, Sparsity) for s in sparsities)
        assert len(sparsities) == int(numpy.prod(shape))

    @check_args(validator)
    def __init__(self, sparsities, shape):
        self.sparsities = numpy.asarray(sparsities, dtype=object).reshape(shape)

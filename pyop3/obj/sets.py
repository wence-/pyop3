import operator
from collections import namedtuple
from functools import reduce

import numpy
from petsc4py import PETSc

from pyop3.utils import cached_property, check_args


class AbstractSet(object):
    subset = False
    constant_layers = False
    Info = namedtuple("Info", ["subset", "extruded", "constant_layers"])

    def validator(self, sizes, halo, comm):
        assert len(sizes) == 3
        assert sizes[0] <= sizes[1] <= sizes[2]

    @check_args(validator)
    def __init__(self, sizes, halo, comm):
        self.sizes = tuple(sizes)
        self.halo = halo
        self.comm = comm

    @cached_property
    def core_size(self):
        return self.sizes[0]

    @cached_property
    def size(self):
        return self.sizes[1]

    @cached_property
    def total_size(self):
        return self.sizes[2]

    @cached_property
    def _codegen_info_(self):
        return AbstractSet.Info(self.subset, self.extruded, self.constant_layers)


class AbstractDataSet(object):
    def validator(self, sizes, halo, comm, shape):
        assert len(sizes) == 3
        assert sizes[0] <= sizes[1] <= sizes[2]
        assert all(s >= 0 for s in shape)

    @check_args(validator)
    def __init__(self, sizes, halo, comm, shape):
        self.sizes = tuple(sizes)
        self.halo = halo
        self.comm = comm
        self.shape = tuple(shape)
        self.value_size = reduce(operator.mul, shape)

    @cached_property
    def core_size(self):
        return self.sizes[0]

    @cached_property
    def size(self):
        return self.sizes[1]

    @cached_property
    def total_size(self):
        return self.sizes[2]


class Set(AbstractSet):
    extruded = False

    _parloop_args_ = ()

    def __init__(self, sizes, halo=None, comm=None):
        super().__init__(sizes, halo, comm)


class ExtrudedSet(AbstractSet):
    extruded = True

    def __init__(self, base, layers):
        super().__init__(base.sizes, base.halo, base.comm)
        self.layers = layers
        self.constant_layers = False  # FIXME

    @cached_property
    def _parloop_args_(self):
        return (self.layers.ctypes.data, )


class Subset(AbstractSet):
    subset = True

    def validator(self, parent, indices):
        assert isinstance(parent, (Set, ExtrudedSet))
        indices = numpy.unique(indices).astype(numpy.int32)
        assert len(indices) > 0 and (indices[0] < 0 or indices[-1] >= parent.sizes[-1])

    @check_args(validator)
    def __init__(self, parent, indices):
        indices = numpy.unique(indices).astype(numpy.int32)
        sizes = tuple(indices[indices < s].sum() for s in parent.sizes)
        super().__init__(sizes, parent.halo, parent.comm)
        self.indices = indices
        self.parent = parent
        self.extruded = parent.extruded

    @cached_property
    def _parloop_args_(self):
        return self.parent._parloop_args_ + (self.indices.ctypes.data, )


class DataSet(AbstractDataSet):
    def validator(self, iterset, shape):
        assert isinstance(iterset, Set)

    @check_args(validator)
    def __init__(self, iterset, shape):
        super().__init__(iterset.sizes, iterset.halo, iterset.comm, shape)
        self.iterset = iterset

    @cached_property
    def lgmap(self):
        lgmap = PETSc.LGMap()
        if self.comm.size == 1:
            lgmap.create(indices=numpy.arange(self.size, dtype=PETSc.IntType),
                         bsize=self.value_size, comm=self.comm)
        else:
            lgmap.create(indices=self.halo.lgmap.indices,
                         bsize=self.value_size, comm=self.comm)
        return lgmap

    @cached_property
    def scalar_lgmap(self):
        if self.value_size == 1:
            return self.lgmap
        else:
            indices = self.lgmap.block_indices
            return PETSc.LGMap().create(indices=indices, bsize=1, comm=self.comm)

    @cached_property
    def unblocked_lgmap(self):
        if self.value_size == 1:
            return self.lgmap
        else:
            indices = self.lgmap.indices
            return PETSc.LGMap().create(indices=indices, bsize=1, comm=self.comm)


class MixedSet(object):
    __slots__ = ("comm", "sets")

    def validator(self, datasets):
        assert all(isinstance(s, AbstractSet) for s in datasets)

    @check_args(validator)
    def __init__(self, sets):
        self.sets = tuple(sets)
        self.comm = sets[0].comm


class MixedDataSet(object):
    __slots__ = ("comm", "datasets")

    def validator(self, datasets):
        assert all(isinstance(s, AbstractDataSet) for s in datasets)

    @check_args(validator)
    def __init__(self, datasets):
        self.datasets = tuple(datasets)
        self.comm = datasets[0].comm

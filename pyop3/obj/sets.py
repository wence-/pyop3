import ctypes
import operator
from collections import namedtuple
from functools import reduce

import numpy
from mpi4py import MPI
from petsc4py import PETSc

from pyop3.utils import cached_property, check_args


class NoopHalo(object):
    local_to_global_begin = lambda d, a, b, op: None
    local_to_global_end = lambda d, a, b, op: None
    global_to_local_begin = lambda d, a, b: None
    global_to_local_end = lambda d, a, b: None


NoopHalo = NoopHalo()


class Halo(object):
    def __init__(self, sf, lgindices):
        self.sf = sf
        self.lgindices = numpy.asarray(lgindices).astype(PETSc.IntType)
        self.comm = sf.comm.tompi4py()

    def local_to_global_begin(self, datatype, in_, out_, op=MPI.SUM):
        self.sf.reduceBegin(datatype, in_, out_, op)

    def local_to_global_end(self, datatype, in_, out_, op=MPI.SUM):
        self.sf.reduceEnd(datatype, in_, out_, op)

    def global_to_local_begin(self, datatype, in_, out_):
        self.sf.bcastBegin(datatype, in_, out_)

    def global_to_local_end(self, datatype, in_, out_):
        self.sf.bcastEnd(datatype, in_, out_)


class AbstractSet(object):
    subset = False
    constant_layers = False
    Info = namedtuple("Info", ["subset", "extruded", "constant_layers"])

    def validator(self, sizes, comm):
        assert len(sizes) == 3
        assert sizes[0] <= sizes[1] <= sizes[2]

    @check_args(validator)
    def __init__(self, sizes, comm):
        """Create a set-like thing.

        :arg sizes: sizes
        :arg comm: Communicator object."""
        self.sizes = tuple(sizes)
        self.comm = comm or MPI.COMM_WORLD

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
    def validator(self, iterset, shape, halo):
        assert isinstance(iterset, Set)
        assert halo is NoopHalo or isinstance(halo, Halo)
        assert all(s >= 0 for s in shape)

    @check_args(validator)
    def __init__(self, iterset, shape, halo):
        self.sizes = iterset.sizes
        self.iterset = iterset
        self.halo = halo
        self.comm = iterset.comm
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
    _parloop_argtypes_ = (ctypes.c_int32, ctypes.c_int32)

    def __init__(self, sizes, comm=None):
        super().__init__(sizes, comm)


class ExtrudedSet(AbstractSet):
    extruded = True

    _parloop_argtypes_ = (ctypes.c_int32, ctypes.c_int32, ctypes.c_voidp)

    def __init__(self, base, layers):
        super().__init__(base.sizes, base.comm)
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
        assert (len(indices) == 0
                or (0 <= indices[0] and indices[-1] < parent.sizes[-1]))

    @check_args(validator)
    def __init__(self, parent, indices):
        indices = numpy.unique(indices).astype(numpy.int32)
        sizes = tuple(indices[indices < s].sum() for s in parent.sizes)
        super().__init__(sizes, parent.halo, parent.comm)
        self.indices = indices
        self.parent = parent
        self.extruded = parent.extruded

    @cached_property
    def _parloop_argtypes_(self):
        return self.parent._parloop_argtypes_ + (ctypes.c_voidp, )

    @cached_property
    def _parloop_args_(self):
        return self.parent._parloop_args_ + (self.indices.ctypes.data, )


class DataSet(AbstractDataSet):
    def validator(self, iterset, shape, halo=None):
        assert isinstance(iterset, Set)

    @check_args(validator)
    def __init__(self, iterset, shape, halo=None):
        super().__init__(iterset, shape, halo)

    @cached_property
    def lgmap(self):
        lgmap = PETSc.LGMap()
        if self.comm.size == 1:
            lgmap.create(indices=numpy.arange(self.size, dtype=PETSc.IntType),
                         bsize=self.value_size, comm=self.comm)
        else:
            lgmap.create(indices=self.halo.lgindices,
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
    def validator(self, sets):
        assert all(isinstance(s, AbstractSet) for s in sets)

    @check_args(validator)
    def __init__(self, sets):
        self.sets = tuple(sets)
        self.comm = sets[0].comm


class MixedDataSet(object):
    def validator(self, datasets):
        assert all(isinstance(s, AbstractDataSet) for s in datasets)

    @check_args(validator)
    def __init__(self, datasets):
        self.datasets = tuple(datasets)
        self.comm = datasets[0].comm

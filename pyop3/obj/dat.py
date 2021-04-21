import ctypes
from itertools import chain, zip_longest
from operator import itemgetter

import numpy

from pyop3.api import ArgType
from pyop3.codegen.builder import DatPack
from pyop3.obj.args import AbstractArg
from pyop3.obj.maps import IdentityMap
from pyop3.obj.sets import DataSet
from pyop3.utils import cached_property, check_args, debug_check_args


def noop_exchange(mode):
    pass


class DatArg(AbstractArg):
    lgmaps = None
    argtype = ArgType.DAT

    # This is faster than namedtuples for construction
    # Required because this gives us the key lookup for the code
    # cache.
    # Properties can be slow because they're only inspected for actual
    # codegen.
    class Info(tuple):
        argtype = ArgType.DAT
        packer = property(itemgetter(0))
        shape = property(itemgetter(1))
        dtype = property(itemgetter(2))
        view_index = property(itemgetter(3))
        mapinfo = property(itemgetter(4))

    def validator(self, obj, map_tuple):
        assert isinstance(obj, (Dat, DatView))

    @debug_check_args(validator)
    def __init__(self, obj, map_tuple):
        self._parloop_args_ = obj._parloop_args_
        self._parloop_argtypes_ = obj._parloop_argtypes_
        self.obj = obj
        self.map_tuple = map_tuple

    def validate(self, iterset):
        m, = self.map_tuple
        return m.validate(iterset, self.obj.dataset.iterset)

    @cached_property
    def _codegen_info_(self):
        mapinfo = self.map_tuple[0]._codegen_info_
        return DatArg.Info((self.obj.packer, self.obj.shape,
                           self.obj.dtype,
                           self.obj.view_index,
                           mapinfo))


class DatViewArg(DatArg):
    @cached_property
    def _codegen_info_(self):
        mapinfo = self.map_tuple[0]._codegen_info_
        return DatViewArg.Info((self.obj.packer, self.obj.parent.shape,
                                self.obj.dtype,
                                self.obj.view_index,
                                mapinfo))


class MixedDatArg(AbstractArg):
    argtype = ArgType.MIXED_DAT
    lgmaps = None

    class Info(tuple):
        argtype = ArgType.MIXED_DAT
        datinfos = property(itemgetter(0))
        dtype = property(itemgetter(1))

    def validator(self, obj, map_tuple):
        assert isinstance(obj, MixedDat)

    @debug_check_args(validator)
    def __init__(self, obj, map_tuple):
        """Create an Arg.

        :arg obj: a mixed dat
        :arg map_tuple: 1-tuple of maps through which the dat is accessed (or None).
        """
        self._parloop_args_ = obj._parloop_args_
        self._parloop_argtypes_ = obj._parloop_argtypes_
        self.obj = obj
        self.map_tuple = map_tuple

    def validate(self, iterset):
        m, = self.map_tuple
        return m != IdentityMap and all(m.validate(iterset, d.dataset.toset)
                                        for m, d in zip(m.maps, self.obj.dats))

    @cached_property
    def _codegen_info_(self):
        return MixedDatArg.Info((tuple(DatArg(d, m)._codegen_info_
                                       for d, m in zip_longest(self.obj.dats,
                                                               self.map_tuple)),
                                 self.obj.dtype))


# Vec + VecGhost?
class Dat(object):
    view_index = None
    packer = DatPack
    _parloop_argtypes_ = (ctypes.c_voidp, )

    def validator(self, dataset, dtype, data=None):
        assert isinstance(dataset, DataSet)

    @check_args(validator)
    def __init__(self, dataset, dtype, data=None):
        shape = (dataset.total_size, ) + dataset.shape
        if data is None:
            self._data = numpy.zeros(shape, dtype=dtype)
        else:
            assert data.dtype == dtype
            self._data = data.reshape(shape)
        self.dataset = dataset
        self.comm = dataset.comm
        self.shape = (dataset.total_size, ) + dataset.shape
        self.dtype = self._data.dtype
        self.halo_valid = True
        if self.comm.size == 1:
            self.g2lbegin = noop_exchange
            self.g2lend = noop_exchange
            self.l2gbegin = noop_exchange
            self.l2gend = noop_exchange

    def g2lbegin(self, mode):
        if not self.halo_valid:
            self.dataset.halo.global_to_local_begin(self, mode)

    def g2lend(self, mode):
        if not self.halo_valid:
            self.dataset.halo.global_to_local_end(self, mode)
        self.halo_valid = True

    def l2gbegin(self, mode):
        self.dataset.halo.local_to_global_begin(self, mode)

    def l2gend(self, mode):
        self.dataset.halo.local_to_global_end(self, mode)
        self.halo_valid = False

    @cached_property
    def _parloop_args_(self):
        return (self._data.ctypes.data, )

    def arg(self, map_tuple=(IdentityMap, )):
        return DatArg(self, map_tuple)


class DatView(object):
    packer = DatPack
    _parloop_argtypes_ = (ctypes.c_voidp, )

    def validator(self, parent, view_index):
        assert isinstance(parent, Dat)
        assert len(view_index) == len(parent.shape[1:])
        assert all(0 <= i < d for i, d in zip(view_index, parent.shape[1:]))

    @check_args(validator)
    def __init__(self, parent, view_index):
        view_index = tuple(view_index)
        self.view_index = view_index
        self.parent = parent
        self.slices = (slice(None), *view_index)
        self.shape = self.parent.shape[:1] + (1, )
        self.dataset = self.parent.dataset
        self.g2lbegin = self.parent.g2lbegin
        self.g2lend = self.parent.g2lend
        self.l2gbegin = self.parent.l2gbegin
        self.l2gend = self.parent.l2gend

    @property
    def _data(self):
        return self.parent._data[self.slices]

    @cached_property
    def dtype(self):
        return self.parent.dtype

    @cached_property
    def _parloop_args_(self):
        return self.parent._parloop_args_

    def arg(self, map_tuple=(IdentityMap, )):
        return DatViewArg(self, map_tuple)


class MixedDat(object):
    def validator(self, dats):
        assert all(isinstance(d, Dat) for d in dats)

    @check_args(validator)
    def __init__(self, dats):
        self.dats = tuple(dats)
        self.comm = dats[0].comm
        self.dtype = numpy.find_common_type([], [d.dtype for d in self.dats])
        if self.comm.size == 1:
            self.g2lbegin = noop_exchange
            self.g2lend = noop_exchange
            self.l2gbegin = noop_exchange
            self.l2gend = noop_exchange

    def g2lbegin(self, mode):
        for d in self.dats:
            d.dataset.halo.global_to_local_begin(self, mode)

    def g2lend(self, mode):
        for d in self.dats:
            d.dataset.halo.global_to_local_end(self, mode)

    def l2gbegin(self, mode):
        for d in self.dats:
            d.dataset.halo.local_to_global_begin(self, mode)

    def l2gend(self, mode):
        for d in self.dats:
            d.dataset.halo.local_to_global_end(self, mode)

    @cached_property
    def _parloop_args_(self):
        return tuple(chain(*(d._parloop_args_ for d in self.dats)))

    @cached_property
    def _parloop_argtypes_(self):
        return tuple(chain(*(d._parloop_argtypes_ for d in self.dats)))

    def arg(self, map_tuple):
        return MixedDatArg(self, map_tuple)

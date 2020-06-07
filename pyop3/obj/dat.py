from collections import namedtuple
from itertools import chain, zip_longest

import numpy

from pyop3.codegen.builder import DatPack
from pyop3.obj.args import AbstractArg, ArgType
from pyop3.obj.sets import DataSet
from pyop3.utils import cached_property, check_args, debug_check_args


class DatArg(AbstractArg):
    lgmaps = None
    Info = namedtuple("Info", ["packer", "shape", "dtype", "view_index",
                               "mapinfo"])
    Info.argtype = ArgType.DAT

    def validator(self, obj, map_tuple=None):
        assert isinstance(obj, (Dat, DatView))
        if map_tuple is not None:
            m, = map_tuple
            assert m.toset == obj.dataset.iterset

    @debug_check_args(validator)
    def __init__(self, obj, map_tuple=None):
        self.obj = obj
        if map_tuple is None:
            self.map_tuple = ()
        else:
            self.map_tuple = map_tuple
        self._parloop_args_ = obj._parloop_args_

    def validate(self, iterset):
        if self.map_tuple != ():
            m, = self.map_tuple
            if m.iterset != iterset:
                return False
        return True

    @cached_property
    def _codegen_info_(self):
        mapinfo = self.map_tuple[0]._codegen_info_ if self.map_tuple else None
        return DatArg.Info(packer=self.obj.packer, shape=self.obj.shape,
                           dtype=self.obj.dtype,
                           view_index=self.obj.view_index,
                           mapinfo=mapinfo)


class DatViewArg(DatArg):
    @cached_property
    def _codegen_info_(self):
        mapinfo = self.map_tuple[0]._codegen_info_ if self.map_tuple else None
        return DatViewArg.Info(packer=self.obj.packer, shape=self.obj.parent.shape,
                               dtype=self.obj.dtype,
                               view_index=self.obj.view_index,
                               mapinfo=mapinfo)


class MixedDatArg(AbstractArg):
    lgmaps = None
    Info = namedtuple("Info", ["datinfos", "dtype"])
    Info.argtype = ArgType.MIXED_DAT

    def validator(self, obj, map_tuple=None):
        assert isinstance(obj, MixedDat)
        if map_tuple is not None:
            map_, = map_tuple
            for d, m in zip(obj.dats, map_.maps):
                if m is not None:
                    assert m.toset == d.dataset.iterset

    @debug_check_args(validator)
    def __init__(self, obj, map_tuple=None):
        self.obj = obj
        self._parloop_args_ = obj._parloop_args_
        if map_tuple is None:
            self.map_tuple = ()
        else:
            self.map_tuple = map_tuple

    def validate(self, iterset):
        if self.map_tuple != ():
            map_, = self.map_tuple
            for m in map_.maps:
                if m is not None and m.iterset != iterset:
                    return False
        return True

    @cached_property
    def _codegen_info_(self):
        return MixedDatArg.Info(datinfos=tuple(DatArg(d, m)._codegen_info_
                                               for d, m in zip_longest(self.obj.dats,
                                                                       self.map_tuple)),
                                dtype=self.obj.dtype)


class Dat(object):
    view_index = None
    packer = DatPack

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

    @cached_property
    def _parloop_args_(self):
        return (self._data.ctypes.data, )

    def arg(self, map_tuple=None):
        return DatArg(self, map_tuple=map_tuple)


class DatView(object):
    packer = DatPack

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

    @property
    def _data(self):
        return self.parent._data[self.slices]

    @cached_property
    def dtype(self):
        return self.parent.dtype

    @cached_property
    def _parloop_args_(self):
        return self.parent._parloop_args_

    def arg(self, map_tuple=None):
        return DatViewArg(self, map_tuple=map_tuple)


class MixedDat(object):
    def validator(self, dats):
        assert all(isinstance(d, Dat) for d in dats)

    @check_args(validator)
    def __init__(self, dats):
        self.dats = tuple(dats)
        self.comm = dats[0].comm
        self.dtype = numpy.find_common_type([], [d.dtype for d in self.dats])

    @cached_property
    def _parloop_args_(self):
        return tuple(chain(*(d._parloop_args_ for d in self.dats)))

    def arg(self, map_tuple=None):
        return MixedDatArg(self, map_tuple=map_tuple)

from pyop3.api import AccessMode, ArgType, IterationRegion
from pyop3.codegen.compiled import build_wrapper, get_c_function
from pyop3.obj.kernel import Kernel
from pyop3.obj.maps import IdentityMap
from pyop3.obj.sets import AbstractSet
from pyop3.utils import cached_property, debug_check_args


def filter_args(args, access_modes):
    reductions = []
    seen = set()
    exchange = []
    dirty = []
    mats = []
    for arg, access_mode in zip(args, access_modes):
        if arg.argtype == ArgType.DAT:
            dirty.append(arg.obj)
            if arg.map_tuple != (IdentityMap, ) and arg.obj not in seen:
                exchange.append((arg.obj, access_mode))
                seen.add(arg.obj)
        if arg.argtype == ArgType.GLOBAL and access_mode != AccessMode.READ:
            reductions.append((arg.obj, access_mode))
        if arg.argtype == ArgType.MAT:
            mats.append((arg.obj, access_mode))
    return tuple(exchange), tuple(dirty), tuple(reductions), tuple(mats)


def noop():
    pass


class ParLoop(object):
    def validator(self, kernel, iterset, *args,
                  iteration_region=IterationRegion.ALL,
                  pass_layer_arg=False):
        assert isinstance(kernel, Kernel)
        assert isinstance(iterset, AbstractSet)
        assert len(args) == len(kernel.access_modes)
        assert isinstance(iteration_region, IterationRegion)
        seen = {}
        for arg, access_mode in zip(args, kernel.access_modes):
            assert arg.validate(iterset)
            try:
                assert seen[arg] == access_mode
            except KeyError:
                seen[arg] = access_mode

    @debug_check_args(validator)
    def __init__(self, kernel, iterset, *args,
                 iteration_region=IterationRegion.ALL,
                 pass_layer_arg=False):
        self.args = tuple(args)
        self.kernel = kernel
        self.iterset = iterset
        self.iteration_region = iteration_region
        self.pass_layer_arg = pass_layer_arg
        exchange, dirty, reductions, mats = filter_args(args, kernel.access_modes)
        self.exchange = exchange
        self.dirty = dirty
        self.reductions = reductions
        self.mats = mats
        # Micro-optimisations
        if not reductions or iterset.comm.size == 1:
            self.reduction_begin = noop
            self.reduction_end = noop
        if not exchange or iterset.comm.size == 1:
            self.g2lbegin = noop
            self.g2lend = noop
            self.l2gbegin = noop
            self.l2gend = noop
        if not dirty or iterset.comm.size == 1:
            self.mark_dirty = noop

    def g2lbegin(self):
        for d, mode in self.exchange:
            d.g2lbegin(mode)

    def g2lend(self):
        for d, mode in self.exchange:
            d.g2lend(mode)

    def l2gbegin(self):
        for d, mode in self.exchange:
            d.l2gbegin(mode)

    def l2gend(self):
        for d, mode in self.exchange:
            d.l2gend(mode)

    def reduction_begin(self):
        for g, mode in self.reductions:
            g.reduction_begin(mode)

    def reduction_end(self):
        for g, mode in self.reductions:
            g.reduction_end(mode)

    def mark_dirty(self):
        for d in self.dirty:
            d.halo_valid = False

    def execute(self):
        self.g2lbegin()
        self.dll(0, self.iterset.core_size, *self.c_arglist)
        self.g2lend()
        self.dll(self.iterset.core_size, self.iterset.size, *self.c_arglist)
        self.reduction_begin()
        # self.l2gbegin()
        self.reduction_end()
        # self.l2gend()
        # self.mark_dirty()

    @cached_property
    def _arglist_and_types(self):
        arglist = self.iterset._parloop_args_
        argtypes = self.iterset._parloop_argtypes_
        maptypes = []
        maplist = []
        seen = set()
        for arg in self.args:
            arglist += arg._parloop_args_
            argtypes += arg._parloop_argtypes_
            for map_ in arg.map_tuple:
                for m, t in zip(map_._parloop_args_, map_._parloop_argtypes_):
                    if m in seen:
                        continue
                    seen.add(m)
                    maplist.append(m)
                    maptypes.append(t)
        return arglist + tuple(maplist), argtypes + tuple(maptypes)

    @cached_property
    def c_argtypes(self):
        return self._arglist_and_types[1]

    @cached_property
    def c_arglist(self):
        return self._arglist_and_types[0]

    code_cache = {}

    @cached_property
    def dll(self):
        key = (self.kernel, self.iterset._codegen_info_,
               *(a._codegen_info_ for a in self.args),
               self.iteration_region,
               self.pass_layer_arg)
        try:
            return self.code_cache[key]
        except KeyError:
            wrapper = build_wrapper(*key[:-2],
                                    iteration_region=self.iteration_region,
                                    pass_layer_arg=self.pass_layer_arg)
            dll = get_c_function(wrapper, self.c_argtypes)
            return self.code_cache.setdefault(key, dll)

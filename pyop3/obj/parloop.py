from enum import IntEnum

from pyop3.obj.kernel import Kernel
from pyop3.obj.sets import AbstractSet
from pyop3.utils import cached_property, debug_check_args


class Part(IntEnum):
    CORE = 0
    OWNED = 1


class IterationRegion(IntEnum):
    ALL = 0
    BOTTOM = 1
    TOP = 2
    INTERIOR_FACETS = 3


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

    def execute(self):
        self.g2lbegin()
        self.compute(Part.CORE)
        self.g2lend()
        self.compute(Part.OWNED)
        self.reduction_begin()
        self.l2gbegin()
        self.reduction_end()
        self.l2gend()

    @cached_property
    def c_arglist(self):
        arglist = self.iterset._parloop_args_
        args = self.args
        maplist = []
        seen = set()
        for arg in args:
            arglist += arg._parloop_args_
            for map_ in arg.map_tuple:
                for m in map_._parloop_args_:
                    if m in seen:
                        continue
                    seen.add(m)
                    maplist.append(m)
        return arglist + tuple(maplist)

    def compute(self, part):
        if part == Part.CORE:
            start = 0
            end = self.iterset.core_size
        elif part == Part.OWNED:
            start = self.iterset.core_size
            end = self.iterset.size
        else:
            raise ValueError(f"Unknown part {part}")
        self.dll(start, end, *self.c_arglist())

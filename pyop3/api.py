from enum import IntEnum

from pyop3.obj.args import ArgType  # noqa: F401
from pyop3.obj.kernel import AccessMode  # noqa: F401


class IterationRegion(IntEnum):
    ALL = 0
    BOTTOM = 1
    TOP = 2
    INTERIOR_FACETS = 3

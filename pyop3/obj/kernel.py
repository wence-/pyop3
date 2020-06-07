from enum import IntEnum

from pyop3.utils import check_args

__all__ = ("Kernel", )


class AccessMode(IntEnum):
    READ = 1
    WRITE = 2
    RW = 3
    INC = 4
    MIN = 5
    MAX = 6


class Kernel(object):
    def validator(self, code, name, *access_modes, headers=None,
                  cpp=None, cflags=None, ldflags=None):
        assert all(isinstance(a, AccessMode) for a in access_modes)

    @check_args(validator)
    def __init__(self, code, name, *access_modes, headers=(),
                 cpp=False, cflags=(), ldflags=()):
        self.code = code
        self.cflags = tuple(cflags)
        self.ldflags = tuple(ldflags)
        self.headers = tuple(headers)
        self.name = name
        self.cpp = cpp
        self.access_modes = tuple(access_modes)

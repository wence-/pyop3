import abc
from enum import IntEnum


class ArgType(IntEnum):
    GLOBAL = 0
    DAT = 1
    MIXED_DAT = 2
    MAT = 3
    MIXED_MAT = 4


class AbstractArg(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def validate(self, iterset):
        pass

    @abc.abstractproperty
    def _codegen_info_(self):
        pass

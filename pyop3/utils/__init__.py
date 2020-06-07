from pyop3.rc import configuration


class cached_property(object):
    """
    A read-only @property that is only evaluated once. The value is cached
    on the object itself rather than the function or class.
    """
    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        return obj.__dict__.setdefault(self.__name__, self.fget(obj))


def check_args(predicate):
    def validator(fn):
        def wrapper(*args, **kwargs):
            predicate(*args, **kwargs)
            return fn(*args, **kwargs)
        return wrapper
    return validator


if configuration["runtime_checking"]:
    debug_check_args = check_args
else:
    def debug_check_args(predicate):
        def noop(fn):
            return fn
        return noop

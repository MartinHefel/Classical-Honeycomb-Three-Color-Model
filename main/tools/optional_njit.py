
# General Imports
import functools

# Internal Imports
from main.config import NUMBA_AVAILABLE, USE_NUMBA

if NUMBA_AVAILABLE:
    from numba import njit
else:
    njit = None



def optional_njit(enabled:bool=USE_NUMBA, **njit_kwargs):
    """
    Decorator that applies Numba's njit if enabled and available, otherwise returns the original
    (uncompiled) function.

    Parameters
    ----------
    enabled : bool
        Whether to apply njit compilation.
    **njit_kwargs : dict
        Keyword arguments passed to numba.njit if enabled.
    """
    def decorator(func):
        if enabled and NUMBA_AVAILABLE:
            return njit(**njit_kwargs)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator



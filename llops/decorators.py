

__all__ = ['numpy_function', 'arrayfire_function', 'real_valued_function', 'integer_function']


def integer_function(func):
    """Decorator for functions which operate only on 16-bit integer arrays."""
    from llops import astype, getDatatype, max, min

    def as_integer(*args, **kwargs):

        # Store original datatype
        original_dtype = getDatatype(args[0])

        # Store original range
        extent = min(args[0]), max(args[0])

        # Change first argument (x) to a numpy backend
        args = list(args)
        args[0] = astype(65535.0 * (args[0] - min(args[0])) / max(args[0]), 'uint16')
        args = tuple(args)

        # Call the function
        return astype(func(*args, **kwargs) / 65535.0 * extent[1] + extent[0], original_dtype)

    return as_integer


def numpy_function(func):
    """Decorator for functions which operate only on numpy arrays."""
    from .base import getBackend, cast

    def as_numpy(*args, **kwargs):

        # Store original backend
        original_backend = getBackend(args[0])

        # Change first argument (x) to a numpy backend
        args = list(args)
        args[0] = cast(args[0], backend='numpy')
        args = tuple(args)

        # Call the function
        return cast(func(*args, **kwargs), backend=original_backend)

    return as_numpy


def arrayfire_function(func):
    """Decorator for functions which operate only on numpy arrays."""
    from .base import getBackend, cast

    def as_arrayfire(*args, **kwargs):

        # Store original backend
        original_backend = getBackend(args[0])

        # Change first argument (x) to a numpy backend
        args = list(args)
        args[0] = cast(args[0], backend='arrayfire')
        args = tuple(args)

        # Call the function
        return cast(func(*args, **kwargs), backend=original_backend)

    return as_arrayfire


def real_valued_function(func):
    """Decorator for functions which operate only on real-valued arrays."""
    from .base import real, imag

    def wrapped(*args, **kwargs):
        # Assume x is first argument
        x = args[0]

        # Split x into real and imaginary parts
        x_real, x_imag = real(x), imag(x)

        # Repackage arguments
        args_imag = tuple([x_imag] + list(args[1:]))
        args_real = tuple([x_real] + list(args[1:]))

        # Run function on both real and imaginary parts
        return func(*args_real, **kwargs) + 1j * func(*args_imag, **kwargs)
    return wrapped

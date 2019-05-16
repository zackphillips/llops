"""
Copyright 2019 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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

"""
Copyright 2019 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import scipy as sp
import math
from . import config
import builtins
from functools import reduce
from .decorators import numpy_function, real_valued_function
import skimage

# Allow division by zero to return NaN without a warning
np.seterr(divide='ignore', invalid='ignore')

# Try to import arrayfire - continue if import fails
try:
    import arrayfire
except ImportError:
    pass

# Try to import torch - continue if import fails
try:
    import torch
except ImportError:
    pass


def bcast(X, v):
    """Performs array broadcasting along last dimension of array."""

    backend = getBackend(X)

    if backend == 'numpy':
        return X * v
    elif backend == 'arrayfire':
        mul = lambda A, v: A * v
        return arrayfire.broadcast(mul, X, v.T)
    else:
        raise NotImplementedError(
            'Backend %s is not implemented!' % backend)


def next_even_number(val):
    """Returns the next even number after the input. If input is even, returns the same value."""
    return math.ceil(val / 2) * 2


def next_fast_even_number(val):
    """Returns the next even number after the input. If input is even, returns the same value."""
    _val = val
    while sp.fftpack.next_fast_len(_val) != next_even_number(_val):
        _val = sp.fftpack.next_fast_len(_val + 1)
    return _val


def setArrayfireBackend(af_backend):
    """
    A function which sets the default backend for the arrayfire interface
    """
    if af_backend in arrayfire.get_available_backends():
        arrayfire.set_backend(af_backend)
    else:
        raise ValueError('Backend %d is not supported! Available backends: %s' %
                         (af_backend, arrayfire.get_available_backends()))


def getArrayfireBackend():
    """
    A function that gets the current backend of the arrayfire interface
    """
    return arrayfire.get_active_backend()


def getNativeDatatype(dtype_in, backend):
    """
    A function to get the correct datatype class given a datatype label and backend
    """

    # Check to see if the dtype is of numpy base class
    if type(dtype_in) is not str:
        if 'complex64' in dtype_in:
            dtype_in = 'complex32'
        elif 'complex128' in dtype_in:
            dtype_in = 'complex64'
        else:
            dtype_in = str(dtype_in)

    if backend == 'numpy':
        if dtype_in == 'complex32':
            return np.complex64
        elif dtype_in == 'complex64':
            return np.complex128
        elif dtype_in == 'uint16':
            return np.uint16
        elif dtype_in in config.valid_dtypes:
            return np.dtype(dtype_in)

    elif backend == 'arrayfire':
        if dtype_in == 'complex32':
            return (arrayfire.Dtype.c32)
        elif dtype_in == 'complex64':
            return (arrayfire.Dtype.c64)
        elif dtype_in == 'float32':
            return (arrayfire.Dtype.f32)
        elif dtype_in == 'float64':
            return (arrayfire.Dtype.f64)
        elif dtype_in == 'int16':
            return (arrayfire.Dtype.s16)
        elif dtype_in == 'uint16':
            return (arrayfire.Dtype.u16)
        elif dtype_in == 'int32':
            return (arrayfire.Dtype.s32)
        elif dtype_in == 'uint32':
            return (arrayfire.Dtype.u32)
        elif dtype_in == 'int64':
            return (arrayfire.Dtype.s64)
        elif dtype_in == 'uint64':
            return (arrayfire.Dtype.u64)
        else:
            raise ValueError(
                'Invalid datatype/backend combination (dtype=%s, backend=%s)' %
                (dtype_in, backend))
    elif backend == 'torch':
        if dtype_in == 'complex32':
            raise ValueError('Pytorch does not support complex dtypes.')
        elif dtype_in == 'complex64':
            raise ValueError('Pytorch does not support complex dtypes.')
        elif dtype_in == 'float32':
            return torch.float32
        elif dtype_in == 'float64':
            return torch.float64
        elif dtype_in == 'int16':
            return torch.int16
        elif dtype_in == 'uint16':
            raise ValueError('Pytorch does not support unsigned dtypes.')
        elif dtype_in == 'int32':
            return torch.int32
        elif dtype_in == 'uint32':
            raise ValueError('Pytorch does not support unsigned dtypes.')
        elif dtype_in == 'int64':
            return torch.int64
        elif dtype_in == 'uint64':
            raise ValueError('Pytorch does not support unsigned dtypes.')
        else:
            raise ValueError(
                'Invalid datatype/backend combination (dtype=%s, backend=%s)' %
                (dtype_in, backend))


def getBackend(x):
    """
    This function determines the the backend of a given variable
    """
    if 'numpy' in str(x.__class__):
        return 'numpy'
    elif 'arrayfire' in str(x.__class__):
        return 'arrayfire'
    elif 'torch' in str(x.__class__):
        return 'torch'
    elif str(x.__class__) in [
            "<class 'complex'>", "<class 'float'>", "<class 'int'>"
    ]:
        return 'scalar'  # This is a hack for now, but numpy will treat scalars as if they were numpy types
    elif 'Operator' in str(x.__class__):
        return x.backend
    elif type(x) is list:
        return 'list'
    elif type(x) is tuple:
        return 'tuple'
    elif x is None:
        return None
    else:
        return type(x)
        # raise ValueError("Type %s is not supported!" % (str(x.__class__)))


def getDatatype(x):
    """
    This function determines the the datatype of a given variable in terms of our
    """
    backend = getBackend(x)
    if 'numpy' in backend:
        if 'complex64' in str(x.dtype):
            return 'complex32'
        elif 'complex128' in str(x.dtype):
            return 'complex64'
        else:
            return str(x.dtype)
    elif 'arrayfire' in backend:
        if x.dtype() == arrayfire.Dtype.c32:
            return 'complex32'
        elif x.dtype() == arrayfire.Dtype.c64:
            return 'complex64'
        elif x.dtype() == arrayfire.Dtype.f32:
            return 'float32'
        elif x.dtype() == arrayfire.Dtype.f64:
            return 'float64'
        elif x.dtype() == arrayfire.Dtype.s16:
            return 'int16'
        elif x.dtype() == arrayfire.Dtype.s32:
            return 'int32'
        elif x.dtype() == arrayfire.Dtype.s64:
            return 'int64'
        elif x.dtype() == arrayfire.Dtype.u16:
            return 'uint16'
        elif x.dtype() == arrayfire.Dtype.u32:
            return 'uint32'
        elif x.dtype() == arrayfire.Dtype.u64:
            return 'uint64'
        elif x.dtype() == arrayfire.Dtype.b8:
            return 'bool'
        else:
            raise ValueError("Invalid arrayfire datatype %s" % x.dtype())
    elif 'torch' in backend:
        return str(x.dtype)
    elif 'Operator' in str(x.__class__):
        return x.dtype
    elif type(x) in (list, tuple):
        if isscalar(x[0]):
            return getDatatype(np.asarray(x[0]))
        else:
            return getDatatype(x[0])
    else:
        raise ValueError("Backend %s is not supported!" % (backend))


def dtype(x):
    """Shorthand for getDatatype(x)."""
    return getDatatype(x)


def getByteOrder(x):
    """
    This function returns the byte order of a given array
    """
    backend = getBackend(x)
    if backend == 'numpy':
        if x.flags['F_CONTIGUOUS']:
            return 'F'
        else:
            return 'C'
    elif backend == 'arrayfire':
        return 'F'
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def setByteOrder(x, new_byte_order):
    """
    This function sets the byte order of an array
    """
    backend = getBackend(x)
    if backend == 'numpy':
        if new_byte_order.lower() == 'f':
            return np.asfortranarray(x)
        elif new_byte_order.lower() == 'c':
            return np.ascontiguousarray(x)
        else:
            print('Invalid byte order %s' % new_byte_order)
    elif backend == 'arrayfire':
        if new_byte_order.lower() == 'f':
            return x
        elif new_byte_order.lower() == 'c':
            raise NotImplementedError(
                'Arrayfire does not support C-contiguous arrays!')
        else:
            print('Invalid byte order %s' % new_byte_order)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def makeComplex(dtype_or_array):
    """Makes a datatype or array complex-valued."""

    if isarray(dtype_or_array):
        return astype(dtype_or_array, makeComplex(getDatatype(dtype_or_array)))
    else:
        if dtype_or_array in ('float64', 'complex64'):
            return 'complex64'
        else:
            return 'complex32'


def precision(x, for_sum=False):
    """
    This function returns the precision of a given datatype using a comporable numpy array
    """
    if 'str' in str(type(x)):
        dtype_np = getNativeDatatype(x, 'numpy')
    else:
        dtype_np = getNativeDatatype(getDatatype(x), 'numpy')

    if not for_sum:
        return np.finfo(dtype_np).eps
    else:
        return np.finfo(dtype_np).eps * size(x)


def concatenate(a, b=None, axis=0):
    """
    Generic concatenate operator for two arrays with backend selector
    """

    if b is not None:
        backend = getBackend(a)
        assert backend == getBackend(b)
        if backend == 'numpy':
            return np.append(a, b, axis)
        elif backend == 'arrayfire':
            return arrayfire.data.join(axis, a, b)
        else:
            raise NotImplementedError(
                'Backend %s is not implemented!' % backend)
    elif type(a) is list and b is None:
        backend = getBackend(a[0])
        assert all([backend == getBackend(_a) for _a in a])
        if backend == 'numpy':
            return np.concatenate(a)
        elif backend == 'arrayfire':
            result = a[0]
            for _a in a[1:]:
                result = arrayfire.data.join(axis, result, _a)
            return result
        else:
            raise NotImplementedError(
                'Backend %s is not implemented!' % backend)


def norm(x):
    """
    A generic norm operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.linalg.norm(x)
    elif backend == 'arrayfire':
        return arrayfire.lapack.norm(x)
    elif backend == 'torch':
        return x.norm(p=2)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def sign(x):
    """
    A generic sign operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.sign(x)
    elif backend == 'arrayfire':
        s = x / arrayfire.arith.sqrt(x * x)
        s[isnan(s)] = 0
        return s
    elif backend == 'torch':
        return x.sign()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def abs(x, return_real=True):
    """
    A generic absolute value operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'scalar':
        if not return_real:
            return np.abs(x).astype(x.dtype)
        else:
            return np.abs(x)
    elif backend == 'numpy':
        if not return_real:
            return np.abs(x).astype(x.dtype)
        else:
            return np.abs(x)
    elif backend == 'arrayfire':
        if not return_real:
            return arrayfire.arith.abs(x).as_type(x.dtype())
        else:
            return arrayfire.arith.abs(x)
    elif backend == 'torch':
        return x.abs()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def angle(x):
    """
    A generic complex angle operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.angle(x)
    elif backend == 'arrayfire':
        # The following two lines should be equilavent
        # return arrayfire.arith.imag(arrayfire.arith.log(x))
        return arrayfire.arith.atan2(
            arrayfire.arith.imag(x), arrayfire.arith.real(x))
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def isComplex(x, check_values=True):
    """
    Checks if x is complex
    """
    if check_values:
        return sum(imag(x)) > 0
    else:
        return 'complex' in getDatatype(x)


def real(x):
    """
    A generic real-part operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.real(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.real(x)
    elif backend == 'scalar':
        return np.real(x)
    elif backend == 'list':
        return np.real(np.asarray(x)).tolist()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def imag(x):
    """
    A generic real-part operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.imag(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.imag(x)
    elif backend == 'list':
        return np.imag(np.asarray(x)).tolist()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def log(x):
    """
    Natural log with backend selector.
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.log(x)
    elif backend == 'numpy':
        return np.log(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.log(x)
    elif backend == 'torch':
        return x.log()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def log10(x):
    """
    Natural log with backend selector.
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.log10(x)
    elif backend == 'numpy':
        return np.log10(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.log10(x)
    elif backend == 'torch':
        return x.log10()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def sqrt(x):
    """
    A generic sqrt operator with backend selector
    """
    backend = getBackend(x)

    if backend == 'scalar':
        return np.sqrt(x)
    elif backend == 'numpy':
        # assert np.all(x >= 0.0), '%s' % x
        return np.sqrt(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.sqrt(x)
    elif backend == 'torch':
        return x.sqrt()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def cos(x):
    """
    A generic cosine operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.cos(x)
    elif backend == 'numpy':
        return np.cos(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.cos(x)
    elif backend == 'torch':
        return x.cos()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def sin(x):
    """
    A generic sine operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.sin(x)
    elif backend == 'numpy':
        return np.sin(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.sin(x)
    elif backend == 'torch':
        return x.sin()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def tan(x):
    """
    A generic tangent operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.tan(x)
    elif backend == 'numpy':
        return np.tan(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.tan(x)
    elif backend == 'torch':
        return x.tan()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def arccos(x):
    """
    A generic cosine operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.arccos(x)
    elif backend == 'numpy':
        return np.arccos(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.acos(x)
    elif backend == 'torch':
        return x.acos()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def arcsin(x):
    """
    A generic sine operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.arcsin(x)
    elif backend == 'numpy':
        return np.arcsin(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.asin(x)
    elif backend == 'torch':
        return x.asin()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def arctan(x):
    """
    A generic tangent operator with backend selector
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.arctan(x)
    elif backend == 'numpy':
        return np.arctan(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.atan(x)
    elif backend == 'torch':
        return x.atan()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def isnan(x):
    """
    This function returns a boolean array indicating the location of NaN values
    """

    # Get the backend of the variable
    backend = getBackend(x)

    if backend == 'scalar':
        return np.isnan(x)
    elif backend == 'numpy':
        return np.isnan(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.isnan(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def exp(x):
    """
    A generic exp operator with backend selector
    TODO (sarah) check for numerical overflow
    """
    backend = getBackend(x)
    if backend == 'scalar':
        return np.exp(x)
    elif backend == 'numpy':
        return np.exp(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.exp(x)
    elif backend == 'torch':
        return x.exp()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def unique(x):
    """
    Returns unique elements in an array.

    Parameters
    ----------
    x: array-like
        The array to find unique elements of.

    Returns
    -------
    array-like:
        The unique elements in the array.

    """
    backend = getBackend(x)
    if backend == 'scalar':
        return x
    elif backend == 'list':
        return np.unique(np.asarray(x)).tolist()
    elif backend == 'numpy':
        return np.unique(x)
    elif backend == 'torch':
        return x.unique()
    elif backend == 'arrayfire':
        return arrayfire.algorithm.set_unique(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def zeros(shape, dtype=None, backend=None):
    """
    Returns an array of zeros.

    Parameters
    ----------
    shape: list or tuple
        The desired shape of the array
    dtype: string
        Optional. The desired datatype, if different from the default.
    backend: string
        Optional. The desired backend, if different from the default.

    Returns
    -------
    array-like:
        An array of zeros of the desired shape, dtype, and backend.

    """

    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    if type(shape) not in [list, tuple, np.ndarray]:
        shape = [shape]
    _dtype = getNativeDatatype(dtype, backend)
    if backend == 'numpy':
        return np.asfortranarray(np.zeros(shape, _dtype))
    elif backend == 'arrayfire':
        if len(shape) == 1:
            return arrayfire.data.constant(0, shape[0], dtype=_dtype)
        elif len(shape) == 2:
            return arrayfire.data.constant(0, shape[0], shape[1], dtype=_dtype)
        elif len(shape) == 3:
            return arrayfire.data.constant(
                0, shape[0], shape[1], shape[2], dtype=_dtype)
        else:
            raise NotImplementedError
    elif backend == 'torch':
        return torch.zeros(shape, dtype=_dtype)
    elif backend == 'list':
        return [0] * prod(shape)
    elif backend == 'tuple':
        return tuple([0] * prod(shape))
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def zeros_like(x):
    """
    Returns an array of zeros. This array has the same shape, datatype, and
    backend as the input.

    Parameters
    ----------
    x: array-like
        The array to draw paremeters from.

    Returns
    -------
    array-like:
        An array of zeros with the same shape, datatype, and backend as the input.

    """
    return zeros(shape(x), getDatatype(x), getBackend(x))


def ones(shape, dtype=None, backend=None):
    """
    Returns an array of ones.

    Parameters
    ----------
    shape: list or tuple
        The desired shape of the array
    dtype: string
        Optional. The desired datatype, if different from the default.
    backend: string
        Optional. The desired backend, if different from the default.

    Returns
    -------
    array-like:
        An array of ones of the desired shape, dtype, and backend.

    """
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    if type(shape) not in [list, tuple, np.ndarray]:
        shape = [shape]
    _dtype = getNativeDatatype(dtype, backend)
    if backend == 'numpy':
        return np.asfortranarray(np.ones(shape, _dtype))
    elif backend == 'arrayfire':
        if len(shape) == 1:
            return arrayfire.data.constant(1, shape[0], dtype=_dtype)
        elif len(shape) == 2:
            return arrayfire.data.constant(1, shape[0], shape[1], dtype=_dtype)
        elif len(shape) == 3:
            return arrayfire.data.constant(
                1, shape[0], shape[1], shape[2], dtype=_dtype)
        else:
            raise NotImplementedError
    elif backend == 'torch':
        return torch.ones(shape, dtype=_dtype)
    elif backend == 'list':
        # TODO support for floats
        if 'complex' in dtype:
            return [1 + 0j] * prod(shape)
        else:
            return [1] * prod(shape)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def ones_like(x):
    """
    Returns an array of ones. This array has the same shape, datatype, and
    backend as the input.

    Parameters
    ----------
    x: array-like
        The array to draw paremeters from.

    Returns
    -------
    array-like:
        An array of ones with the same shape, datatype, and backend as the input.

    """
    return ones(shape(x), getDatatype(x), getBackend(x))


def randn(shape, dtype=None, backend=None):
    """
    Returns an array of random values drawn from a normal distribution.

    Parameters
    ----------
    shape: list or tuple
        The desired shape of the array
    dtype: string
        Optional. The desired datatype, if different from the default.
    backend: string
        Optional. The desired backend, if different from the default.

    Returns
    -------
    array-like:
        An array of random values drawn from the normal distribution.

    """
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    if type(shape) not in [list, tuple, np.ndarray]:
        shape = [shape]
    _dtype = getNativeDatatype(dtype, backend)
    if backend == 'numpy':
        if len(shape) == 1:
            return np.asfortranarray(np.random.randn(shape[0]).astype(_dtype))
        elif len(shape) == 2:
            return np.asfortranarray(
                np.random.randn(shape[0], shape[1]).astype(_dtype))
        elif len(shape) == 3:
            return np.asfortranarray(
                np.random.randn(shape[0], shape[1],
                                shape[2]).astype(_dtype).asfortranarray())
        else:
            raise NotImplementedError

    elif backend == 'arrayfire':
        if len(shape) == 1:
            return arrayfire.random.randn(shape[0], dtype=_dtype)
        elif len(shape) == 2:
            return arrayfire.random.randn(shape[0], shape[1], dtype=_dtype)
        elif len(shape) == 3:
            return arrayfire.random.randn(
                shape[0], shape[1], shape[2], dtype=_dtype)
        else:
            raise NotImplementedError

    elif backend == 'torch':
        return torch.randn(shape, dtype=_dtype)

    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def randn_like(x):
    """
    Returns an array of random values drawn from a normal distribution. This
    array has the same shape, datatype, and backend as the input.

    Parameters
    ----------
    x: array-like
        The array to draw paremeters from.

    Returns
    -------
    array-like:
        An array of random values drawn from the normal distribution with the
        same shape, datatype, and backend as the input.

    """
    return randn(shape(x), getDatatype(x), getBackend(x))


def randu(shape, dtype=None, backend=None):
    """
    Returns an array of random values drawn from a uniform distribution.

    Parameters
    ----------
    shape: list or tuple
        The desired shape of the array
    dtype: string
        Optional. The desired datatype, if different from the default.
    backend: string
        Optional. The desired backend, if different from the default.

    Returns
    -------
    array-like:
        An array of random values drawn from the uniform distribution.

    """
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    if type(shape) not in [list, tuple, np.ndarray]:
        shape = [shape]
    _dtype = getNativeDatatype(dtype, backend)
    if backend == 'numpy':
        if len(shape) == 1:
            return np.asfortranarray(np.random.rand(shape[0]).astype(_dtype))
        elif len(shape) == 2:
            return np.asfortranarray(
                np.random.rand(shape[0], shape[1]).astype(_dtype))
        elif len(shape) == 3:
            return np.asfortranarray(
                np.random.rand(shape[0], shape[1], shape[2]).astype(_dtype))
        else:
            raise NotImplementedError

    elif backend == 'arrayfire':
        if len(shape) == 1:
            return arrayfire.random.randu(shape[0], dtype=_dtype)
        elif len(shape) == 2:
            return arrayfire.random.randu(shape[0], shape[1], dtype=_dtype)
        elif len(shape) == 3:
            return arrayfire.random.randu(
                shape[0], shape[1], shape[2], dtype=_dtype)
        else:
            raise NotImplementedError

    elif backend == 'torch':
        return torch.rand(shape, dtype=_dtype)

    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def randu_like(x):
    """
    Returns an array of random values drawn from a uniform distribution. This
    array has the same shape, datatype, and backend as the input.

    Parameters
    ----------
    x: array-like
        The array to draw paremeters from.

    Returns
    -------
    array-like:
        An array of random values drawn from the uniform distribution with the
        same shape, datatype, and backend as the input.

    """
    return randu(shape(x), getDatatype(x), getBackend(x))


def rand(shape, dtype=None, backend=None):
    """
    Returns an array of random values drawn from a uniform distribution.
    Internally, this function calls the randu function.

    Parameters
    ----------
    shape: list or tuple
        The desired shape of the array
    dtype: string
        Optional. The desired datatype, if different from the default.
    backend: string
        Optional. The desired backend, if different from the default.

    Returns
    -------
    array-like:
        An array of random values drawn from the uniform distribution.

    """
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype
    return (randu(shape, dtype, backend))


def rand_like(x):
    """
    Returns an array of random values drawn from a uniform distribution. This
    array has the same shape, datatype, and backend as the input.

    Parameters
    ----------
    x: array-like
        The array to draw paremeters from.

    Returns
    -------
    array-like:
        An array of random values drawn from the uniform distribution with the
        same shape, datatype, and backend as the input.

    """
    return rand(shape(x), getDatatype(x), getBackend(x))


def where(x):
    """
    Returns a list of locations with non-zero values in an array

    Parameters
    ----------
    x: array-like
        The array to search for non-zero values

    Returns
    -------
    tuple:
        Tuple of positions in an array (one tuple for position)

    """
    # Get backend
    backend = getBackend(x)

    # Get precision
    tol = precision(x)

    if backend == 'numpy':
        return tuple([(i[0], i[1]) for i in np.asarray(np.where(np.abs(x) > tol)).T])
    elif backend == 'arrayfire':
        return tuple([
            tuple(reversed(np.unravel_index(i, tuple(reversed(x.shape)))))
            for i in np.asarray(arrayfire.algorithm.where(abs(x) > tol))
        ])
    elif backend == 'torch':
        return torch.where(x.abs() > tol, 0, 1)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def max(x, axis=None):
    """
    Returns the maximum of an array across all dimensions.

    Parameters
    ----------
    x: array-like
        The array to evaluate the maximum of. Evaluates only the real values of
        the input.

    Returns
    -------
    float:
        The maximum real value of the array across all dimensions.

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.max(real(x), axis=axis)
    elif backend == 'arrayfire':
        return scalar(arrayfire.algorithm.max(real(x), dim=axis))
    elif backend == 'scalar':
        return x
    elif backend in ['list', 'tuple']:
        return builtins.max(x)
    elif backend == 'torch':
        return torch.max(x, dim=axis)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def amax(x):
    """Short-hand for llops.max."""
    return max(x)


def argmax(x, axis=None):
    """
    Returns the coordinates of the global maximum of an array. Only considers
    the real part of the array.

    Parameters
    ----------
    x: array-like
        The first vector to dot product

    Returns
    -------
    tuple:
        The coordinates of the mininum value

    """
    backend = getBackend(x)
    _shape = shape(x)
    if backend == 'numpy':
        return tuple(np.unravel_index(np.argmax(real(x)), _shape))
    elif backend == 'arrayfire':
        return tuple(np.unravel_index(arrayfire.algorithm.imax(real(x.T))[1], _shape))
    elif backend == 'scalar':
        return x
    elif backend in ('list', 'tuple'):
        return argmax(np.asarray(x), axis=axis)
    elif backend == 'torch':
        return torch.argmax(x, dim=axis)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def min(x, axis=None):
    """
    Returns the minimum of an array across all dimensions.

    Parameters
    ----------
    x: array-like
        The array to evaluate the minimum of. Evaluates only the real values of
        the input.

    Returns
    -------
    float:
        The minimum real value of the array across all dimensions.

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.min(real(x), axis=axis)
    elif backend == 'arrayfire':
        return scalar(arrayfire.algorithm.min(real(x), dim=axis))
    elif backend == 'scalar':
        return x
    elif backend in ['list', 'tuple']:
        return builtins.min(x)
    elif backend == 'torch':
        return torch.min(x, dim=axis)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def amin(x, axis=None):
    """Short-hand for llops.min."""
    return min(x, axis=axis)


def argmin(x, axis=None):
    """
    Returns the coordinates of the global mininum of an array. Only conisders
    the real part of the input.

    Parameters
    ----------
    x: array-like
        The first vector to dot product

    Returns
    -------
    tuple:
        The coordinates of the mininum value

    """
    backend = getBackend(x)
    _shape = shape(x)
    if backend == 'numpy':
        return tuple(np.unravel_index(np.argmin(real(x)), _shape))
    elif backend == 'arrayfire':
        return tuple(
            np.unravel_index(arrayfire.algorithm.imin(real(x.T))[1], _shape))
    elif backend == 'scalar':
        return x
    elif backend == 'torch':
        return torch.argmin(x, dim=axis)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def conj(x):
    """
    Element-wise complex conjugate of an array

    Parameters
    ----------
    x: array-like
        The first vector to dot product

    Returns
    -------
    array-like:
        The complex conjugate of x

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.conj(x)
    elif backend == 'arrayfire':
        return arrayfire.arith.conjg(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def dot(lhs, rhs):
    """
    Dot product of two operators

    Parameters
    ----------
    lhs: array-like
        The first vector to dot product
    rhs: array-like
        The second vector to dot product

    Returns
    -------
    array-like:
        The dot product of lhs and rhs

    """
    backend = getBackend(rhs)

    if backend == 'numpy' or backend == 'arrayfire' or backend == 'torch':
        return sum(lhs * rhs)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def flip(x, axis=None):
    """
    Flip an operator about an axis.

    Parameters
    ----------
    x: array-like
        The array we wish to fftshift.
    axes:
        Optional. Axes to flip across.

    Returns
    -------
    array-like:
        The flipped array

    """
    backend = getBackend(x)

    # Parse axes
    if axis is None:
        axis = list(range(ndim(x)))
    elif type(axis) is not list:
        axis = [axis]

    # If no axes are provided, just return x
    if len(axis) == 0:
        return x

    if backend == 'numpy':
        for ax in axis:
            x = np.flip(x, ax)
        return x
    elif backend == 'arrayfire':
        for ax in axis:
            x = arrayfire.data.flip(x, ax)
        return x
    elif backend == 'torch':
        return torch.flip(x, dim=axis)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def fliplr(x):
    """Helper function to flip an array along second dimension."""
    return flip(x, axis=1)


def flipud(x):
    """Helper function to flip an array along first dimension."""
    return flip(x, axis=0)


def roll(x, shift, axis=None, y=None):
    """
    Roll an array about an axis

    This function rolls an array along one or more axes.

    Parameters
    ----------
    x: array-like
        The array we wish to fftshift
    shift: list, tuple
        Amount to shift
    axes:
        Optional. Axes to shift along

    Returns
    -------
    array-like:
        The roll array

    """
    backend = getBackend(x)

    # Deal with lists using recursion
    if getBackend(shift) in ('list', 'tuple', 'arrayfire', 'numpy'):
        if axis is None:
            axis = range(len(shift))

        for sh, ax in zip(shift, axis):
            x = roll(x, int(real(sh)), ax)

        return x

    # Set axis to default
    if axis is None:
        axis = 0
    else:
        assert axis in range(ndim(x)), 'Axis %s is invalid' % str(axis)

    if backend == 'numpy':
        return np.roll(x, shift, axis)
    elif backend == 'arrayfire':
        if axis == 0:
            return arrayfire.data.shift(x, shift)
        elif axis == 1:
            return arrayfire.data.shift(x, 0, shift)
        elif axis == 2:
            return arrayfire.data.shift(x, 0, 0, shift)
        else:
            raise NotImplementedError
    elif backend in ('list', 'tuple'):
        return x[-shift:] + x[:-shift]
    elif backend == 'torch':
        if shift < 0:
            shift = -shift
            gap = x.index_select(axis, torch.arange(shift))
            return torch.cat([x.index_select(axis, torch.arange(shift, x.size(axis))), gap], dim=axis)

        else:
            shift = x.size(axis) - shift
            gap = x.index_select(axis, torch.arange(shift, x.size(axis)))
            return torch.cat([gap, x.index_select(axis, torch.arange(shift))], dim=axis)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def circshift(x, shift, axis=None):
    """
    Circular shift an array.

    This function performs a circshift operation, which rolls an array along one
    or more axes.

    Parameters
    ----------
    x: array-like
        The array we wish to fftshift
    shift: list, tuple
        Amount to shift
    axes:
        Optional. Axes to shift along

    Returns
    -------
    array-like:
        The circshifted array

    """
    return roll(x, shift, axis)


def fftshift(x, inverse=False):
    """
    FFT shift an array.

    This function performs a fftshift operation. It is the same as ifftshift
    for arrays with even shapes.

    Parameters
    ----------
    x: array-like
        The array we wish to fftshift
    inverse: bool
        Whether to inverse (ifftshift) the array

    Returns
    -------
    array-like:
        The fft-shifted array

    """
    backend = getBackend(x)
    if backend == 'numpy':
        if inverse:
            return sp.fftpack.ifftshift(x)
        else:
            return sp.fftpack.fftshift(x)
    elif backend == 'arrayfire':
        if inverse:
            s = [math.floor(i / 2) for i in x.shape]
        else:
            s = [math.ceil(i / 2) for i in x.shape]

        if len(s) == 1:
            return arrayfire.data.shift(x, s[0])
        elif len(s) == 2:
            return arrayfire.data.shift(x, s[0], s[1])
        elif len(s) == 3:
            return arrayfire.data.shift(x, s[0], s[1], s[2])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def ifftshift(x):
    """
    Inverse FFT shift an array.

    This function performs an ifftshift operation. It is the same as fftshift
    for arrays with even shapes.

    Parameters
    ----------
    x: array-like
        The array we wish to fftshift

    Returns
    -------
    array-like:
        The inverse fft-shifted array

    """
    return fftshift(x, True)


def transpose(x, hermitian=True):
    """
    Transpose an array.

    This function performs calculates the transpose of an array. It returns
    the hermitian transpose by default.

    Parameters
    ----------
    x: array-like
        The array we wish to tranpose.
    hermitian: bool
        Whether to conjigate the array in additon to transposing it.

    Returns
    -------
    array-like:
        The (hermitian) transposed array.

    """
    backend = getBackend(x)
    if backend == 'numpy':
        if hermitian:
            return np.conj(x.T)
        else:
            return x.T
    elif backend == 'arrayfire':
        return arrayfire.array.transpose(x, conj=hermitian)
    elif backend == 'torch':
        return x.transpose(-1,0)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def matmul(A, x, y=None):
    """
    Matrix-vector multiplication.

    This function performs matrix-vector multiplication.

    Parameters
    ----------
    A : array-like
        The matrix to multiply
    x : array-like
        The vector (array) we wish to multiply. This will be vectorized if it
        is not already.
    y: array-like
        Optional. Output to write to.

    Returns
    -------
    array-like:
        The output of the matrix-vector multiplcation or None if y is not None.

    """
    backend = getBackend(x)

    if backend == 'numpy':
        if y is not None:
            np.matmul(A, x, y)
        else:
            return np.matmul(A, x)
    elif backend == 'arrayfire':
        if y is not None:
            y[:] = arrayfire.blas.matmul(A, x, y)
        else:
            return arrayfire.blas.matmul(A, x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def innerproduct(x, y):
    """
    A generic inner product operation with backend selector
    """
    return matmul(transpose(x, conj=True), y)


def all(x):
    """
    Returns whether all values are True in the input.
    Analyzes the real values only.

    Parameters
    ----------
    x: array-like or scalar
        The array to search for boolean true values

    Returns
    -------
    bool:
        Whether all of the values in the array are True.

    """

    # Check if x is a boolean
    if type(x) is bool:
        return x
    else:
        backend = getBackend(x)

        if backend == 'numpy':
            return np.any(real(x))
        elif backend == 'arrayfire':
            return arrayfire.algorithm.all_true(real(x)) != 0
        elif backend in ['list', 'tuple']:
            return builtins.all(x)
        else:
            raise NotImplementedError('Backend %s is not implemented!' % backend)


def any(x):
    """
    Returns whether any values are True in the input.
    Analyzes the real values only.

    Parameters
    ----------
    x: array-like or scalar
        The array to search for boolean true values

    Returns
    -------
    bool:
        Whether any of the values in the array are True.

    """
    # Check if x is a boolean
    if type(x) is bool:
        return x
    else:
        backend = getBackend(x)
        if backend == 'numpy':
            return np.all(real(x))
        elif backend == 'arrayfire':
            return arrayfire.algorithm.any_true(real(x)) != 0
        elif backend in ['list', 'tuple']:
            return builtins.any(x)
        else:
            raise NotImplementedError('Backend %s is not implemented!' % backend)


def prod(x, axes=None):
    """
    A generic product operator with backend selector.
    """
    backend = getBackend(x)

    if backend in ['tuple', 'list']:
        return reduce(lambda x, y: x * y, x) if len(x) > 0 else []

    if backend in ['numpy', 'tuple', 'list']:
        if axes is None:
            return np.prod(x)
        else:
            return np.prod(x, axis=tuple(axes), keepdims=True)
    elif backend == 'arrayfire':
        if axes is None:
            _axes = list(range(len(shape(x))))
        else:
            _axes = axes

        # Sum over defined axes
        a = x.copy()
        for axis in _axes:
            a = arrayfire.algorithm.prod(a, axis)

        if axes is None:
            return scalar(a.as_type(x.dtype()))
        else:
            return a.as_type(x.dtype())
    elif backend == 'scalar':
        return x
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def sum(x, axis=None):
    """
    A generic sum operator with backend selector
    """
    # Ensure axis is a list
    if axis is not None and type(axis) not in [list, tuple, np.ndarray]:
        axis = [axis]

    backend = getBackend(x)
    if backend == 'numpy':
        if axis is None:
            return np.sum(x)
        else:
            return np.sum(x, axis=tuple(axis), keepdims=True)
    elif backend == 'arrayfire':
        if axis is None:
            _axes = list(range(len(shape(x))))
        else:
            _axes = axis

        # Sum over defined axes
        a = x.copy()
        for axis in _axes:
            a = arrayfire.algorithm.sum(a, axis)

        if axis is None:
            return scalar(a.as_type(x.dtype()))
        else:
            return a.as_type(x.dtype())
    elif backend == 'list':
        return builtins.sum(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def tile(x, reps):
    """
    A generic tile operation with backend selector
    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.tile(x, reps)
    elif backend == 'arrayfire':
        if len(reps) == 1:
            return arrayfire.data.tile(x, int(reps[0]))
        elif len(reps) == 2:
            return arrayfire.data.tile(x, int(reps[0]), int(reps[1]))
        elif len(reps) == 3:
            return arrayfire.data.tile(x, int(reps[0]), int(reps[1]),
                                       int(reps[2]))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def isscalar(x):
    """ TODO: Only works for numpy arrays """
    return not hasattr(x, "__len__")


def shape(x, ndim=None):
    """
    A method which returns the shape of an array in row-major format
    """
    backend = getBackend(x)

    if backend == 'numpy':
        if isscalar(x):
            return (1,)
        else:
            return tuple(np.asarray(x).shape)
    elif backend == 'arrayfire':
        # Arrayfire arrays ALWAYS have 4 dimensions. The .shape property squeezes
        # out all extra dimensions, which is inconsistent with numpy.shape.
        # The ndim parameter compensates for this by enforcing the return of a
        # tuple of len ndim regardless of the number of dimensions used.
        _shape = x.shape
        if ndim is not None:
            if len(_shape) != ndim:
                _shape = tuple(list(_shape) + [1] * (ndim - len(_shape)))
        return _shape
    elif backend == 'scalar':
        return (1,)
    elif backend == 'torch':
        return tuple(x.shape)
    elif backend in ['tuple', 'list']:
        return len(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def scalar(x):
    """
    A method which returns the first value of the array as either a complex or float
    """
    if isscalar(x):
        return x
    else:
        backend = getBackend(x)
        datatype = getDatatype(x)
        if backend in ['numpy', 'arrayfire', 'torch']:
            if 'complex' in datatype:
                return np.complex(np.asarray(x).item(0))
            else:
                return np.float32(np.asarray(x).item(0))
        else:
            raise NotImplementedError('Backend %s is not implemented!' % backend)


def changeBackend(x, new_backend=None):
    """
    A method which converts an array to the given backend
    """

    # Use default backend by default
    if new_backend is None:
        new_backend = config.default_backend

    # Deal with tuples and lists
    if type(x) in (list, tuple):
        x = [changeBackend(_x) for _x in x]

    # Get current backend
    current_backend = getBackend(x)

    # Check and change backend
    if new_backend == current_backend:
        return x
    else:
        if current_backend == 'numpy' and new_backend == 'arrayfire':
            """ Numpy to arrayfire """
            return arrayfire.interop.np_to_af_array(x)
        elif current_backend == 'numpy' and new_backend == 'torch':
            """ Numpy to pytorch """
            return torch.from_numpy(x)
        elif current_backend == 'numpy' and new_backend == 'list':
            return x.toList()
        elif current_backend == 'numpy' and new_backend == 'tuple':
            return tuple(x.toList())
        elif current_backend == 'arrayfire':
            """ arrayfire to numpy """
            return changeBackend(x.__array__(), new_backend)
        elif current_backend in ("list", "tuple", "scalar"):
            """ List/tuple to any other backend."""
            return changeBackend(np.asarray(x), new_backend)
        elif current_backend is 'torch':
            return changeBackend(x.numpy(), new_backend)
        else:
            raise ValueError(
                "Array with backend %s cannot be converted to new backend %s" %
                (current_backend, new_backend))


def asbackend(x, new_backend=None):
    """ Wrapper class for changeBackend for convenience """
    return changeBackend(x, new_backend)


def asarray(x, dtype=None, backend=None):
    """ Wrapper class for changeBackend for convenience """

    # Ensure output is complex if input is complex
    if "complex" in getDatatype(x):
        dtype = "complex32"

    # If x is a list, convert to numpy first, then to the appropriate dtype
    if type(x) in (list, tuple):
        x = changeBackend(np.asarray(x), backend)
    else:
        x = changeBackend(x, backend)

    # Convert datatype
    x = astype(x, dtype)

    # Return
    return x


def astype(x, new_dtype=None):
    """
    A method which converts an array to the given datatype
    """

    # Use default backend if no argument passed
    if new_dtype is None:
        new_dtype = config.default_dtype

    # Pre-convert to tuples or lists
    if type(x) in [tuple, list]:
        x = np.asarray(x)

    # Get current backend and datatype
    backend = getBackend(x)
    current_dtype = getDatatype(x)

    if new_dtype == current_dtype:
        return x  # No change
    else:
        if backend == 'numpy':
            # Take the real part if we're converting from complex to real
            if 'complex' in current_dtype and 'complex' not in new_dtype:
                x = real(x)

            return x.astype(getNativeDatatype(new_dtype, 'numpy'))
        elif backend == 'arrayfire':
            if 'complex' in getDatatype(x) and 'complex' not in new_dtype:
                return arrayfire.arith.cast(
                    real(x), getNativeDatatype(new_dtype, 'arrayfire'))
            else:
                return arrayfire.arith.cast(
                    x, getNativeDatatype(new_dtype, 'arrayfire'))
        else:
            raise ValueError(
                "Array with backend %s cannot be operated on" % (backend))


def isarray(x):
    """
    Determines if the input is an array.

    Parameters
    ----------
    x: object
        The object to observe.

    Returns
    -------
    bool:
        True if the array has one of the valid backends of this package.

    """
    return getBackend(x) in config.valid_backends


def cast(x, dtype=None, backend=None):
    """
    Casts an object to a specific dtype and backend
    """

    if backend is None:
        backend = config.default_backend

    if dtype is None:
        dtype = config.default_dtype

    return astype(asbackend(x, backend), dtype)


def cast_like(x, template):
    """
    Casts an input array to be the same dtype and backend as the second input.

    Parameters
    ----------
    x: array-like
        The array to cast.
    template: array-like
        The array to use as a template for backend and dtype.

    Returns
    -------
    array-like:
        The first input cast to be like the second input

    """
    # Get backend and dtype of template
    template_backend = getBackend(template)
    template_dtype = getDatatype(template)

    # Return casted array
    return cast(x, dtype=template_dtype, backend=template_backend)


def reshape(x, N, no_warnings=False):
    """
    A method which vectorizes an array
    """
    # If array is already the same shape, just return
    if tuple(N) == shape(x):
        return x

    if type(N) not in [list, tuple, np.ndarray]:
        N = [N]
    elif type(N) is np.ndarray:
        N = N.tolist()

    # If array is already the same shape, just return
    if tuple(N) == shape(x):
        return x

    # If this is just a dimension expansion, call expandDims
    if len(N) > ndim(x) and N[-1] is 1:
        return expandDims(x, len(N))

    # Store existing backend
    backend = getBackend(x)

    # Check that number of elements is consistent
    assert np.prod(N) == size(x), "Number of elements is not consistent (size(x)=%d, N=%d)" % (np.prod(N), size(x))

    if config.WARN_FOR_EXPENSIVE_OPERATIONS and not no_warnings:
        print("WARNING: calling reshape can be an expensive operation, it is normally advised to avoid this.")

    if backend == 'numpy':
        return np.reshape(x, N)
    elif backend == 'arrayfire':
        if len(N) is 1:
            return vectorize(x)
        elif len(N) is 2:
            y = arrayfire.transpose(
                arrayfire.moddims(arrayfire.transpose(x), N[1], N[0]))
            garbageCollect(backend)
            return y
        elif len(N) is 3:
            y = arrayfire.transpose(
                arrayfire.moddims(arrayfire.transpose(x), N[2], N[1], N[0]))
            garbageCollect(backend)
            return y
    elif backend == 'torch':
        return x.view(N)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def garbageCollect(backend=None):
    """
    A method which calls garbage collect for the selected backend
    """

    # If backend is not provided, call for all available backends
    if backend is None:
        for _backend in config.valid_backends:
            garbageCollect(_backend)
        return

    if backend == 'numpy':
        return
    elif backend == 'arrayfire':
        arrayfire.device.device_gc()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def squeeze(x):
    """
    A method which removes extra dimensions from an array
    """
    backend = getBackend(x)

    if backend == 'numpy':
        return np.squeeze(x)
    elif backend == 'arrayfire':
        dims = list(range(4))
        for dim in range(ndim(x)):
            if shape(x)[dim] <= 1:
                dims.append(dims.pop(dims[dim]))

        return arrayfire.data.reorder(x, dims[0], dims[1], dims[2], dims[3])
    elif backend == 'torch':
        return torch.squeeze(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def vectorize(x, no_warnings=False):
    """
    Return the vectorized (1D) version of the array.

    This function can be a performance bottleneck.

    If config.WARN_FOR_EXPENSIVE_OPERATIONS is True, this will print a warning
    every time this function is called. pass no_warnings=True to over-ride.

    Parameters
    ----------
    x: array-like
        The array to vectorize.

    Returns
    -------
    scalar:
        The vectorized array. Output is stacked along first dimension.

    """
    backend = getBackend(x)

    # squeeze dimensions that have length 1
    if backend == 'numpy':
        x = np.squeeze(x)

    # If array is already vector, just return
    if len(shape(x)) == 1:
        return x

    if config.WARN_FOR_EXPENSIVE_OPERATIONS and not no_warnings:
        print(
            "WARNING: calling reshape can be an expensive operation, it is normally advised to avoid this."
        )

    if backend == 'numpy':
        return x.ravel()
    elif backend == 'arrayfire':
        return arrayfire.data.flat(transpose(x, hermitian=False))
    elif backend == 'torch':
        return x.view(size(x))
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def vec(x, no_warnings=False):
    """Short-hand for llops.vectorize."""
    return vectorize(x, no_warnings)


def size(x):
    """
    Return the number of elements of the input array.

    Parameters
    ----------
    x: array-like
        The array to determine the number of element of.

    Returns
    -------
    scalar:
        The number of elements in the input.

    """
    return prod(shape(x))


def ndim(x):
    """
    Return the number of dimensions of the input array.

    Note that arrays with the arrayfire backend have fixed array count. For these
    arrays, this function will return the maximum dimension with non-unity size.

    Parameters
    ----------
    x: array-like
        The array to check the dimensions of.

    Returns
    -------
    scalar:
        The number of dimensions in the input.

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return x.ndim
    elif backend == 'arrayfire':
        return len(x.dims())
    elif backend == 'list':
        _ndim = 1
        _x = x
        while type(_x[0]) is list:
            _x = _x[0]
            _ndim += 1

        return _ndim
    elif backend == 'scalar':
        return 1
    elif backend == 'torch':
        return x.ndimension()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def expandDims(x, new_dim_count):
    """
    Expands the dimensions of an array to match new_dim_count.

    Note that arrays with the arrayfire backend have fixed array count. For these
    arrays, this function does nothing.

    Parameters
    ----------
    x: array-like
        The array to expand dimensions of.
    new_dim_count: tuple or list
        The desired number of dimensions in the array.

    Returns
    -------
    array-like:
        The expanded array with dimension new_dim_count.

    """
    backend = getBackend(x)
    assert new_dim_count <= 4
    if backend == 'numpy':
        while np.ndim(x) < new_dim_count:
            x = x[:, np.newaxis]
        return x
    elif backend == 'arrayfire':
        return x  # arrayfire arrays are always 4D
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def alloc(shape, dtype, backend):
    """
    Allocates an empty array in memory with the given shape, dtype, and backend.

    Parameters
    ----------
    shape: tuple or list
        The desired shape of the array.
    dtype: str
        Optinal. The desired datatype. If None, uses llops.config.default_dtype.
    backend: str
        Optinal. The desired backend. If None, uses llops.config.default_backend.

    Returns
    -------
    array-like:
        The allocated array.

    """
    _dtype = getNativeDatatype(dtype, backend)
    if backend == 'numpy':
        return np.empty(shape, dtype=_dtype, order='F')
    elif backend == 'arrayfire':
        if type(shape) not in [tuple, list, np.ndarray]:
            shape = [shape]
        return arrayfire.Array(dims=shape, dtype=_dtype)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def dealloc(x):
    """
    Frees memory associated with the input.

    Parameters
    ----------
    x: array-like
        The array to erase from memory.

    Returns
    -------

    """

    backend = getBackend(x)
    if backend == 'numpy':
        x = None
    elif backend == 'arrayfire':
        x = None
        arrayfire.device_gc()
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def free(x):
    """Helper method for llops.dealloc."""
    dealloc(x)


def dcopy(x):
    """
    Returns a deep copy of an array.

    Parameters
    ----------
    x: array-like
        The array to create a deep copy of.

    Returns
    -------
    array-like:
        The deep copy of the input.

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return x.copy()
    elif backend == 'arrayfire':
        xc = x.copy()
        arrayfire.device_gc()
        return xc
    elif backend == 'list':
        x_new = []
        for _x in x:
            x_new.append(dcopy(_x))
        return x_new
    elif backend == 'tuple':
        x_new = []
        for _x in x:
            x_new.append(dcopy(_x))
        return tuple(x_new)
    elif backend == 'scalar':
        return x
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def copy(x):
    return dcopy(x)


def std(x):
    """
    Returns the standard deviation of all elements of an array.

    Parameters
    ----------
    x: array-like or scalar
        The array to take the standard deviation of

    Returns
    -------
    array-like:
        The standard deviation of the input.

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.std(x)
    elif backend == 'arrayfire':
        return arrayfire.stdev(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def outer(a, b):
    """
    Returns the outer product of two vectors.

    Parameters
    ----------
    a: array-like
        The first (column) array
    b: array-like
        The second (row) array

    Returns
    -------
    array-like:
        The outer product of a and b.

    """
    # Get backend
    backend = getBackend(a)

    if backend == 'numpy':
        return a[:, np.newaxis] * b[np.newaxis, :]
    elif backend == 'arrayfire':
        mul = lambda x, y: x * y
        return arrayfire.broadcast(mul, a, b.T)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)

def median(x):
    """
    Returns the median of an array.

    Parameters
    ----------
    x: array-like or scalar
        The array to take the global median of

    Returns
    -------
    array-like:
        The median of the input.

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return np.median(x)
    elif backend == 'arrayfire':
        if 'complex' in getDatatype(x):
            m = arrayfire.median(real(x)) + 1j * arrayfire.median(imag(x))
        else:
            m = arrayfire.median(x)
        arrayfire.device_gc()
        return m
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def mean(x, axis=None):
    """
    Returns the mean of an array.

    Parameters
    ----------
    x: array-like or scalar
        The array to take the global mean of
    axis: int
        Optional. The axis over which to take the mean.

    Returns
    -------
    array-like:
        The mean of the input.

    """
    if axis is None:
        return scalar(sum(x) / size(x))
    else:
        return sum(x, axis=axis) / shape(x)[axis]


def fill(x, val):
    """
    Fill all elements off array with the same value
    """
    backend = getBackend(x)
    if backend == 'numpy':
        x.fill(scalar(val))
    elif backend == 'arrayfire':
        x[:] = scalar(val)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def pad(x, M, crop_start=(0, 0), pad_value=0, y=None, center=False):
    """
    Pad operation with backend selector
    """
    # Get shape, backend, and datatype
    N = shape(x)

    if type(M) not in (list, tuple, np.ndarray):
        M = [M]

    if len(N) is not len(M):
        N = list(N) + [1] * (len(M) - len(N))

    backend = getBackend(x)
    dtype = getDatatype(x)

    # Check sizes
    # assert builtins.all([crop_start[i] >= 0 for i in range(len(M))]), "crop_start must be > 0!"
    # assert builtins.all(shape(x)[i] + crop_start[i] <= M[i] for i in range(len(M))), "crop_start would force pad outside of dimensions!"

    # Determine if the cropped region is outside of the FOV
    # Define a mask for assigning the output (used if crop extends outside object size)
    input_roi = [slice(0, n) for n in N]

    # If center flag is specified, over-ride the crop_start value
    if center:
        crop_start = [int(math.ceil(m / 2) - math.ceil(n / 2)) for (m, n) in zip(M, N)]

    # If crop region resides outside bounds, shrink the output ROI to reflect this
    for i in range(len(M)):
        if crop_start[i] < 0:
            input_roi[i] = slice(int(-crop_start[i]), int(N[i]))
        elif crop_start[i] + N[i] > M[i]:
            input_roi[i] = slice(0,  int(M[i] - crop_start[i]))
    input_roi = tuple(input_roi)

    # Take note of whether we need to return y or not
    return_y = y is None

    # Allocate or check output variable
    if y is None:
        y = ones(M, dtype=dtype, backend=backend)
    else:
        assert getBackend(y) == backend, "Wrong backend for output (%s, needs to be %s)" % (getBackend(y), backend)
        assert getDatatype(y) == dtype, "Wrong dtype for output (%s, needs to be %s)" % (getDatatype(y), dtype)
        y = reshape(y, M)

    # Determine how to pad the value
    if isinstance(pad_value, str):
        if pad_value == 'mean':
            fill(y, mean(x))

        elif pad_value == 'median':
            fill(y, median(x))

        elif pad_value == 'maximum':
            fill(y, max(x))

        elif pad_value == 'minimum':
            fill(y, min(x))

        elif pad_value in ['repeat', 'wrap']:
            shifts = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            shifts.remove((0, 0))
            for shift in shifts:
                shift_amount = [shift[i] * N[i] for i in range(len(N))]

                # Determine where to place copies of object
                padded_start = [
                    builtins.max(shift_amount[i] + crop_start[i], 0)
                    for i in range(len(N))
                ]
                padded_end = [
                    builtins.min(shift_amount[i] + crop_start[i] + N[i], M[i])
                    for i in range(len(N))
                ]

                slc_padded = []
                for i in range(len(N)):
                    slc_padded += [
                        slice(padded_start[i], padded_end[i]),
                    ]

                # Determine where to place copies of object
                slc_input = []
                for i in range(len(N)):
                    slc_input += [
                        slice(
                            padded_start[i] + -1 * shift[i] * N[i] -
                            crop_start[i], -crop_start[i] + padded_end[i] +
                            -1 * shift[i] * N[i]),
                    ]

                # Assign value in array
                if builtins.all([pstart != pend for (pstart, pend) in zip(padded_start, padded_end)]):
                    y[tuple(slc_padded)] = x[tuple(slc_input)]

        elif pad_value == 'reflect':
            shifts = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
            shifts.remove((0, 0))
            for shift_0 in shifts:
                shift = [-1 * s for s in shift_0]
                shift_amount = [shift[i] * N[i] for i in range(len(N))]

                # Determine where to place copies of object
                padded_start = [
                    builtins.max(shift_amount[i] + crop_start[i], 0)
                    for i in range(len(N))
                ]
                padded_end = [
                    builtins.min(shift_amount[i] + crop_start[i] + N[i], M[i])
                    for i in range(len(N))
                ]

                slc_padded = []
                for i in range(len(N)):
                    slc_padded += [
                        slice(padded_start[i], padded_end[i]),
                    ]

                # Determine where to place copies of object
                input_start = [
                    padded_start[i] -
                    shift[i] * (padded_end[i] - padded_start[i]) - crop_start[i]
                    for i in range(len(N))
                ]
                input_end = [
                    padded_end[i] - shift[i] * (padded_end[i] - padded_start[i])
                    - crop_start[i] for i in range(len(N))
                ]

                slc_input = []
                for i in range(len(N)):
                    slc_input += [slice(input_start[i], input_end[i]),]

                # Flip positions if necessary
                axes_to_flip = []
                for axis, sh in enumerate(shift):
                    if np.abs(sh) > 0:
                        axes_to_flip.append(axis)

                # Assign value in array
                if builtins.all([pstart != pend for (pstart, pend) in zip(padded_start, padded_end)]):
                    y[tuple(slc_padded)] = flip(x[tuple(slc_input)], axis=axes_to_flip)

        elif pad_value == 'rand':
            # Pad with random values from a uniform distribution
            # Keep same mean as image
            values = rand(shape(y), dtype=dtype, backend=backend)

            # Get object statistics
            x_mean, x_range = mean(x), max(x) - min(x)

            # Ensure  padded values have same statistics
            values *= x_range / 2
            values += x_mean

            # Assign values
            y[:] = values

        elif pad_value == 'randn':
            # Pad with random values from a uniform distribution
            # Keep same mean as image
            values = randn(shape(y), dtype=dtype, backend=backend)

            # Get object statistics
            x_mean, x_range = mean(x), max(x) - min(x)

            # Ensure  padded values have same statistics
            values *= x_range / 2
            values += x_mean

            # Assign values
            y[:] = values

        elif pad_value == 'edge':

            # Determine regions of interest
            shifts = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]

            # Remove center
            shifts.remove((0, 0))

            # Loop over regions
            for shift_0 in shifts:
                shift = [-1 * s for s in shift_0]
                shift_amount = [shift[i] * N[i] for i in range(len(N))]

                # Determine where to place copies of object
                padded_start = [builtins.max(shift_amount[i] + crop_start[i], 0) for i in range(len(N))]
                padded_end = [builtins.min(shift_amount[i] + crop_start[i] + N[i], M[i]) for i in range(len(N))]

                slc_padded = []
                for i in range(len(N)):
                    slc_padded += [slice(padded_start[i], padded_end[i]), ]

                # Determine Edge values and repeat to match the padded size
                input_start = []
                input_end = []
                slc_input = []
                for axis_index in range(len(N)):

                    # Calculate start and end coordinates
                    input_start = (0 if shift[axis_index] <= 0 else N[axis_index] - 1)
                    input_end = (N[axis_index] if shift[axis_index] >= 0 else 1)

                    # Generate slices
                    slc_input += [slice(input_start, input_end), ]

                # Calculate padded edge vector
                if all([pstart != pend for (pstart, pend) in zip(padded_start, padded_end)]):

                    # Get padded edge
                    _padded_edge = x[tuple(slc_input)]

                    # Calculate shape
                    _shape = shape(_padded_edge, ndim=len(N))

                    # Extend edge vector to edge of padded region
                    for axis_index in range(len(N)):
                        if _shape[axis_index] == 1:
                            tile_count = [1 if n is not axis_index else abs(padded_start[axis_index] - padded_end[axis_index]) for n in range(len(N))]
                            _padded_edge = tile(_padded_edge, tile_count)

                    # Assign value in array if the size of the padded area is not zero
                    if all([s > 0 for s in shape(_padded_edge)]):
                        y[tuple(slc_padded)] = _padded_edge

        else:
            raise ValueError('Invalid pad_value (%s)' % pad_value)

    elif getBackend(pad_value) == 'scalar':
        fill(y, pad_value)
    elif isarray(pad_value):
        # pad_value is array
        y[:] = pad_value
    else:
        raise ValueError('Invalid pad value %s' % str(pad_value))

    # Determine ROI of y to assign x to
    output_roi = []
    for i in range(len(N)):
        output_roi += [slice(int(builtins.max(crop_start[i], 0)), int(builtins.min(crop_start[i] + N[i], M[i]))), ]
    output_roi = tuple(output_roi)

    # Assign output
    y[output_roi] = x[input_roi]

    if return_y:
        return y


def crop(x, M, crop_start=(0, 0), y=None, out_of_bounds_placeholder=None, center=False):
    """Crop a measurement."""
    # Get backend, dtype, and shape
    backend = getBackend(x)
    dtype = getDatatype(x)
    N = x.shape

    # Define a mask for assigning the output (used if crop extends outside object size)
    output_roi = [slice(0, m) for m in M]

    # If center flag is specified, over-ride the crop_start value
    if center:
        crop_start = [int(math.floor(n / 2 - m / 2)) for (m, n) in zip(M, shape(x))]

    # If crop region resides outside bounds, shrink the output ROI to reflect this
    for i in range(len(N)):
        if crop_start[i] < 0:
            output_roi[i] = slice(int(-crop_start[i]), int(M[i]))
        elif crop_start[i] + M[i] > N[i]:
            output_roi[i] = slice(0, int(N[i] - crop_start[i]))
    output_roi = tuple(output_roi)

    # Determine crop region
    input_roi = []
    for i in range(len(N)):
        input_roi += [slice(int(builtins.max(crop_start[i], 0)), int(builtins.min(crop_start[i] + M[i], N[i]))), ]
    input_roi = tuple(input_roi)

    # Check whether we nee to return
    return_y = y is None

    # Allocate y if it's not provided
    if y is None:
        if out_of_bounds_placeholder is not None:
            y = ones(M, dtype=dtype, backend=backend) * out_of_bounds_placeholder
        else:
            y = ones(M, dtype=dtype, backend=backend) * np.nan
    else:
        if out_of_bounds_placeholder is not None:
            y[:] = out_of_bounds_placeholder
        else:
            y[:] = np.nan

    # Perform assignment
    y[output_roi] = x[input_roi]

    if return_y:
        return y


def grid(shape, scale=1, offset=None, center=True, dtype=None, backend=None):
    """
    MATLAB-style meshgrid operator. Takes a shape and scale and produces a list of coordinate grids.

    Parameters
    ----------
    shape: list, tuple
        The desired shape of the grid
    scale: list, tuple, int
        Optinal. The scale of the grid. If provided as an integer, provides the
        same scale across all axes. If provided as a list or tuple, must be of
        the same length as shape
    offset: list, tuple, int
        Optinal. Offset of the grid. If provided as an integer, provides the
        same offset across all axes. If provided as a list or tuple, must be of
        the same length as shape.
    dtype: string
        Optional. The desired datatype, if different from the default.
    backend: string
        Optional. The desired backend, if different from the default.

    Returns
    -------
    list:
        List of arrays with provided backend and dtype corresponding to
        coordinate systems along each dimension.

    """
    # Parse scale operation
    if type(scale) not in [list, tuple, np.array, np.ndarray]:
        scale = [scale] * len(shape)

    # Parse offset operation
    if offset is None:
        offset = [0] * len(shape)

    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    if backend == 'numpy':
        dtype_np = getNativeDatatype(dtype, 'numpy')
        if len(shape) == 1:
            grid = (
                (np.arange(size, dtype=dtype) - shape[0] // 2) * scale[0]
                - offset[0]).astype(dtype_np)

            if not center:
                grid -= min(grid)

            return grid

        elif len(shape) == 2:
            # print((np.arange(shape[0], dtype=dtype) - shape[0] // 2))
            dtype_np = getNativeDatatype(dtype, 'numpy')
            lin_y = (np.arange(shape[0], dtype=dtype_np) - shape[0] // 2) * scale[0] - offset[0]
            lin_x = (np.arange(shape[1], dtype=dtype_np) - shape[1] // 2) * scale[1] - offset[1]
            grid_y = (lin_y[:, np.newaxis] * np.ones_like(lin_x)[np.newaxis, :]).astype(dtype_np)
            grid_x = (lin_x[np.newaxis, :] * np.ones_like(lin_y)[:, np.newaxis]).astype(dtype_np)

            if not center:
                grid_y -= min(grid_y)
                grid_x -= min(grid_x)

            return ((grid_y, grid_x))

        elif len(shape) == 3:
            grid_z = ((np.arange(shape[0], dtype=dtype) - shape[0] // 2) *
                      scale[0] - offset[0]).astype(dtype_np)
            grid_y = ((np.arange(shape[1], dtype=dtype) - shape[1] // 2) *
                      scale[1] - offset[1]).astype(dtype_np)
            grid_x = ((np.arange(shape[2], dtype=dtype) - shape[2] // 2) *
                      scale[2] - offset[2]).astype(dtype_np)

            if not center:
                grid_y -= min(grid_y)
                grid_x -= min(grid_x)
                grid_z -= min(grid_z)

            return ((grid_z, grid_y, grid_x))

    elif backend == 'arrayfire':
        if len(shape) == 1:
            grid = arrayfire.range(shape[0]) - offset[0]

            if not center:
                grid -= min(grid)

            return grid

        elif len(shape) == 2:
            grid_y = (arrayfire.range(shape[0], shape[1], dim=0) -
                      shape[0] // 2) * scale[0] - offset[0]
            grid_x = (arrayfire.range(shape[0], shape[1], dim=1) -
                      shape[1] // 2) * scale[0] - offset[1]

            if not center:
                grid_y -= min(grid_y)
                grid_x -= min(grid_x)

            return ((grid_y, grid_x))

        elif len(shape) == 3:
            grid_z = (arrayfire.range(shape[0], shape[1], shape[2], dim=0) -
                      shape[0] // 2) * scale[0] - offset[0]
            grid_y = (arrayfire.range(shape[0], shape[1], shape[2], dim=1) -
                      shape[1] // 2) * scale[0] - offset[1]
            grid_x = (arrayfire.range(shape[0], shape[1], shape[2], dim=2) -
                      shape[2] // 2) * scale[0] - offset[2]

            if not center:
                grid_y -= min(grid_y)
                grid_x -= min(grid_x)
                grid_z -= min(grid_z)

            return ((grid_z, grid_y, grid_x))
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def assert_equality(x1, x2, metric='max', threshold=None):
    """
    Check the equality of two arrays. The arrays need not be of the same datatype.

    Parameters
    ----------
    x1: array-like
        The first array to compare
    x2: array-like
        The second array to compare
    metric: str
        Optional. Defines the metric used to determine equality. Can be 'max' or 'ssd'
    threshold: float
        Optional. The threshold to hold the SSD of x1 and x2 below.
        If not provided, uses llops.precision(x1)

    Returns
    -------
    None

    """
    # Ensure both arrays are the same backend
    x2 = asbackend(x2, getBackend(x1))

    # Get threshold
    threshold = precision(x1) * size(x1) if threshold is None else threshold

    # Vectorize both arrays to remove extra dimensions
    x1, x2 = vec(x1), vec(x2)

    if metric is 'ssd':
        # Determine SSD between arrays
        ssd = sum(abs(x2 - x1) ** 2)

        # Check equality
        assert ssd < threshold, "SSD of inputs (%g) was greater then threshold (%g)" % (ssd, threshold)
    elif metric is 'max':
        # Check that max difference is less than precision
        max_difference = max(abs(x1 - x2))

        # Check equality
        assert max_difference < threshold, "Max difference of inputs (%g) was greater then threshold (%g)" % (max_difference, threshold)


def ramp(shape, axis=0, min_value=0.0, max_value=1.0, reverse=False, dtype=None, backend=None):
    """
    Return a linear ramp along a given axis with a given shape.

    Parameters
    ----------
    shape: list or tuple
        The desired shape of the array.
    axis: int
        Optional. The axis over which to create the ramp.
    min_value: float
        Optional. The mininum value of the ramp.
    max_value: float
        Optional. The maximum value of the ramp.
    reverse: bool
        Optional. If true, the ramp is decreasing instead of increasing along axis.
    dtype: string
        Optional. The desired datatype, if different from the default.
    backend: string
        Optional. The desired backend, if different from the default.

    Returns
    -------
    array-like:
        A linear ramp along the given axis with the given shape.

    """
    # Get default dtype and backend if none proided
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    # Return unit array if max and min values are the same
    if max_value == min_value:
        return ones(shape, dtype=dtype, backend=backend)

    # Generate ramp
    if reverse:
        delta = builtins.round((min_value - max_value) / shape[axis], 8)
        ramp_1d = slice(max_value, min_value + delta, delta)
    else:
        delta = builtins.round((max_value - min_value) / shape[axis], 8)
        ramp_1d = slice(min_value, max_value - delta, delta)

    # Generate slice coordinates
    coordinates = [slice(0, sz) for sz in shape]
    coordinates[axis] = ramp_1d

    # Create ramp
    ramp = resize(np.mgrid[coordinates][axis], shape)

    # Return
    return asarray(ramp, dtype, backend)


def round(x):
    """
    Rounds all elements of an array to the nearest integer.
    By convention, 0.5 is rounded to 1.0.
    This function keeps the same datatype.

    Parameters
    ----------
    x: array-like
        Array to round.
    Returns
    -------
    array-like:
        A linear ramp along the given axis with the given shape.

    """

    backend = getBackend(x)

    if backend == 'numpy':
        return cast_like(np.round(x), x)
    elif backend == 'arrayfire':
        if isComplex(x):
            return cast_like(arrayfire.arith.round(real(x)) + 1j * arrayfire.arith.round(imag(x)), x)
        else:
            return cast_like(arrayfire.arith.round(x), x)
    elif backend == 'scalar':
        return round(x)
    elif backend == 'list':
        return [round(item) for item in x]
    elif backend == 'tuple':
        return tuple([round(item) for item in x])
    else:
        raise ValueError('Backend %s is not supported!' % backend)

@numpy_function
@real_valued_function
def resize(x, new_shape):
    """Resize an array, allowing the number of dimensions to change."""
    return skimage.transform.resize(x, new_shape, anti_aliasing=True, mode='edge', preserve_range=True)

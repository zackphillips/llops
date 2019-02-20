"""
Copyright 2017 Zachary Phillips, Waller Lab, University of California, Berkeley.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import time
import functools
import collections

from .base import randu, shape, getDatatype, matmul, transpose, vec, sum, abs, isComplex, real, asbackend, cast, getBackend, ndim, real, imag
from .roi import Roi

__all__ = ['gradientCheck', 'sanatizeDictionary']


def sanatizeiterable(iterable):
    if isinstance(iterable, collections.Iterable):
        if type(iterable) is tuple:
            # Convert to list so it's writable
            iterable = list(iterable)
        elif type(iterable) in (np.array, np.ndarray):
            iterable = iterable.tolist()

        for index, item in enumerate(iterable):
            if type(item) in (list, tuple):
                iterable[index] = sanatizeiterable(item)
            elif type(item) is dict:
                sanatizeDictionary(item)
            elif np.isscalar(item):
                iterable[index] = sanatizeScalar(item)
            else:
                print('Skipping %s' % str(item))

    return iterable


def sanatizeScalar(scalar):
    if np.isscalar(scalar):
        if np.iscomplex(np.asarray(scalar)) and np.sum(np.imag(np.asarray(scalar))) != 0:
            return {'real': np.asscalar(np.real(np.asarray(scalar))),
                    'imag': np.asscalar(np.imag(np.asarray(scalar)))}
        else:
            return np.asscalar(np.real(np.asarray(scalar)))


def sanatizeDictionary(dict_to_sanatize):
    """Convert all dict elements from a numpy arrays to lists (normally used for seralizing for a json file)."""
    if type(dict_to_sanatize) is dict:
        for key_name in dict_to_sanatize:
            if '__' not in key_name:
                if type(dict_to_sanatize[key_name]) in (list, tuple, np.array, np.ndarray):
                    dict_to_sanatize[key_name] = sanatizeiterable(dict_to_sanatize[key_name])
                elif type(dict_to_sanatize[key_name]) is dict:
                    sanatizeDictionary(dict_to_sanatize[key_name])
                elif np.isscalar(dict_to_sanatize[key_name]):
                    dict_to_sanatize[key_name] = sanatizeScalar(dict_to_sanatize[key_name])


def gradientCheck(forward, gradient, size, dtype, backend, eps=1e-4, step=1e-3, x=None, direction=None):
    """
    Check a gradient function using a numerical check.

    Parameters
    ----------
    forward : function
        The forward function of the operator
    gradient : function
        The gradient function of the operator
    size : tuple
        The size of the input to the forward and gradient methods
    dtype : string
        The datatype of the input, as expressed as a string
    backend : string
        The backend of the input
    eps : scalar
        The precision to which the gradient comparison will be held
    x : array-like
        Optional. Array to use for testing the gradient
    direction : array-like
        Optional. Direction to use for testing gradinet

    Returns
    -------
    bool
        True if successful, False otherwise.

    """
    # Make sure step is of correct datatype
    step = np.float64(step)

    # Generate x
    if x is None:
        x = randu(size, dtype, backend)
        x[x == 0] = 0.1
    else:
        assert size(x) == shape, "Size of provided input %s does not equal shape of operator %s" % (size(x), shape)
        assert getDatatype(x) == dtype

    # Generate direction
    if direction is None:
        # Pick random directions until we get some change in the objective function
        direction = randu(size, dtype, backend)
    else:
        assert size(direction) == shape, "Size of provided direction %s does not equal shape of operator %s" % (size(direction), shape)
        assert getDatatype(direction) == dtype

    # Calculate an approximate gradient
    approx_gradient = (forward(x + step * direction) - forward(x - step * direction)) / (2 * step)

    # Calculate Gradient to test (override warnings from vec() call)
    g = matmul(transpose(vec(gradient(x), no_warnings=True), hermitian=True), vec(direction, no_warnings=True))  # dot product

    if not isComplex(g):
        error = sum(abs(g - approx_gradient)) / sum(abs(g + approx_gradient))
    else:
        error = sum(abs(real(g) - approx_gradient)) / sum(abs(g + approx_gradient))

    # Check error
    assert error < eps, "Gradient was off by %.4e (threshold is %.4e)" % (error, eps)

    # Return Error
    return error

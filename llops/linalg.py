"""
Copyright 2019 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np

# Allow division by zero to return NaN without a warning
np.seterr(divide='ignore', invalid='ignore')

# Try to import arrayfire - continue if import fails
try:
    import arrayfire
except ImportError:
    pass


def norm(x):
    """
    A generic norm operator with backend selector
    """
    from .base import getBackend
    backend = getBackend(x)
    if backend == 'numpy':
        return np.linalg.norm(x)
    elif backend == 'arrayfire':
        return arrayfire.lapack.norm(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def svd(x, **kwargs):
    """
    A generic svd operator with backend selector
    """
    from .base import getBackend
    backend = getBackend(x)
    if backend == 'numpy':
        return np.linalg.svd(x, **kwargs)
    elif backend == 'arrayfire':
        return arrayfire.lapack.svd(x,  **kwargs)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def lstsq(A, b):
    """Solve linear least squares problems."""
    from .base import getBackend

    # Get backend
    backend = getBackend(b)
    assert getBackend(A) == getBackend(b)

    # Perform inversion
    if backend == 'numpy':
        return np.linalg.lstsq(A, b, rcond=None)[0]
    elif backend == 'arrayfire':
        return arrayfire.lapack.solve(A, b)

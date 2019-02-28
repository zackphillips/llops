import numpy as np
from .operators import Operator, Crop, FourierTransform, Diagonalize, Sum, PhaseRamp, Identity, Shift
from .stack import Hstack, Vstack

from llops import config, abs, min, max, real, zeros_like, isnan, squeeze, sum, boundingBox, crop, where, astype, dcopy, tile, size, roll, Ft, iFt, reshape, asarray, argmax, zeros, abs, conj, ndim, getBackend, getDatatype, vec, fftshift, getNativeDatatype, changeBackend, where, scalar, asbackend
import types
import math
import llops as yp


class RealFilter(Operator):

    def __init__(self, shape, dtype=None, backend=None, label='REAL'):
        # Configure backend and datatype
        self.backend = backend if backend is not None else yp.config.default_backend
        self.dtype = dtype if dtype is not None else yp.config.default_dtype

        # Shape
        M, N = shape, shape

        super(self.__class__, self).__init__((M, N), self.dtype, self.backend, cost=1,
                                             condition_number=1.0, label=label,
                                             forward=self._forward, adjoint=self._adjoint)
                                             
    def _forward(self, x, y):
        y[:] = yp.astype(yp.real(x), self.dtype)

    def _adjoint(self, x, y):
        y[:] = yp.astype(yp.real(x), self.dtype)


class ImagFilter(Operator):

    def __init__(self, shape, dtype=None, backend=None, label='IMAG'):
        # Configure backend and datatype
        backend = backend if backend is not None else yp.config.default_backend
        dtype = dtype if dtype is not None else yp.config.default_dtype

        super(self.__class__, self).__init__(shape, dtype, backend, cost=1,
                                             condition_number=1.0, label=label,
                                             forward=self._forward, adjoint=self._adjoint)

    def _forward(self, x, y):
        y[:] = yp.imag(x)

    def _adjoint(self, x, y):
        y[:] = yp.imag(x)

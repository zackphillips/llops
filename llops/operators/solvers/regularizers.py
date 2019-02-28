import ndoperators as ops
from ndoperators import Operator, L2Norm, L1Norm, FourierTransform, OperatorSum
import numpy as np
from llops import fill, shape, abs, config, prod, any, all, real, sum, conj, vectorize, zeros, sqrt, roll, pad, crop
from llops.filter import softThreshold
import math
from . import denoisers
import builtins


def parse(reg_types, object_shape):
    """Parse regularization objects."""

    if reg_types is None or len(reg_types) == 0:
        return None

    # Parse regularizer
    regularizer_list = []
    if 'l2' in reg_types.keys():
        regularizer_list.append(reg_types['l2'] * L2Norm(object_shape))
    if 'l1' in reg_types.keys():
        regularizer_list.append(reg_types['l1'] * L1Norm(object_shape))
    if 'l1F' in reg_types.keys():
        regularizer_list.append(reg_types['l1F'] * L1Norm(object_shape)
                                                 * FourierTransform(object_shape, pad=True))
    if 'wavelet' in reg_types.keys():
        regularizer_list.append(reg_types['wavelet'] * WaveletSparsity(object_shape, wavelet_type='db4',
                                                                       extention_mode='symmetric',
                                                                       level=None,
                                                                       use_cycle_spinning=True,
                                                                       axes=None))

    if 'tv' in reg_types.keys():
        regularizer_list.append(reg_types['tv'] * TV(object_shape))

    if 'bilateral' in reg_types.keys():
        regularizer_list.append(reg_types['bilateral'] * RegDenoiser(object_shape, denoise_type='bilateral'))

    if 'median' in reg_types.keys():
        regularizer_list.append(reg_types['median'] * RegDenoiser(object_shape, denoise_type='median'))

    if 'tv_wavelet' in reg_types.keys():
        regularizer_list.append(reg_types['tv_wavelet'] * RegDenoiser(object_shape, denoise_type='tv_bregman', weight=1.0))

    # Defining Cost Function
    reg = regularizer_list[0]
    for _reg in regularizer_list[1:]:
        reg += _reg
    return reg


def L2(size, dtype=None, backend=None):
    ''' A L2 regularizer '''
    # Simply return the L2 Norm squared operator
    return ops.L2Norm(size, dtype, backend)


def L1(size, dtype=None, backend=None):
    ''' A L1 regularizer '''
    # Simply return the L2 Norm squared operator
    return ops.L1Norm(size, dtype, backend)


def WaveletSparsity(size, dtype=None, backend=None, wavelet_type='haar', extention_mode='symmetric', level=None, use_cycle_spinning=True, axes=None):

    # Make wavelet transform operator
    if wavelet_type == 'haar':
        W = Haar(size, dtype, backend)
    else:
        W = ops.WaveletTransform(size, wavelet_type, extention_mode, level, use_cycle_spinning, axes, dtype, backend)

    # Make L1 norm operator
    L1 = ops.L1Norm(W.M, dtype, backend)

    # Store sparse operator for easy access
    op = L1 * W
    op.sparse_op = W

    return op


def FourierSparsity(size, dtype=None, backend=None):

    # Make Fourier Transform Operator
    F = ops.FourierTransform(size, dtype, backend)

    # Make L1 norm operator
    L1 = ops.L1Norm(F.M, dtype, backend)

    # Store sparse operator for easy access
    op = L1 * F
    op.sparse_op = F

    return op


class Positivity(Operator):

    def __init__(self, shape, dtype=None, backend=None, label='1^+'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        super(self.__class__, self).__init__(((1,1), shape), dtype, backend,
                                             label=label, cost=prod(shape),
                                             forward=self._forward,
                                             proximal=self._proximal,
                                             smooth=False,
                                             repr_latex=self._latex)
    def _forward(self, x, y):
        if any(real(x) < 0):
            y[:] = np.inf
        else:
            y[:] = 0

    def _proximal(self, x, alpha):
        x[real(x) < 0] = 0.0
        return x

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return '1^+\\{' + latex_input + '\\}'
        else:
            return '1^+\\{ \\cdot \\}'


class RegDenoiser(Operator):
    # TODO kwargs?
    def __init__(self, shape, denoise_type='tv',
                 dtype=None, backend=None, **denoise_args):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        # select denoiser function
        if denoise_type == 'tv':
            denoise_fn = denoisers.tv
        elif denoise_type == 'tv_bregman':
            denoise_fn = denoisers.tv_bregman
        elif denoise_type == 'bilateral':
            denoise_fn = denoisers.bilateral
        elif denoise_type == 'wavelet':
            denoise_fn = denoisers.wavelet
        elif denoise_type == 'weiner':
            denoise_fn = denoisers.weiner
        elif denoise_type == 'nl_means':
            denoise_fn = denoisers.nl_means
        elif denoise_type == 'median':
            denoise_fn = denoisers.median_denoise
        else:
            raise NotImplementedError

        self.denoiser = lambda image: denoise_fn(image, **denoise_args)

        # Instantiate Metaclass
        super(self.__class__, self).__init__(((1, 1), shape), dtype, backend,
                                             forward=self._forward,
                                             gradient=self._gradient,
                                             inverse=self._inverse,
                                             smooth=True,
                                             repr_latex=self._latex)

    def _forward(self, x, y):
        y[:] = 0.5 * sum(conj(x) * (x-self.denoiser(x)))

    def _gradient(self, x=None, inside_operator=None):
        def grad_forward(x, y):
            y[:] = x-self.denoiser(x)
        return Operator((self.N, self.N),
                        self.dtype,
                        self.backend,
                        forward=grad_forward,
                        adjoint=None,
                        smooth=True,
                        repr_str='∇(' + self.repr_str + ')')

    def _inverse(self, x, y):
        y[:] = self.denoiser(x)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return '0.5'+latex_input+'^H( '+latex_input+' -d('+latex_input+'))'
        else:
            return '0.5\\cdot^H( \\cdot -d(\\cdot))'


class TV(Operator):
    """
    Implements TV using theHaar transform and inverse transform.
    D: Number of dimensions for Haar transform. Should be 2 or 3.
    """

    def __init__(self, shape, label='TV', dtype=None, backend=None):

        # Configure backend and datatype
        self.backend = backend if backend is not None else config.default_backend
        self.dtype = dtype if dtype is not None else config.default_dtype

        self.D = len(shape)
        self.threshold = 1.41 * 2 * self.D

        # Allocate prox array before calculation for efficency
        self._forward_y = zeros(shape, self.dtype, self.backend)
        self._prox_y = zeros(shape, self.dtype, self.backend)
        self._haar_w = zeros(shape, self.dtype, self.backend)
        self._haar_y = zeros(shape, self.dtype, self.backend)

        # Instantiate Metaclass
        super(self.__class__, self).__init__(((1, 1), shape), self.dtype, self.backend,
                                             label=label,
                                             forward=self._forward,
                                             proximal=self._proximal,
                                             repr_latex=self._latex,
                                             smooth=False)

    def _forward(self, x, y):
        fill(self._forward_y, 0)

        # Take finite differences
        for dim in range(len(self.shape)):
            self._forward_y += abs(roll(x, 1, axis=dim) - x)

        # Normalize
        self._forward_y[:] = sqrt(self._forward_y)

        # Return sum
        y[:] = sum(self._forward_y)

    def _proximal(self, x, alpha):
        # alpha is the product of the step size and the scaled amount
        fill(self._prox_y, 0)

        # Only soft threshold along the differencing axis (detail coefficients)
        for axis in range(self.D):
            self._prox_y += self._iht3(self._ht3(x, axis, False, alpha=alpha, soft_thresh=True), axis, False)
            self._prox_y += self._iht3(self._ht3(x, axis, True, alpha=alpha, soft_thresh=True), axis, True)

        # Normalize and return
        return self._prox_y / (2 * self.D)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return 'TV(' + latex_input + ')'
        else:
            return 'TV( \\cdot )'

    def _ht3(self, x, ax, shift, alpha=1, soft_thresh=False):
        """
        Implementation of Haar Transform for use in TV prox.
        x: 3 dimensional array.
        ax: Axis to compute transform on. Must be 0, 1, or 2.
        shift: if True, shifts input before transforming.
        soft_thresh: If True, soft thresholds detail coefficients.
        """
        assert ax in [0, 1], 'Input axis must be <= 2.'

        # Fill input with zeros
        fill(self._haar_w, 0)

        # Scaling fav
        C = 1 / sqrt(2)
        if shift:
            x = roll(x, -1, axis=ax)

        m = self.shape[1][ax] // 2
        if ax == 0:  # y axis
            self._haar_w[:m, :] = C * (x[1::2, :] + x[::2, :])
            self._haar_w[m:, :] = C * (x[1::2, :] - x[::2, :])
            if soft_thresh:  # soft threshold detail coeffs.
                self._haar_w[m:, :] = softThreshold(self._haar_w[m:, :], self.threshold * alpha)
        elif ax == 1:  # x axis
            self._haar_w[:, :m] = C * (x[:, 1::2] + x[:, ::2])
            self._haar_w[:, m:] = C * (x[:, 1::2] - x[:, ::2])
            if soft_thresh:
                self._haar_w[:, m:] = softThreshold(self._haar_w[:, m:], self.threshold * alpha)
        return self._haar_w

    def _iht3(self, w, ax, shift):
        """
        Implementation of InverseHaar Transform for use in TV prox.
        """
        assert ax in [0, 1, 2], 'Input axis must be <= 2.'

        # Fill input with zeros
        fill(self._haar_y, 0)

        # Scaling
        C = 1 / sqrt(2)

        # Generate cutoff shape
        m = self.shape[1][ax] // 2

        # Perform inverse wavelet transform
        if ax == 0:
            self._haar_y[::2, :] = C * (w[:m, :] - w[m:, :])
            self._haar_y[1::2, :] = C * (w[:m, :] + w[m:, :])
        elif ax == 1:
            self._haar_y[:, ::2] = C * (w[:, :m] - w[:, m:])
            self._haar_y[:, 1::2] = C * (w[:, :m] + w[:, m:])

        # Shift if requested
        if shift:
            self._haar_y = roll(self._haar_y, 1, axis=ax)

        return self._haar_y


class Haar(Operator):
    """Haar wavelet transform - impleneted for both numpy and arrayfire backends."""
    def __init__(self, M, label='HAAR', dtype=None, backend=None):
        # Configure backend and datatype
        self.backend = backend if backend is not None else config.default_backend
        self.dtype = dtype if dtype is not None else config.default_dtype

        self.D = len(M)
        self.threshold = 1.41 * 2 * self.D

        # Instantiate Metaclass
        super(self.__class__, self).__init__((M, M), self.dtype, self.backend,
                                             label=label,
                                             forward=self._forward,
                                             adjoint=self._adjoint,
                                             condition_number=1,
                                             smooth=True)

    def _forward(self, x, y):
        """
        Implementation of Haar Transform for use in TV prox.
        x: 3 dimensional array.
        ax: Axis to compute transform on. Must be 0, 1, or 2.
        shift: if True, shifts input before transforming.
        soft_thresh: If True, soft thresholds detail coefficients.
        """

        # Fill input with zeros
        fill(y, 0)

        # Scaling fav
        C = 1 / sqrt(2)

        for ax in range(len(self.shape)):
            m = self.shape[1][ax] // 2

            if ax == 0:  # y axis
                y[:m, :] = C * (x[1::2, :] + x[::2, :])
                y[m:, :] = C * (x[1::2, :] - x[::2, :])
            elif ax == 1:  # x axis
                y[:, :m] = C * (x[:, 1::2] + x[:, ::2])
                y[:, m:] = C * (x[:, 1::2] - x[:, ::2])

        return y

    def _adjoint(self, x, y):
        """
        Implementation of InverseHaar Transform for use in TV prox.
        """

        # Fill input with zeros
        fill(y, 0)

        # Scaling
        C = 1 / sqrt(2)

        for ax in range(len(self.shape)):
            # Generate cutoff shape
            m = self.shape[1][ax] // 2

            # Perform inverse wavelet transform
            if ax == 0:
                y[::2, :] = C * (x[:m, :] - x[m:, :])
                y[1::2, :] = C * (x[:m, :] + x[m:, :])
            elif ax == 1:
                y[:, ::2] = C * (x[:, :m] - x[:, m:])
                y[:, 1::2] = C * (x[:, :m] + x[:, m:])

        return y

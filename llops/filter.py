"""
Copyright 2019 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from .base import angle, sign, exp, ndim, asbackend, real, imag, changeBackend, abs, min, max, sum, astype, dtype, crop, shape, pad, dcopy, getDatatype, getBackend, asarray, real, conj, circshift, zeros_like, argmax, zeros
from .fft import iFt, Ft
from . import config
from .decorators import numpy_function, real_valued_function, integer_function
import scipy.signal as sig
import numpy as np
import skimage
import scipy
from skimage import feature

try:
    import arrayfire
except ImportError:
    pass

__all__ = ['canny', 'sobel', 'roberts', 'prewitt', 'scharr', 'laplace', 'gaussian', 'linearErodeInner', 'softThreshold']


def sobel(x):
    """
    Filter an array using the sobel filter.

    Parameters
    ----------
    x: array-like:
        The array to be filtered

    Returns
    -------
    array-like:
        The sobel-filtered array

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return _complex_filter(_normalize(x), skimage.filters.sobel)
    elif backend == 'arrayfire':
        return _complex_filter(_normalize(x), arrayfire.image.sobel_filter)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def canny(x, sigma=1, low_threshold=0.03, high_threshold=0.1):
    """
    Filter an array using the canny filter.

    Parameters
    ----------
    x: array-like:
        The array to be filtered

    Returns
    -------
    array-like:
        The canny-filtered array

    """

    backend = getBackend(x)
    if backend == 'numpy':
        return _complex_filter(x, skimage.feature.canny, sigma=sigma,
                                                         low_threshold=low_threshold,
                                                         high_threshold=high_threshold,
                                                         use_quantiles=False)
    elif backend == 'arrayfire':
        # x_filtered = gaussian(x, sigma)
        # return _complex_filter(x, arrayfire.image.canny, low_threshold=0.1, high_threshold=0.4)
        return asbackend(_complex_filter(asbackend(x, 'numpy'), skimage.feature.canny, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold), 'arrayfire')
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def gradient(x):
    """
    Filter an array using the gradinet filter.
    Returns a tuple of gradients in the x and y dimension.

    Parameters
    ----------
    x: array-like:
        The array to be filtered

    Returns
    -------
    array-like:
        The canny-filtered array

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return (np.diff(x, axis=0), np.diff(y, axis=0))
    elif backend == 'arrayfire':
        return arrayfire.image.gradient(x)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def roberts(x):
    """
    Filter an array using the roberts filter.

    Parameters
    ----------
    x: array-like:
        The array to be filtered

    Returns
    -------
    array-like:
        The roberts-filtered array

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return _complex_filter(x, skimage.filters.roberts)
    elif backend == 'arrayfire':
        return asbackend(_complex_filter(asbackend(x, 'numpy'), skimage.filters.roberts), 'arrayfire')
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def prewitt(x):
    """
    Filter an array using the prewitt filter.

    Parameters
    ----------
    x: array-like:
        The array to be filtered

    Returns
    -------
    array-like:
        The prewitt-filtered array

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return _complex_filter(x, skimage.filters.prewitt)
    elif backend == 'arrayfire':
        return asbackend(_complex_filter(asbackend(x, 'numpy'), skimage.filters.prewitt), 'arrayfire')
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def scharr(x):
    """
    Filter an array using the scharr filter.

    Parameters
    ----------
    x: array-like:
        The array to be filtered

    Returns
    -------
    array-like:
        The scharr-filtered array

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return _complex_filter(x, skimage.filters.scharr)
    elif backend == 'arrayfire':
        return asbackend(_complex_filter(asbackend(x, 'numpy'), skimage.filters.scharr), 'arrayfire')
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def laplace(x, ksize=10):
    """
    Filter an array using the scharr filter.

    Parameters
    ----------
    x: array-like:
        The array to be filtered

    Returns
    -------
    array-like:
        The scharr-filtered array

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return _complex_filter(x, skimage.filters.laplace, ksize=ksize)
    elif backend == 'arrayfire':
        return asbackend(_complex_filter(asbackend(x, 'numpy'), skimage.filters.laplace, ksize=ksize), 'arrayfire')
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def gaussian(x, sigma=10):
    """
    Filter an array using the gaussian filter.

    Parameters
    ----------
    x: array-like:
        The array to be filtered
    sigma:
        The standard deviation of the kernel

    Returns
    -------
    array-like:
        The gaussian-filtered array

    """
    backend = getBackend(x)
    if backend == 'numpy':
        return _complex_filter(x, skimage.filters.gaussian, sigma=sigma)
    elif backend == 'arrayfire':
        return asbackend(_complex_filter(asbackend(x, 'numpy'), skimage.filters.gaussian, sigma=sigma), 'arrayfire')
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


@integer_function
def median(x, diameter=3):
    """
    Filter an array using the median filter.

    Parameters
    ----------
    x: array-like:
        The array to be filtered

    Returns
    -------
    array-like:
        The median-filtered array
    """

    backend = getBackend(x)

    if backend == 'numpy':
        from skimage.morphology import disk

        # Perform median filter
        return skimage.filters.median(x, selem=disk(diameter))
    elif backend == 'arrayfire':
        return arrayfire.signal.medfilt(x, diameter, diameter)
    else:
        raise NotImplementedError('Backend %s is not implemented!' % backend)


def _normalize(x):
    _range = max(abs(x)) - min(abs(x))
    x = (abs(x) - min(abs(x))) / _range
    return x

def _complex_filter(x, filter_function, dtype='float32', **kwargs):

    # Apply filter
    if 'complex' in getDatatype(x):
        # Process real and imaginary part seperatly
        x_re = filter_function(astype(real(x), dtype), **kwargs)
        x_im = filter_function(astype(imag(x), dtype), **kwargs)
        x = astype(x_re, getDatatype(x)) + 1j * astype(x_im, getDatatype(x))
    else:
        x = filter_function(astype(real(x), dtype), **kwargs)

    # Return
    return x


# def _bilateral_filter(source, filter_diameter, sigma_i, sigma_s):
#     # Source: https://github.com/anlcnydn/bilateral/blob/master/bilateral_filter.py
#     import math
#     import numpy as np
#
#     def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
#
#         hl = diameter/2
#         i_filtered = 0
#         Wp = 0
#         i = 0
#         while i < diameter:
#             j = 0
#             while j < diameter:
#                 neighbour_x = x - (hl - i)
#                 neighbour_y = y - (hl - j)
#                 if neighbour_x >= len(source):
#                     neighbour_x -= len(source)
#                 if neighbour_y >= len(source[0]):
#                     neighbour_y -= len(source[0])
#                 gi = gaussian(source[neighbour_x, neighbour_y] - source[x,y], sigma_i)
#                 gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
#                 w = gi * gs
#                 i_filtered += source[neighbour_x][neighbour_y] * w
#                 Wp += w
#                 j += 1
#             i += 1
#         i_filtered = i_filtered / Wp
#         filtered_image[x][y] = int(round(i_filtered))
#
#     def distance(x, y, i, j):
#         return np.sqrt((x-i)**2 + (y-j)**2)
#
#     def gaussian(x, sigma):
#         return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
#
#     filtered_image = np.zeros(source.shape)
#
#     i = 0
#     while i < len(source):
#         j = 0
#         while j < len(source[0]):
#             apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
#             j += 1
#         i += 1
#     return filtered_image
#

def linearErodeInner(x, blend_size=10):
    """
    Linear ramp erosion of an array.

    This function thresholds an array to values greater than zero, then erodes
    all edges to within the support of the thresholded array. If the array is
    not binary, it is reduced to a binary form, normalized from 0-1.

    Parameters
    ----------
    x: array-like
        The array we wish to erode
    blend_size: int
        The amount of erosion we wish to perform. This number corresponds to the
        maximum distance inside the edge of the support of x.

    Returns
    -------
    array-like:
        The eroded version of x.

    """
    # Get dtype and backend
    dtype = getDatatype(x)
    backend = getBackend(x)

    # Normalize x
    _x = dcopy(x)
    _x[:] -= min(x)
    _x[:] = asarray(_x > 0, dtype, backend)

    # Create triangular wave
    s = sig.triang(2 * blend_size + 1)
    triang_2d = s[:, np.newaxis] * s
    triang_2d /= sum(triang_2d)
    triang_2d = asarray(triang_2d, dtype, backend)

    # Generate crop operator which pads edges for convolutionx
    padded_size = [scipy.fftpack.next_fast_len(int(s + 2 * blend_size)) for s in shape(_x)]
    x_padded = pad(_x, padded_size, pad_value='edge', center=True)

    # Pad triangular wave as well
    triang_2d_padded = pad(triang_2d, padded_size, center=True, pad_value=0)

    coords = argmax(triang_2d_padded)
    triang_2d_padded = zeros_like(triang_2d_padded)
    triang_2d_padded[coords] = 1.0

    # Perform FFT convolution
    x_filtered = iFt(Ft(triang_2d_padded) * Ft(x_padded))
    # x_filtered = sig.fftconvolve(np.asarray(triang_2d_padded), np.asarray(x_padded), mode='same')
    x_filtered = crop(x_filtered, shape(x), center=True)

    # # Threshold to within existing range
    x_filtered[:] -= 0.5
    x_filtered[:] *= (real(x_filtered) >= 0)
    x_filtered[:] *= 2

    # Return
    return x_filtered


def softThreshold(x, alpha):
    """Soft-threshold operator."""
    if 'complex' not in getDatatype(x):
        x = astype(x, 'complex32')
        return real(exp(1j * angle(x)) * (abs(x) - alpha) * (abs(x) > alpha))
    else:
        return exp(1j * angle(x)) * (abs(x) - alpha) * (abs(x) > alpha)


def _gaussianKernel(shape, sigma, dtype=None, backend=None):

    # Get datatype and backend
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    # Check length
    assert len(shape) == 2, 'Only 2D kernels are supported'

    # Create kernel
    if backend == 'numpy':
        max_dim = max(shape)
        w = scipy.signal.gaussian(max_dim, sigma)
        kernel = crop(w[:, np.newaxis] * w[np.newaxis, :], shape, center=True)
        return kernel / sum(kernel)
    elif backend == 'arrayfire':
        return arrayfire.image.gaussian_kernel(shape[0], shape[1], sigma, sigma)
    else:
        raise ValueError('Backend %s is not implemented!' % backend)


@numpy_function
@real_valued_function
def downsample(x, factor=2):
    """
    Decimate an input along all axes.
    """

    # Calculate new size
    if type(factor) in (list, tuple):
        new_size = tuple([sh // f for (sh, f) in zip(shape(x), factor)])
    else:
        new_size = tuple([sh // factor for sh in shape(x)])

    # Return new size
    return skimage.transform.resize(x, new_size, anti_aliasing=False, mode='reflect', preserve_range=True)


def decimate(x, factor=2):
    """Convenience function for downsample."""
    return downsample(x, factor=factor)


def bin(image, factor):
    """ Reduce image size by binning. """
    image_binned = zeros([sh // factor for sh in shape(image)], dtype=getDatatype(image), backend=getBackend(image))
    bin_shape = shape(image_binned)
    for x in range(factor):
        for y in range(factor):
            image_binned += image[y:bin_shape[0] * factor:factor, x:bin_shape[1] * factor:factor]

    image_binned /= factor * factor

    return image_binned




@numpy_function
@real_valued_function
def upsample(x, factor=2):
    """
    Decimate an input along all axes.
    """

    # Calculate new size
    new_size = tuple([int(sh * factor) for sh in shape(x)])

    # Return new size
    return skimage.transform.resize(x, new_size, preserve_range=True, anti_aliasing=True, mode='edge')

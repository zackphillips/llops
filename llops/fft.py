"""
Copyright 2017 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import scipy as sp
from scipy.fftpack import next_fast_len
from scipy.linalg import dft
from .base import alloc, abs, flip, real, fftshift, ifftshift, getDatatype, getBackend, setByteOrder, shape, pad, crop, transpose, astype, conj, next_fast_even_number, where, asarray, roll
from .config import valid_fft_backends, default_dtype, default_backend, default_fft_backend
from .roi import boundingBox

# Try to import arrayfire
try:
    import arrayfire
except ModuleNotFoundError:
    pass

# Try to import pyfftw
try:
    import pyfftw
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)
except ModuleNotFoundError:
    pass


def dftMatrix(size):
    """Form an explicit DFT matrix."""
    M = 1
    for dim in size:
        m = dft(dim)
        M = np.kron(M, m)
    return(M)


def fft(x, axes=None, center=True, normalize=False, backend=None, inverse=False,
        fft_backend=None, pad=False, y=None, allow_c2r=False):
    """Perform the FFT of an input."""
    # Get backend
    if backend is None:
        backend = getBackend(x)

    # Determine optimal size to pad, if desired
    original_size = shape(x)
    if pad:
        padded_size = list(original_size)
        for ind, d in enumerate(original_size):
            if next_fast_len(d) != d:
                padded_size[ind] = next_fast_len(d)
    else:
        padded_size = original_size

    # Get FFT functions
    fft_fun, ifft_fun = fftfuncs(padded_size, axes, center,
                                 normalize, getDatatype(x),
                                 backend, fft_backend,
                                 allow_c2r=allow_c2r)

    # Select FFT inverse
    FFT = fft_fun if not inverse else ifft_fun

    # Set correct byte order
    x[:] = setByteOrder(x, 'f')

    if padded_size is not original_size:
        return crop(FFT(pad(x, padded_size, center=True), y=y), original_size, center=True)
    else:
        return FFT(x, y=y)


def Ft(x, axes=None, center=True, normalize=False, backend=None, fft_backend=None,
       pad=False, y=None):
    """Shorthand implementation of Fourier Transform."""
    return fft(x, axes=axes, center=center, normalize=normalize, backend=backend,
               inverse=False, fft_backend=fft_backend, y=y, allow_c2r=False)


def iFt(x, axes=None, center=True, normalize=False, backend=None, fft_backend=None, pad=False, y=None):
    """Shorthand implementation of inverse Fourier Transform."""
    return fft(x, axes=axes, center=center, normalize=normalize, backend=backend,
               inverse=True, fft_backend=fft_backend, y=y, allow_c2r=False)


def fftfuncs(N, axes=None, center=True, normalize=True, dtype=default_dtype,
             backend=default_backend, fft_backend=default_fft_backend,
             in_place=False, allow_c2r=False):
    """
    Generate FFT and IFFT functions with FFTW library.
    Parameters
    ----------
    N : tuple of int, input shape
    axes : tuple of int, axes to perform FFT
    center : bool, optional
    num_threads : int, optional
    normalize : bool, optional
    dtype : data-type, optional

    Returns fft_fun and ifft_fun.
    """

    if axes is None:
        axes = tuple(range(len(N)))
    elif type(axes) not in (list, tuple):
        axes = [axes]

    # Calculate scale
    scale = np.sqrt(np.prod([N[i] for i in axes]))

    # Define output dtypes (these may be over-ridden by allow_c2r kwarg)
    forward_output_dtype = dtype
    inverse_output_dtype = dtype

    # Ensure returned array is complex
    if 'complex' not in dtype:
        if '32' in dtype:
            inverse_output_dtype = 'float32'
            forward_output_dtype = 'complex32'
        else:
            inverse_output_dtype = 'float64'
            forward_output_dtype = 'complex64'

    # Parse FFT backend
    if fft_backend is None:
        fft_backend = default_fft_backend

    # Numpy / scipy FFT
    if backend == 'numpy':

        # Ensure thr provided fft backend is valid
        assert fft_backend in valid_fft_backends, 'FFT backend %s is not valid!' % fft_backend

        # Use fftw if it is installed
        if fft_backend == 'fftw':
            def fft_fun(x, y=None):
                # Allocate y if not provided
                if y is None:
                    return_y = True
                    y = alloc(shape(x), forward_output_dtype, getBackend(x))
                else:
                    return_y = False

                if center:
                    y[:] = sp.fftpack.fftshift(pyfftw.interfaces.numpy_fft.fft2(sp.fftpack.fftshift(x, axes=axes), axes=axes), axes=axes)
                else:
                    y[:] = pyfftw.interfaces.numpy_fft.fft2(x, axes=axes)

                if normalize:
                    y[:] /= scale

                if return_y:
                    return y

            def ifft_fun(x, y=None):
                # Allocate y if not provided
                if y is None:
                    return_y = True
                    y = alloc(shape(x), inverse_output_dtype, getBackend(x))
                else:
                    return_y = False

                if center:
                    y[:] = astype(sp.fftpack.ifftshift(pyfftw.interfaces.numpy_fft.ifft2(sp.fftpack.ifftshift(x, axes=axes), axes=axes), axes=axes), inverse_output_dtype)
                else:
                    y[:] = pyfftw.interfaces.numpy_fft.ifft2(x, axes=axes)

                if normalize:
                    y[:] *= scale

                if return_y:
                    return y

        # Use scipy as second option
        elif fft_backend == 'scipy':
            def fft_fun(x, y=None):
                # Allocate y if not provided
                if y is None:
                    return_y = True
                    y = alloc(shape(x), forward_output_dtype, getBackend(x))
                else:
                    return_y = False

                if center:
                    y[:] = sp.fftpack.fftshift(sp.fftpack.fft2(sp.fftpack.fftshift(x, axes=axes), axes=axes), axes=axes)
                else:
                    y[:] = sp.fftpack.fft2(x)

                if normalize:
                    y /= scale

                if return_y:
                    return y

            def ifft_fun(x, y=None):
                # Allocate y if not provided
                if y is None:
                    return_y = True
                    y = alloc(shape(x), inverse_output_dtype, getBackend(x))
                else:
                    return_y = False

                if center:
                    y[:] = sp.fftpack.ifftshift(sp.fftpack.ifft2(sp.fftpack.ifftshift(x, axes=axes), axes=axes), axes=axes)
                else:
                    y[:] = sp.fftpack.ifft2(x, axes=axes)

                if normalize:
                    y[:] *= scale

                if return_y:
                    return y

        # Or use numpy as a last resort
        elif fft_backend == 'numpy':
            def fft_fun(x, y=None):
                # Allocate y if not provided
                if y is None:
                    return_y = True
                    y = alloc(shape(x), forward_output_dtype, getBackend(x))
                else:
                    return_y = False

                if center:
                    y[:] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=axes), axes=axes), axes=axes)
                else:
                    y[:] = np.fft.fft2(x)

                if normalize:
                    y[:] /= scale

                if return_y:
                    return y

            def ifft_fun(x, y=None):
                # Allocate y if not provided
                if y is None:
                    return_y = True
                    y = alloc(shape(x), inverse_output_dtype, getBackend(x))
                else:
                    return_y = False

                if center:
                    y[:] = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
                else:
                    y[:] = np.fft.ifft2(x)

                if normalize:
                    y[:] *= scale

                if return_y:
                    return y

    # Arrayfire backend
    elif backend == 'arrayfire':

        # Define fft function based on number of axes specified, and whether to
        # perform this fft as an in-place operation.
        if in_place:
            if len(axes) == 1:
                _af_fft = arrayfire.signal.fft_inplace
                _af_ifft = arrayfire.signal.ifft_inplace
            elif len(axes) == 2:
                _af_fft = arrayfire.signal.fft2_inplace
                _af_ifft = arrayfire.signal.ifft2_inplace
            elif len(axes) == 3:
                _af_fft = arrayfire.signal.fft3_inplace
                _af_ifft = arrayfire.signal.ifft3_inplace
            else:
                raise ValueError('Arrayfire does not support nd-fft with > 3 dims')

        elif 'complex' not in dtype and allow_c2r:
            if len(axes) == 1:
                _af_fft = arrayfire.signal.fft_r2c
                _af_ifft = arrayfire.signal.fft_c2r
            elif len(axes) == 2:
                _af_fft = arrayfire.signal.fft2_r2c
                _af_ifft = arrayfire.signal.fft2_c2r
            elif len(axes) == 3:
                _af_fft = arrayfire.signal.fft3_r2c
                _af_ifft = arrayfire.signal.fft3_c2r
            else:
                raise ValueError('Arrayfire does not support nd-fft with > 3 dims')
        else:
            if len(axes) == 1:
                _af_fft = arrayfire.signal.fft
                _af_ifft = arrayfire.signal.ifft
            elif len(axes) == 2:
                _af_fft = arrayfire.signal.fft2
                _af_ifft = arrayfire.signal.ifft2
            elif len(axes) == 3:
                _af_fft = arrayfire.signal.fft3
                _af_ifft = arrayfire.signal.ifft3
            else:
                raise ValueError('Arrayfire does not support nd-fft with > 3 dims')

        # Unlike numpy / scipy / fftw, arrayfire has seperate functions
        # based on dimensionality. Thus, need to assign this function first.
        if len(axes) != len(N) and any([ax != n for (ax, n) in zip(axes, range(len(N)))]):

            # Determine necessary re-ordered dimensions
            dims = list(range((len(N))))
            for axis in axes:
                dims.pop(axis)
            fft_dims = axes + dims

            # Turn dimensions into dictionary
            fft_dims_dict, fft_dims_dict_reversed = {}, {}
            for index in range(len(fft_dims)):
                label = 'd' + str(index)
                fft_dims_dict[label] = fft_dims[index]
                fft_dims_dict_reversed[label] = fft_dims[-(index + 1)]

            # Wrap the fft and ifft functions in reorder functions
            def af_fft(x, scale):
                return arrayfire.data.reorder(_af_fft(arrayfire.data.reorder(x, **fft_dims_dict), scale=scale), **fft_dims_dict)

            def af_ifft(x, scale):
                return arrayfire.data.reorder(_af_ifft(arrayfire.data.reorder(x, **fft_dims_dict), scale=scale), **fft_dims_dict)
        else:
            af_fft, af_ifft = _af_fft, _af_ifft

        def fft_fun(x, y=None):

            # Allocate y if not provided
            if y is None and not in_place:
                return_y = True
                y = alloc(shape(x), forward_output_dtype, getBackend(x))
            else:
                return_y = False

            if center:
                y[:] = fftshift(astype(af_fft(fftshift(x), scale=1/scale), forward_output_dtype))
            else:
                y[:] = af_fft(x, scale=1/scale)

            if normalize:
                y[:] = y / scale

            if return_y:
                return y

        def ifft_fun(x, y=None):
            # Allocate y if not provided
            if y is None and not in_place:
                return_y = True
                y = alloc(shape(x), inverse_output_dtype, getBackend(x))
            else:
                return_y = False

            if center:
                y[:] = ifftshift(astype(af_ifft(ifftshift(x), scale=1/scale), inverse_output_dtype))
            else:
                y[:] = af_ifft(x, scale=1/scale)

            if normalize:
                y[:] = y * scale

            if return_y:
                return y
    else:
        raise NotImplementedError('Backend %s is not supported!' % backend)

    return fft_fun, ifft_fun


def conv_functions(input_shape, kernel, mode='same', axis=None, debug=False,
                   pad_value='edge', pad_convolution=True, fourier_input=False,
                   pad_fft=True, fft_backend=None, force_full=False):
    """Inverse and forward convolution functions."""

    # Get dtype and backend
    dtype = getDatatype(kernel)
    backend = getBackend(kernel)

    # Parse circular input condition
    if mode == 'circular':
        pad_convolution = False
        pad_fft = False

    # Fourier input only supports same-size, unpadded convolution
    if fourier_input:
        mode = 'same'
        pad_convolution = False
        pad_fft = False

    # Determine size of kernel
    if pad_convolution:
        kernel_start, kernel_size = boundingBox(flip(kernel))

        # Get rid of extra dimension in size (1 -> 0)
        kernel_size_even = [sz // 2 * 2 for sz in kernel_size]

        # Determine crop start
        crop_start = [k_sz // 2 for k_sz in kernel_size_even]

        # Determine convolution shape
        conv_shape = tuple([sh + ksh for (sh, ksh) in zip(input_shape, kernel_size)])

        # Determine valid shape
        valid_shape = tuple([(sz_x - sz_h + 1) for (sz_x, sz_h) in zip(input_shape, kernel_size)])
    else:
        kernel_size = (0, 0)
        crop_start = (0, 0)
        conv_shape = input_shape

    # Determine optimal size for FFT
    if pad_fft:
        conv_shape = tuple([sp.fftpack.next_fast_len(int(sz)) for sz in conv_shape])

    # Determine output shape
    if mode == 'valid':
        output_shape = valid_shape
    elif mode == 'full':
        output_shape = conv_shape
    elif mode == 'same':
        output_shape = input_shape
    elif mode == 'circular':
        output_shape = input_shape
    else:
        raise ValueError('Invalid mode %s.' % mode)


    # Create FFT functions
    fft, ifft = fftfuncs(conv_shape,
                         axes=axis,
                         center=True,
                         normalize=False,
                         dtype=dtype,
                         backend=backend,
                         fft_backend=fft_backend,
                         in_place=False)

    # Pad kernel to conv shape
    kernel = pad(kernel, conv_shape, pad_value=0, center=True)

    # Pre-process kernel for forward, adjoint, and inverse
    kernel_f = fft(kernel)
    kernel_f_conj = conj(kernel_f)
    kernel_f_intensity = abs(kernel_f) ** 2

    # Initialize padded input array
    x_padded = alloc(conv_shape, dtype, backend)
    conv = alloc(conv_shape, dtype, backend)

    # Check if kernel is a delta function - if so, calculate shift
    if not force_full and len(where(kernel)) == 1:

        # Get position of delta function
        position = asarray(where(kernel > 1e-6)[0])
        shift = position - asarray(shape(kernel)) / 2.0

        # Forward convolution function
        def conv_forward(x, y=None):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            # Pad input
            pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

            # Perform convolution
            conv[:] = roll(x, shift)

            # Crop to output
            crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

        # Adjoint convolution function
        def conv_adjoint(x, y=None):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            # Pad input
            pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

            # Perform convolution
            conv[:] = roll(x, -1 * shift)

            # Crop to output
            crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

        # Adjoint convolution function
        def conv_inverse(x, y=None, regularization=0):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            # Pad input
            pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

            # Perform convolution
            conv[:] = roll(x, -1 * shift)

            # Crop to output
            crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

    elif backend == 'numpy':

        # Forward convolution function
        def conv_forward(x, y=None):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            # Pad input
            pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

            # Perform convolution
            conv[:] = ifft(fft(x_padded) * kernel_f)

            # Crop to output
            crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

        # Adjoint convolution function
        def conv_adjoint(x, y=None):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            # Pad input
            pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

            # Perform convolution
            conv[:] = ifft(fft(x_padded) * kernel_f_conj)

            # Crop to output
            crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

        # Adjoint convolution function
        def conv_inverse(x, y=None, regularization=0):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            # Apply regularization
            if regularization is None:
                kernel_f_inv = kernel_f_conj / (kernel_f_intensity)
            else:
                kernel_f_inv = kernel_f_conj / (kernel_f_intensity + regularization)

            # Pad input
            pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

            # Perform convolution
            conv[:] = ifft(fft(x_padded) * kernel_f_inv)

            # Crop to output
            crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

    elif backend == 'arrayfire':
        # import arrayfire as af

        # Forward convolution function
        def conv_forward(x, y=None):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            if mode == 'circular':
                y[:] = astype(ifft(fft(x) * kernel_f), dtype)
            else:
                # Pad input
                pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

                # Perform convolution
                conv[:] = astype(ifft(fft(x_padded) * kernel_f), dtype)
                # conv[:] = astype(af.signal.convolve2(x_padded, kernel), dtype)

                # Crop to output
                crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

        # Adjoint convolution function
        def conv_adjoint(x, y=None):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            if mode == 'circular':
                y[:] = astype(ifft(fft(x) * kernel_f_conj), dtype)
            else:
                # Pad input
                pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

                # Perform convolution
                conv[:] = astype(ifft(fft(x_padded) * kernel_f_conj), dtype)

                # Crop to output
                crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

        # Adjoint convolution function
        def conv_inverse(x, y=None, regularization=0):

            # Allocate output
            if y is None:
                return_y = True
                y = alloc(output_shape, dtype, backend)
            else:
                return_y = False

            # Apply regularization
            kernel_f_inv = kernel_f_conj / (kernel_f_intensity + regularization)

            if mode == 'circular':
                y[:] = astype(ifft(fft(x) * kernel_f_inv), dtype)
            else:
                # Pad input
                pad(x, conv_shape, crop_start=crop_start, pad_value=pad_value, y=x_padded)

                # Perform convolution
                conv[:] = astype(ifft(fft(x_padded) * kernel_f_inv), dtype)

                # Crop to output
                crop(astype(conv, dtype), output_shape, crop_start=crop_start, y=y)

            # Return output
            if return_y:
                return y

    # Return functions and output_shape
    return conv_forward, conv_adjoint, conv_inverse, output_shape


def convolve(x, h, mode='same', axis=None, debug=False,
             pad_convolution=True, pad_value='edge', pad_fft=True,
             fourier_input=False, y=None, in_place=False,
             fft_functions=None, fft_backend=None):
    """Convolution function."""

    # Get convolution functions
    conv_func, conv_adj_func, conv_inv_func, _ = conv_functions(shape(x), h,
                                                                mode='same',
                                                                axis=axis,
                                                                pad_value=pad_value,
                                                                pad_convolution=pad_convolution,
                                                                fourier_input=fourier_input,
                                                                pad_fft=pad_fft,
                                                                fft_backend=fft_backend)

    # Perform convoluntion
    return conv_func(x, y=y)


def deconvolve(x, h, reg=0, mode='same', axis=None, debug=False,
               pad_convolution=True, pad_value='edge', pad_fft=True,
               fourier_input=False, y=None, in_place=False,
               fft_functions=None, fft_backend=None):
    """Convolution function."""

    # Get convolution functions
    conv_func, conv_adj_func, conv_inv_func, _ = conv_functions(shape(x), h,
                                                                mode='same',
                                                                axis=axis,
                                                                pad_value=pad_value,
                                                                pad_convolution=pad_convolution,
                                                                fourier_input=fourier_input,
                                                                pad_fft=pad_fft,
                                                                fft_backend=fft_backend)

    # Perform convoluntion
    return conv_inv_func(x, y=y, regularization=reg)


def fftpad(x):
    return pad(x, [next_fast_even_number(sh) for sh in shape(x)], center=True)


def interpolate(x, interpolation_factor=2):

    # Pad x to appropriate shape for F.T.
    x_padded = fftpad(x)

    # Determine interpolated shape
    interpolated_shape = tuple([next_fast_even_number(sh * interpolation_factor) for sh in shape(x_padded)])

    # Take fourier transform of input
    x_f = Ft(x_padded)

    # Pad in frequency domain
    x_f_padded = pad(x_f, interpolated_shape, center=True, pad_value=0)

    # Perform interpolation
    x_interpolated = iFt(x_f_padded) * interpolation_factor * 4

    # Return
    return crop(abs(x_interpolated), [int(sh * interpolation_factor) for sh in shape(x)], center=True)

"""
Copyright 2017 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from .operators import Operator, Crop, FourierTransform, Diagonalize, Sum, PhaseRamp, Identity, Shift, Convolution
from .stack import Hstack, Vstack

from llops import config, abs, min, max, real, zeros_like, isnan, squeeze, sum, boundingBox, crop, where, astype, dcopy, tile, size, roll, Ft, iFt, reshape, asarray, argmax, zeros, abs, conj, ndim, getBackend, getDatatype, vec, fftshift, getNativeDatatype, changeBackend, where, shape, scalar, asbackend
import types
import math

from scipy.fftpack import next_fast_len

__all__ = ['Derivative', 'CrossCorrelation', 'OperatorSum', 'Crop',
           'Pad', 'Registration', 'FFTShift']


def Tile(N, axes, reps, normalize=True, dtype=None, backend=None,
         repr_latex=None):

    # Configure backend and datatype
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    assert (len(axes) == len(reps))
    M = list(N)
    for i in range(len(axes)):
        M[axes[i]] = reps[i]
    M = tuple(M)
    A = Sum(M, axes=axes, normalize=normalize, dtype=dtype, backend=backend).H
    A.__repr__ = 'Tile'

    return A


def ConvolutionOld(kernel, dtype=None, backend=None, normalize=False,
                mode='circular', label='C', pad_value='mean', pad_size=None,
                fft_backend=None, inverse_regularizer=0,
                center=False, inside_operator=None,
                force_full_convolution=False):
    """Convolution linear operator"""

    # Get temporary kernel to account for inner_operator size
    _kernel = kernel if inside_operator is None else inside_operator * kernel

    # Check number of dimensions
    N = _kernel.shape
    dim_count = len(N)
    assert dim_count == ndim(_kernel)

    # Get kernel backend
    if backend is None:
        backend = getBackend(kernel)
    else:
        # Convert kernel to provided backend
        kernel = asbackend(kernel, backend)

    # Get kernel dtype
    if dtype is None:
        dtype = getDatatype(kernel)
    else:
        kernel = astype(kernel, dtype)

    # Determine if the kernel is a shifted delta function - if so, return a
    # shift operator masked as a convolution
    position_list = tuple([tuple(np.asarray(pos) - np.asarray(shape(_kernel)) // 2) for pos in where(_kernel != 0.0)])
    mode = 'discrete' if len(position_list) == 1 and not force_full_convolution else mode

    # Discrete convolution
    if mode == 'discrete':
        # Create shift operator, or identity operator if there is no shift.
        if all([pos == 0.0 for pos in position_list[0]]):
            op = Identity(N)
        else:
            op = Shift(N, position_list[0])

        loc = where(_kernel != 0)[0]

        # If the kernel is not binary, normalize to the correct value
        if scalar(_kernel[loc[0], loc[1]]) != 1.0:
            op *= scalar(_kernel[loc[0], loc[1]])

        # Update label to indicate this is a shift-based convolution
        label += '_{shift}'

        # Normalize if desired
        if normalize:
            op *= 1 / np.sqrt(np.size(kernel))

    elif mode in ['windowed', 'circular']:
        # The only difference between circular and non-circular convolution is
        # the pad size. We'll define this first, then define the convolution in
        # a common framework.

        if mode == 'circular':

            # Pad kernel to effecient size
            N_pad = list(N)
            for ind, d in enumerate(N):
                if next_fast_len(d) != d:
                    N_pad[ind] = next_fast_len(d)

            crop_start = [0] * len(N_pad)

        elif mode == 'windowed':
            if pad_size is None:
                # Determine support of kernel
                kernel_support_roi = boundingBox(kernel, return_roi=True)
                N_pad_raw = (np.asarray(N) + np.asarray(kernel_support_roi.size).tolist())
                N_pad = [next_fast_len(sz) for sz in N_pad_raw]

                # Create pad and crop operator
                crop_start = [(N_pad[dim] - N[dim]) // 2 for dim in range(len(N))]
            else:
                if type(pad_size) not in (list, tuple, np.ndarray):
                    pad_size = (pad_size, pad_size)
                N_pad = (pad_size[0] + N[0], pad_size[1] + N[1])

                crop_start = None

        # Create pad operator
        P = Pad(N, N_pad, pad_value=pad_value, pad_start=crop_start,
                backend=backend, dtype=dtype)

        # Create F.T. operator
        F = FourierTransform(N_pad, dtype=dtype, backend=backend,
                             normalize=normalize,
                             fft_backend=fft_backend,
                             pad=False,
                             center=center)

        # Optionally create FFTShift operator
        if not center:
            FFTS = FFTShift(N_pad, dtype=dtype, backend=backend)
        else:
            FFTS = Identity(N_pad, dtype=dtype, backend=backend)

        # Diagonalize kernel
        K = Diagonalize(kernel, inside_operator=F * FFTS * P,
                        inverse_regularizer=inverse_regularizer, label=label)

        # Generate composite op
        op = P.H * F.H * K * F * P

        # Define inversion function
        def _inverse(self, x, y):
            # Get current kernel
            kernel_f = F * FFTS * P * kernel

            # Invert and create operator
            kernel_f_inv = conj(kernel_f) / (abs(kernel_f) ** 2 + self.inverse_regularizer)
            K_inverse = Diagonalize(kernel_f_inv, backend=backend, dtype=dtype, label=label)

            # Set output
            y[:] = P.H * F.H * K_inverse * F * P * x

        # Set inverse function
        op._inverse = types.MethodType(_inverse, op)

    else:
        raise ValueError(
            'Convolution mode %s is not defined! Valid options are "circular" and "windowed"'
            % mode)

    # Append type to label
    if '_' not in label:
        label += '_{' + mode + '}'

    # Set label
    op.label = label

    # Set inverse_regularizer
    op.inverse_regularizer = inverse_regularizer

    # Set latex to be just label
    def repr_latex(latex_input=None):
        if latex_input is None:
            return op.label
        else:
            return op.label + ' \\times ' + latex_input
    op.repr_latex = repr_latex

    return op


def CrossCorrelation(kernel, mode='circular', dtype=None, backend=None, label='X',
                     pad_value=0, axis=None, pad_convolution=True, pad_fft=True,
                     invalid_support_value=1, fft_backend=None):

    # Flip kernel
    kernel = iFt(conj(Ft(kernel)))

    # Call Convolution
    return Convolution(kernel, mode=mode, dtype=dtype, backend=backend, label=label,
                       pad_value=pad_value, axis=None, pad_convolution=pad_convolution,
                       pad_fft=pad_fft, invalid_support_value=invalid_support_value,
                       fft_backend=fft_backend)


def Derivative(N, axis=0, dtype=None, backend=None, label=None):

    # Configure backend and datatype
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    # Generate shift grid using numpy
    r_x = np.arange(-N[1] / 2, N[1] / 2, 1.0) / N[1]
    r_y = np.arange(-N[0] / 2, N[0] / 2, 1.0) / N[0]
    grid_np = np.meshgrid(r_x, r_y)

    # Convert to correct backend and datatype
    grid = []
    for g in grid_np:
        grid.append(
            changeBackend(g.astype(getNativeDatatype(dtype, 'numpy')), backend))

    # Generate operator
    G = Diagonalize(2 * np.pi * 1j * grid[axis], dtype=dtype, backend=backend)
    F = FourierTransform(N, dtype=dtype, backend=backend)
    op = F.H * G * F

    # Set label and latex representation
    if label is None:
        if (axis == 0):
            op.label = "∂y"
            latex_str = "\\frac{\partial}{\partial y}"
        elif (axis == 1):
            op.label = "∂x"
            latex_str = "\\frac{\partial}{\partial x}"
    else:
        op.label = label
        latex_str = label

    # Set latex to be just label
    def repr_latex(latex_input=None):
        if latex_input is None:
            return latex_str
        else:
            return latex_str + ' \\times ' + latex_input

    op.repr_latex = repr_latex

    # Set condition number to 1 TODO: The operators above should have condition 1
    op.condition_number = 1.0

    return (op)


def Gradient(N, dtype=None, backend=None, label=None):

    # Configure backend and datatype
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    # Make derivative operators
    Dy = Derivative(N, axis=0, dtype=dtype, backend=backend)
    Dx = Derivative(N, axis=1, dtype=dtype, backend=backend)
    return (Dx + Dy)


def Pad(N, M, pad_start=None, pad_value=0, dtype=None, backend=None, label='P',
        out_of_bounds_placeholder=None):
    """
    Zero-Pad linear operator
    While this is a composite operator, we keep it here because it is closely related to the Crop Operator
    """

    # Configure backend and datatype
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype

    # Check to ensure that the pad amount is >= the initial amount
    assert all(M[i] >= N[i] for i in range(len(M)))

    if any(M[i] > N[i] for i in range(len(M))):
        A = Crop(M, N, crop_start=pad_start, pad_value=pad_value,
                 dtype=dtype, backend=backend,
                 out_of_bounds_placeholder=out_of_bounds_placeholder).H

        A.repr_str = 'Pad (with %s)' % pad_value

        # Update latex function
        def repr_latex(latex_input=None):
            if latex_input is None:
                return label
            else:
                return label + ' \\times ' + latex_input

        A.repr_latex = repr_latex
    else:
        A = Identity(N, dtype=dtype, backend=backend)

    return A


def OperatorSum(op_list):
    """Sum of many operators."""
    assert all([op_list[0].shape == op.shape for op in op_list]), "All operators must have the same output shape!"
    assert all([op_list[0].dtype == op.dtype for op in op_list]), "All operators must have the same datatype!"
    assert all([op_list[0].backend == op.backend for op in op_list]), "All operators must have the same backend!"

    # Turn an addition into a Multiplication
    Id = Identity(op_list[0].M, op_list[0].dtype, op_list[0].backend)
    I_ = Hstack([Id] * len(op_list))
    O_ = Vstack(op_list)

    # This is the operator we're going to return - just need to over-ride (monkey-patch) a few methods
    sum_op = I_ * O_

    # Define new label
    sum_op.label = '\\Sigma'

    # Define new latex operator
    def _latex(latex_input=None):
        if all([op_list[0].label == op_list[i].label for i in range(len(op_list))]):
            latex = '\\Sigma_{i=0}^{' + str(len(op_list)) + '}(' + op_list[0].repr_latex(latex_input) + ')_i'
        else:
            latex = '\\Sigma_{i=0}^{' + str(len(op_list)) + '} O_i \\times ' + latex_input

        return latex

    sum_op.repr_latex = _latex

    # Determine condition number
    if all([op.linear for op in op_list]):
        sum_op.condition_number = max([op.condition_number for op in op_list])

    # Return this new operator
    return sum_op


def Registration(array_to_register_to, dtype=None, backend=None, label='R', inside_operator=None,
                 center=False, axes=None, debug=False):
    """Registeration operator for input x and operator input."""
    # Configure backend and datatype
    backend = backend if backend is not None else config.default_backend
    dtype = dtype if dtype is not None else config.default_dtype
    _shape = shape(array_to_register_to) if inside_operator is None else inside_operator.M
    axes = axes if axes is not None else tuple(range(ndim(array_to_register_to)))

    # Create sub-operators
    PR = PhaseRamp(_shape, dtype, backend, center=center, axes=axes)
    F = FourierTransform(_shape, dtype, backend, center=center, axes=axes)
    X = Diagonalize(-1 * array_to_register_to, dtype, backend, inside_operator=F * inside_operator, label='x')
    _R = X * PR

    # Compute Fourier Transform of array to register to
    array_to_register_to_f = F * array_to_register_to

    # Define inverse
    def _inverse(x, y):
        """Inverse using phase correlation."""
        # Extract two arrays to correlate
        xf_1 = array_to_register_to_f
        xf_2 = x

        # Compute normalized cross-correlation
        phasor = (conj(xf_1) * xf_2) / abs(conj(xf_1) * xf_2)
        phasor[isnan(phasor)] = 0

        # Convert phasor to delta function
        delta = F.H * phasor

        # If axes is defined, return only one axis
        if len(axes) != ndim(x) or any([ax != index for (ax, index) in zip(axes, range(len(axes)))]):
            axes_not_used = [index for index in range(ndim(x)) if index not in axes]

            delta = squeeze(sum(delta, axes=axes_not_used))

        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 3))
            plt.subplot(131)
            plt.imshow(abs(F.H * xf_1))
            plt.subplot(132)
            plt.imshow(abs(F.H * xf_2))
            plt.subplot(133)
            if ndim(delta) > 1:
                plt.imshow(abs(delta))
            else:
                plt.plot(abs(delta))

        # Calculate maximum and return
        if not center:
            y[:] = reshape(asarray(argmax(delta)), shape(y))
        else:
            y[:] = reshape(asarray(argmax(delta)) - asarray(delta.shape) / 2, shape(y))

        # Deal with negative values
        sizes = reshape(asarray([_shape[ax] for ax in axes]), shape(y))
        mask = y[:] > sizes / 2
        y[:] -= mask * sizes

        if debug:
            plt.title(str(np.real(np.asarray(argmax(delta))).tolist()) + ' ' + str(np.abs(np.asarray(y).ravel())))

    # Define operator name
    repr_str = 'Registration'

    # Create a new operator from phase ramp
    R = Operator(_R.shape, _R.dtype, _R.backend, repr_str=repr_str, label=label,
                 forward=_R.forward,  # Don't provide adjoint, implies nonlinear
                 gradient=_R._gradient, inverse=_inverse,
                 cost=_R.cost, convex=_R.convex, smooth=True,
                 set_arguments_function=X._setArgumentsFunction,
                 get_arguments_function=X._getArgumentsFunction,
                 inverse_regularizer=_R.inverse_regularizer,
                 repr_latex=_R.repr_latex)

    def show_xc(x, figsize=(11, 3)):
        xf_1 = F * _R.arguments[0]
        xf_2 = F * x

        # Compute normalized cross-correlation
        phasor = (conj(xf_1) * xf_2) / abs(conj(xf_1) * xf_2)

        import matplotlib.pyplot as plt
        import llops as yp

        plt.figure(figsize=figsize)
        plt.subplot(121)
        plt.imshow(yp.angle(phasor))
        plt.title('Phase of Frequency domain')
        plt.subplot(122)
        plt.imshow(yp.abs(yp.iFt(phasor)))
        plt.title('Amplitude of Object domain')

    # Return operator with prepended inverse Fourier Transform
    op = F.H * R

    # Set the show_xc in this operator, since it is removed by multiplication
    op.show_xc = show_xc

    # Set the set and get argument functions
    op._setArgumentsFunction = R._setArgumentsFunction
    op._getArgumentsFunction = R._getArgumentsFunction

    # Return result
    return op


def FFTShift(N, dtype=None, backend=None, axes=None, label='S_FFT'):

    _shift = tuple([math.ceil(n / 2) for n in N])
    return Shift(N, _shift, dtype=dtype, backend=backend)

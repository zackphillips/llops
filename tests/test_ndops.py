import numpy as np
import scipy as sp
import scipy.misc as misc
import imageio
from numpy.testing import assert_array_almost_equal

# Libwallerlab imports
import ndops as ops
import llops as yp
from llops.fft import Ft, iFt

eps = 1e-3
global_dtype = 'complex32'
global_backend = 'numpy'

object_file_name = '../../../common/test_images/brain_to_scan.png'   # Image to use when generating object
object_color_channel = 2                                                # Color channel to use when generating object
image_size = np.array([32, 64])                                         # Image size to simulate
h_size = np.array([4, 4])                                               # Convolution kernel size
n_illum = 10                                                            # number of illumination values

class Test:

    def setup_method(self, test_method):
        # Load object and crop to size
        x_0 = yp.rand(image_size)

        # Convert object to desired backend
        self.x = yp.changeBackend(x_0, global_backend)

        # Generate convolution kernel h
        h_size = np.array([4, 4])
        self.h = yp.zeros(image_size, global_dtype, global_backend)
        self.h[image_size[0] // 2 - h_size[0] // 2:image_size[0] // 2 + h_size[0] // 2,
          image_size[1] // 2 - h_size[1] // 2:image_size[1] // 2 + h_size[1] // 2] = yp.randn((h_size[0], h_size[1]), global_dtype, global_backend)

        # A = ops.Convolution(image_size, h, dtype=global_dtype, fft_backend='numpy', backend=global_backend)
        self.A = ops.FourierTransform(image_size, dtype=global_dtype, backend=global_backend, center=True)
        self.y = self.A(yp.vectorize(self.x))

    def test_mechanical_operator_vector_sum(self):
        ''' Mechanical test of an operator-vector sum '''
        # Test sum operations here
        A_s = self.A + self.y

        # Forward operator
        assert np.sum(np.abs(A_s * yp.vec(self.x) - (self.A * yp.vec(self.x) + self.y))) < eps

        # Adjoint
        assert np.sum(np.abs(A_s.H(yp.vec(self.x)) - self.A.H(yp.vec(self.x)))) < eps

        # Gradient Numerical Check
        self.A.gradient_check()

    def test_methanical_condition_number(self):
        ''' Mechanical test of condition number calculation '''

        # Unitary Matrix
        F = ops.FourierTransform(image_size, dtype=global_dtype, backend=global_backend)
        assert F.condition_number == 1
        assert not F.condition_number_is_upper_bound

        # Matrix with a condition number
        hh = yp.changeBackend((np.random.rand(image_size[0], image_size[1]) + 0.1).astype(np.complex64), global_backend)
        D = ops.Diagonalize(hh, dtype=global_dtype, backend=global_backend)
        assert not D.condition_number_is_upper_bound

        # Product of two unitary matricies
        assert (F * F).condition_number == 1
        assert not (F * F).condition_number_is_upper_bound

        # Product of one unitary and one non-singular matrix
        assert (F * D).condition_number == D.condition_number
        assert not (F * D).condition_number_is_upper_bound # because one matrix is unitary, this condition number is NOT an upper bound. This can be checked numerically.

        # Product of two non-singular matricies.
        hh_2 = yp.changeBackend((np.random.rand(image_size[0], image_size[1]) + 0.1).astype(np.complex64), global_backend)
        D2 = ops.Diagonalize(hh_2, dtype=global_dtype, backend=global_backend)
        assert (D * D2).condition_number >= D.condition_number
        assert (D * D2).condition_number_is_upper_bound

    def test_mechanical_linear_flag(self):
        F = ops.FourierTransform(image_size, dtype=global_dtype, axes=(0, 1)) # Linear Operator
        L2 = ops.L2Norm(image_size[0] * image_size[1], dtype=global_dtype, backend=global_backend) # Non-linear operator

        assert F.linear
        assert not L2.linear
        assert not (L2 * F).linear
        assert (F + F).linear
        assert not (L2 * F + L2 * F).linear

    def test_mechanical_smooth_flag(self):
        F = ops.FourierTransform(image_size, dtype=global_dtype, axes=(0, 1)) # Linear Operator
        L2 = ops.L2Norm(image_size[0] * image_size[1], dtype=global_dtype, backend=global_backend) # Non-linear operator

        assert F.linear
        assert not L2.linear
        assert not (L2 * F).linear
        assert (F + F).linear
        assert not (L2 * F + L2 * F).linear

    def test_mechanical_sum_of_norms(self):
        L2 = ops.L2Norm(image_size[0] * image_size[1], dtype=global_dtype, backend=global_backend)
        F = ops.FourierTransform(image_size, dtype=global_dtype, axes=(0, 1))
        D = ops.Diagonalize(self.h, dtype=global_dtype, backend=global_backend)

        O_1 = L2 * ((F.H * D * F) - self.y)
        O_2 = 1e-3 * L2 * F
        O = O_1 + O_2

        # Check gradient operator (adjoint form)
        O.gradient_check()

    def test_mechanical_gradient_0(self):
        ''' Mechanical test for calculating the gradient of a single linear operator '''
        EXP = ops.Exponential(image_size, dtype=global_dtype, backend=global_backend)
        EXP.gradient_check()

    def test_mechanical_gradient_1(self):
        ''' Mechanical test for calculating the gradient of a single linear operator '''
        # Check gradient numerically
        self.A.gradient_check()

    def test_mechanical_gradient_2(self):
        ''' Mechanical test for calculating the gradient of chained linear operators '''
        F = ops.FourierTransform(image_size, dtype=global_dtype, axes=(0, 1))
        D = ops.Diagonalize(self.h, dtype=global_dtype, backend=global_backend)
        A = F.H * D * F
        A.label = 'A'

        # Check gradient numerically
        A.gradient_check()

    def test_mechanical_gradient_3(self):
        ''' Mechanical test for an inner linear operator with outer non-linear operator '''

        L2 = ops.L2Norm(image_size[0] * image_size[1], dtype=global_dtype, backend=global_backend)
        # Data difference function
        Delta = (self.A - self.y)

        # Objective Function
        O = L2 * Delta

        # Check forward operator
        assert np.all(np.abs(O(yp.vec(self.x)) - 0.5 * np.sum(np.abs(Delta * yp.vec(self.x)) ** 2)) < eps)

        # Check gradient operator (adjoint form)
        O.gradient_check()

    def test_mechanical_gradient_4(self):
        ''' Mechanical test for an outer non-linear operator with outer non-linear operator and a linear operator in-between'''
        shift_true = np.asarray((-5,3)).astype(yp.getNativeDatatype(global_dtype, 'numpy'))

        # Inner non-linear operator, linear operator in middle, and norm on outside
        F = ops.FourierTransform(image_size, dtype=global_dtype, backend=global_backend)
        D_object = ops.Diagonalize((F * yp.vec(self.x)).reshape(image_size), label='D_{object}', dtype=global_dtype, backend=global_backend)
        R = ops.PhaseRamp(image_size, dtype=global_dtype, backend=global_backend)
        A_shift = F.H * D_object * R
        y = A_shift(shift_true)
        L2 = ops.L2Norm(image_size[0] * image_size[1], dtype=global_dtype, backend=global_backend)
        objective = L2 * (A_shift - self.y)

        # Check gradient
        objective.gradient_check()

    ###### Primary Operator Tests ######
    def test_operator_identity(self):
        ''' Identity Operator '''
        I = ops.Identity(image_size, dtype=global_dtype, backend=global_backend)

        # Check forward operator
        assert np.sum(I * self.x - self.x) < eps

        # Check gradient
        I.gradient_check()

    def test_operator_diagonalize(self):
        ''' Diagonal Operator '''
        K = ops.Diagonalize(self.h, dtype=global_dtype, backend=global_backend)

        # Check forward operator
        assert(np.sum(np.abs((K * self.x) - (self.h * self.x))) < eps)

        # Check gradient
        K.gradient_check()

    def test_operator_wavelet(self):
        ''' Wavelet Transform Operator '''
        import pywt
        wavelet_list = ['db1', 'haar', 'rbio1.1', 'bior1.1', 'bior4.4', 'sym12']
        for wavelet_test in wavelet_list:
            # Wavelet Transform
            W = ops.WaveletTransform(image_size, wavelet_type=wavelet_test, use_cycle_spinning=False)

            # Check forward operation
            coeffs = pywt.wavedecn(self.x, wavelet=wavelet_test)
            x_wavelet, coeff_slices = pywt.coeffs_to_array(coeffs)
            assert yp.sumb(yp.abs(yp.changeBackend(W * self.x, 'numpy') - x_wavelet)) < eps, "Difference %.6e"

            # Check inverse operation
            coeffs_from_arr = pywt.array_to_coeffs(x_wavelet, coeff_slices)
            cam_recon = pywt.waverecn(coeffs_from_arr, wavelet=wavelet_test)
            assert yp.sumb(yp.abs(W.H * W * self.x - self.x)) < 1e-2

            # Ensure that the wavelet transform isn't just identity (weird bug)
            if W.shape[1] is yp.size(self.x):
                assert yp.sumb(yp.abs(W * yp.vec(self.x) - yp.vec(self.x))) > 1e-2, "%s" % wavelet_test

    def test_operator_fourier_transform(self):
        # Define "true" FFTs
        Ft = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=(0, 1)), axes=(0, 1), norm='ortho'), axes=(0, 1))
        iFt = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(x, axes=(0, 1)), axes=(0, 1), norm='ortho'), axes=(0, 1))

        eps_fft = yp.precision(self.x, for_sum=True)

        if global_backend == 'numpy':
            fft_backends = ['scipy', 'numpy']
        else:
            fft_backends = ['af']

        for fft_backend in fft_backends:

            # Create Operator
            F = ops.FourierTransform(image_size, dtype=global_dtype, axes=(0, 1), fft_backend=fft_backend, center=True, backend=global_backend)

            # Check forward model
            assert yp.sumb(yp.abs(Ft(self.x).reshape(image_size) - yp.changeBackend(F * self.x, 'numpy').reshape(image_size))) < eps_fft, '%f' % yp.sumb(yp.abs(Ft(x).reshape(image_size) - yp.changeBackend(F * vec(self.x), 'numpy').reshape(image_size)))
            assert yp.sumb(yp.abs(iFt(self.x).reshape(image_size) - yp.changeBackend((F.H * self.x), 'numpy').reshape(image_size))) < eps_fft

            # Check reciprocity
            assert yp.sumb(yp.abs(F * F.H * self.x - self.x)) < eps_fft, "%.4e" % yp.sumb(yp.abs(F * F.H * vec(self.x) - vec(self.x)))

            # Check Gradient
            F.gradient_check()

    def test_operator_convolution_circular(self):
        ''' Circular Convolution Operator '''
        # Generate circular convolution operator
        C = ops.Convolution(self.h)

        # Test forward operator
        conv2 = lambda x, h: yp.changeBackend(np.fft.ifftshift((np.fft.ifft2(np.fft.fft2(x, axes=(0,1), norm='ortho') * np.fft.fft2(h, axes=(0,1), norm='ortho'), axes=(0,1), norm='ortho')), axes=(0,1)).astype(yp.getNativeDatatype(global_dtype, 'numpy')), global_backend)

        x_np = yp.changeBackend(self.x, 'numpy')
        h_np = yp.changeBackend(self.h, 'numpy')

        assert np.sum(np.abs(yp.reshape(C * yp.vec(self.x), image_size) - conv2(x_np, h_np)) ** 2) < 1e-3, \
            'SSE (%.4e) is greater than tolerance (%.4e)' % ((np.sum(np.abs((C * yp.vec(self.x)).reshape(image_size)-conv2(x,h)))), eps)

        # Check gradient
        C.gradient_check()

    def test_operator_crosscorrelation(self):
        ''' Cross-Correlation Operator '''
        XC = ops.CrossCorrelation(self.h)

        # Check gradient
        XC.gradient_check()

    def test_operator_crop(self):
        ''' Crop Operator '''
        # Generate Crop Operator
        crop_size = (image_size[0] // 2, image_size[1] // 2)
        CR = ops.Crop(image_size, crop_size, pad_value=0,  dtype=global_dtype, backend=global_backend)

        # Check forward operator
        crop_start = tuple(np.asarray(image_size) // 2 - np.asarray(crop_size) // 2)
        y_1 = yp.changeBackend(CR * self.x, 'numpy')
        y_2 = yp.changeBackend(yp.crop(self.x, crop_size, crop_start), 'numpy')
        assert yp.sumb(yp.abs(y_1 - y_2)) < eps

        # Check Adjoint Operator
        pad_size = [int((image_size[i] - crop_size[i]) / 2) for i in range(len(image_size))]
        y_3 = yp.pad(yp.crop(self.x, crop_size, crop_start), image_size, crop_start, pad_value=0)
        y_4 = CR.H * CR * self.x
        assert yp.sumb(yp.abs(y_3 - y_4)) < eps

        # Check gradient
        CR.gradient_check()

    def test_operator_crop_non_centered(self):
        ''' Non-centered Crop Operator '''
        # Generate Crop Operator
        crop_size = (image_size[0] // 2, image_size[1] // 2)
        crop_start = (6, 6)
        CR = ops.Crop(image_size, crop_size, pad_value=0,  dtype=global_dtype, backend=global_backend, crop_start=crop_start)

        # Check forward operator
        y_1 = yp.changeBackend(CR * self.x, 'numpy')
        y_2 = yp.changeBackend(yp.crop(self.x, crop_size, crop_start), 'numpy')
        assert yp.sumb(yp.abs(y_1 - y_2)) < eps

        # Check Adjoint Operator
        pad_size = [int((image_size[i] - crop_size[i]) / 2) for i in range(len(image_size))]
        y_3 = yp.pad(yp.crop(self.x, crop_size, crop_start), image_size, crop_start, pad_value=0)
        y_4 = yp.reshape(CR.H * CR * self.x, image_size)
        assert yp.sumb(yp.abs(y_3 - y_4)) < eps

        # Check gradient
        CR.gradient_check()

    def test_operator_shift(self):
        ''' Shift Operator '''
        # Normal shift
        shift = (0, 10) # should be y, x
        T = ops.Shift(image_size, shift)

        def shift_func(x, shift):
            x = yp.changeBackend(self.x, 'numpy')
            for ax, sh in enumerate(shift):
                x = np.roll(self.x, int(sh), axis=ax)
            return(x)

        # Check Forward Operator
        y_1 = yp.changeBackend(T * self.x, 'numpy')
        y_2 = shift_func(yp.changeBackend(self.x, 'numpy'), shift)
        assert yp.sumb(yp.abs(y_1 - y_2)) < eps

        # Check Adjoint Operator
        assert yp.sumb(yp.abs(T.H * T * self.x - self.x)) < eps

        # Check gradient
        T.gradient_check()

    def test_operator_sum(self):
        ''' Element-wise Sum Operator '''
        axis_to_sum = (0,1)
        Σ = ops.Sum(image_size, axes=axis_to_sum)

        # Check forward operator
        y_1 = yp.changeBackend(Σ * self.x, 'numpy')
        y_2 = yp.sumb(yp.changeBackend(self.x, 'numpy'), axes=axis_to_sum)
        assert yp.abs(yp.sumb(y_1 - y_2)) < eps

        # Check adjoint operator
        y_3 = yp.changeBackend(Σ.H * Σ * self.x, 'numpy')
        reps = [1, ] * len(image_size)
        axes = list(range(len(image_size))) if axis_to_sum is 'all' else axis_to_sum
        scale = 1
        for axis in axes:
            reps[axis] = image_size[axis]
            scale *= 1 / image_size[axis]
        y_4 = yp.tile(y_2, reps) * scale
        assert yp.sumb(yp.abs(y_3 - y_4)) < eps

        # Check gradient
        Σ.gradient_check(eps=1)

    def test_operator_flip(self):
        ''' Flip Operator '''

        flip_axis = 0
        L = ops.Flip(image_size, axis=flip_axis)

        # Check forward operator
        assert yp.sumb(yp.abs(L * self.x - yp.flip(self.x, flip_axis))) < eps, "%f" % yp.sumb(yp.abs(L * self.x - vec(yp.flip(self.x, flip_axis))))

        # Check gradient
        L.gradient_check()

    def test_operator_norm_l2(self):
        L2 = ops.L2Norm(image_size[0] * image_size[1], dtype=global_dtype, backend=global_backend)

        # Check forward operator
        assert np.sum(np.abs(L2 * yp.vec(self.x) - 0.5 * yp.norm(yp.vec(self.x)) ** 2)) < eps, '%f' % np.sum(np.abs(L2 * yp.vec(self.x) - 0.5 * yp.norm(yp.vec(self.x)) ** 2))

        # Check gradient
        L2.gradient_check()

    def test_operator_norm_l1(self):
        L1 = ops.L1Norm(image_size[0] * image_size[1], dtype=global_dtype)

        # Forward operator
        assert np.sum(np.abs(L1 * yp.vec(self.x) - np.sum(np.abs(yp.vec(self.x))))) < eps

    def test_operator_phase_ramp(self):
        eps_phase_ramp = 1e-4
        shift = yp.changeBackend(np.asarray((-5,3)).astype(yp.getNativeDatatype(global_dtype, 'numpy')), global_backend)

        # Generate phase ramp
        R = ops.PhaseRamp(image_size)
        r = R * shift

        F = ops.FourierTransform(image_size, dtype=global_dtype, normalize=False, backend=global_backend)
        D_R = ops.Diagonalize(r, dtype=global_dtype)
        S_R = F.H * D_R * F

        # Pixel-wise shift operator
        S = ops.Shift(image_size, shift)

        # Check that phase ramp is shifting by correct amount
        assert yp.sumb(yp.abs(yp.changeBackend(S_R * self.x, 'numpy') - yp.changeBackend(S * self.x, 'numpy'))) < 1e-3

        # Check gradient of phase ramp convolution
        S_R.gradient_check()

        # Check gradient of phase ramp
        R.gradient_check(eps=1e-1)

    def test_intensity(self):
        I = ops.Intensity(image_size)

        # Check forward operator
        assert yp.sumb(yp.abs((yp.abs(yp.changeBackend(self.x, 'numpy')) ** 2) - yp.changeBackend(I * self.x, 'numpy'))) < eps

        # Check gradient
        I.gradient_check()

    def test_operator_exponential(self):
        L2 = ops.L2Norm(image_size)
        F = ops.FourierTransform(image_size)
        EXP = ops.Exponential(image_size)

        # Forward model
        assert yp.sumb(yp.abs(yp.changeBackend(EXP * self.x, 'numpy') - np.exp(yp.changeBackend(self.x, 'numpy')))) < eps

        # Check gradient
        EXP.gradient_check()

        # Generate composite operator
        D = ops.Diagonalize(self.h)
        L2 = ops.L2Norm(image_size)

        EXP_COMP = L2 * F * EXP
        EXP_COMP.gradient_check()

        EXP_COMP_2 = L2 * F * EXP * D
        EXP_COMP_2.gradient_check()

    def test_operator_matrix_multiply(self):
        matrix_size = (10,10)
        m = yp.rand(matrix_size, global_dtype, global_backend)
        xm = yp.rand(matrix_size[1], global_dtype, global_backend)
        M = ops.MatrixMultiply(m)

        # Check Forward operator
        assert yp.sumb(yp.abs(yp.vec(yp.changeBackend(M * xm, 'numpy')) - yp.vec(yp.changeBackend(m, 'numpy').dot(yp.changeBackend(xm, 'numpy'))))) < eps, "%f" % yp.sumb(yp.abs(yp.changeBackend(M * xm, 'numpy') - yp.changeBackend(m, 'numpy').dot(yp.changeBackend(xm, 'numpy'))[:, np.newaxis]))

        # Check Adjoint
        assert yp.sumb(yp.abs(yp.vec(yp.changeBackend(M.H * xm, 'numpy')) - yp.vec(np.conj(yp.changeBackend(m, 'numpy').T).dot(yp.changeBackend(xm, 'numpy'))))) < eps, "%f" % yp.sumb(yp.abs(yp.changeBackend(M.H * xm, 'numpy') - np.conj(yp.changeBackend(m, 'numpy').T).dot(yp.changeBackend(xm, 'numpy'))[:, np.newaxis]))

        # Check gradient
        M.gradient_check()

    def test_operator_stacking_linear(self):
        # Create list of operators
        op_list_linear = [
            ops.FourierTransform(image_size, dtype=global_dtype, backend=global_backend),
            ops.Identity(image_size, dtype=global_dtype, backend=global_backend),
            ops.Exponential(image_size, dtype=global_dtype, backend=global_backend)
        ]

        # Horizontally stacked operators
        H_l = ops.Hstack(op_list_linear)

        # Vertically stack x for forward operator
        x_np = yp.changeBackend(self.x, 'numpy')
        x3 = yp.changeBackend(np.vstack((x_np, x_np, x_np)), global_backend)

        # Check forward operation
        y2 = yp.zeros(op_list_linear[0].N, op_list_linear[0].dtype, op_list_linear[0].backend)

        for op in op_list_linear:
            y2 = y2 + op * self.x

        assert yp.sumb(yp.abs(yp.changeBackend(H_l(x3) - y2, 'numpy'))) < eps, "%.4e" % yp.sumb(yp.abs(H_l(x3) - y2))

        # Check gradient
        H_l.gradient_check()

        # Create vertically stacked operator
        V_l = ops.Vstack(op_list_linear)

        # Check forward operator
        y3 = np.empty((0,image_size[1]), dtype=yp.getNativeDatatype(global_dtype, 'numpy'))
        for index, op in enumerate(op_list_linear):
            y3 = np.append(y3, (op * self.x), axis=0)

        y3 = yp.changeBackend(y3, global_backend)
        assert yp.sumb(yp.abs(V_l * self.x - y3)) < eps, "%.4e" % yp.sumb(yp.abs(V_l * vec(x) - y3))

        # Check gradient
        V_l.gradient_check()

    def test_operator_stacking_nonlinear(self):
        # Create list of operators
        op_list_nonlinear = [
                    ops.FourierTransform(image_size),
                    ops.Identity(image_size),
                    ops.Exponential(image_size)
        ]
        # Horizontally stacked operators
        H_nl = ops.Hstack(op_list_nonlinear)

        # Vertically stack x for forward operator
        x3 = yp.changeBackend(np.vstack((self.x, self.x, self.x)), global_backend)

        # Check forward operation
        y2 = yp.zeros(op_list_nonlinear[0].shape[0], op_list_nonlinear[0].dtype, op_list_nonlinear[0].backend)
        for op in op_list_nonlinear:
            y2 = y2 + yp.vec(op * self.x)

        assert yp.sumb(yp.abs(yp.vec(H_nl(x3)) - y2)) < eps, "%.4e" % yp.sumb(yp.abs(yp.vec(H_nl(x3)) - y2))

        # Check gradient
        H_nl.gradient_check()

        # Create vertically stacked operator
        V_nl = ops.Vstack(op_list_nonlinear)

        # Check forward operator
        y3 = np.empty((0,image_size[1]), dtype=yp.getNativeDatatype(global_dtype, 'numpy'))
        for index, op in enumerate(op_list_nonlinear):
            y3 = np.append(y3, (op * self.x), axis=0)

        y3 = yp.changeBackend(y3, global_backend)

        assert yp.sumb(yp.abs(V_nl * self.x - y3)) < eps, "%.4e" % yp.sumb(yp.abs(V_nl * self.x - y3))

        # Check gradient
        V_nl.gradient_check()

    # def test_operator_blurkernelbasis(self):
    #     n_basis_splines = [6,5]
    #     illums = np.random.uniform(size=n_illum)
    #     spl_y = blurkernel.get_basis_splines(n_illum, n_basis_splines[0])
    #     spl_x = blurkernel.get_basis_splines(n_illum, n_basis_splines[1])
    #     B = ops.BlurKernelBasis(image_size, (spl_y, spl_x), illums, dtype=global_dtype, backend=global_backend)
    #     B.gradient_check()

import numpy as np
import scipy as sp
import scipy.misc as misc
import imageio
import math
from numpy.testing import assert_array_almost_equal
import pytest

# Libwallerlab imports
import llops as bops

def checkarrayfire():
    try:
        import arrayfire
        return True
    except:
        return False

eps = 1e-3
shape = (10,10)
dtype = 'complex32'

object_file_name = './brain_to_scan.png'   # Image to use when generating object
object_color_channel = 2                                                # Color channel to use when generating object
image_size = np.array([32, 64])                                         # Image size to simulate
h_size = np.array([4, 4])                                               # Convolution kernel size

@pytest.mark.skipif(not checkarrayfire() ,reason="arrayfire module could not be found/loaded")
class Test:
    def setup_method(self, test_method):
        # Initialize arrays

        self.eps = bops.precision(dtype) * np.prod(shape)
        self.x_np_ones = bops.ones(shape, dtype, 'numpy')
        self.x_ocl_ones = bops.ones(shape, dtype, 'arrayfire')

        self.x_ocl_randn = bops.randn(shape, dtype, 'arrayfire')
        self.x_np_randn = np.asarray(self.x_ocl_randn)

        self.x_ocl_randu = bops.randu(shape, dtype, 'arrayfire')
        self.x_np_randu = np.asarray(self.x_ocl_randu)

        # Create random matricies
        self.A_ocl = bops.randn((10,10), backend='arrayfire')
        self.A_np = np.asarray(self.A_ocl)

        self.sz = (10,20)
        self.x_np_rect = bops.randn(self.sz, backend='numpy')
        assert bops.shape(self.x_np_rect) == self.sz

        self.x_ocl_rect = bops.randn(self.sz, backend='arrayfire')
        assert bops.shape(self.x_ocl_rect) == self.sz

        # Load object and crop to size
        brain = imageio.imread(object_file_name)
        x_0 = sp.misc.imresize(brain, size=image_size)[:, :, object_color_channel].astype(bops.getNativeDatatype(dtype, 'numpy')) / 255.

        # Convert object to desired backend
        self.x = bops.changeBackend(x_0, 'arrayfire')

    def test_sign(self):
        a = bops.sign(self.x_ocl_ones)
        b = bops.sign(self.x_np_ones)
        assert np.sum(np.abs(bops.sign(self.x_np_ones) - np.asarray(bops.sign(self.x_ocl_ones)))) < 1e-4

    def test_base(self):
        assert np.abs(bops.norm(self.x_np_ones) - bops.norm(self.x_ocl_ones)) < 1e-8
        assert np.abs(np.linalg.norm(self.x_np_ones) - bops.norm(self.x_ocl_ones)) < 1e-8
        assert np.sum(np.abs(bops.angle(self.x_np_randn) - np.asarray(bops.angle(self.x_ocl_randn)))) < 1e-4
        assert np.sum(np.abs(bops.abs(self.x_np_randn) - np.asarray(bops.abs(self.x_ocl_randn)))) < 1e-4
        assert np.sum(np.abs(bops.exp(self.x_np_randn) - np.asarray(bops.exp(self.x_ocl_randn)))) < 1e-4
        assert np.sum(np.abs(bops.conj(self.x_np_randn) - np.asarray(bops.conj(self.x_ocl_randn)))) < 1e-4
        assert np.sum(np.abs(bops.flip(self.x_np_randn) - np.asarray(bops.flip(self.x_ocl_randn)))) < 1e-4
        assert np.sum(np.abs(bops.transpose(self.x_np_randn) - np.asarray(bops.transpose(self.x_ocl_randn)))) < 1e-4

        assert np.sum(np.abs(self.A_np - np.asarray(self.A_ocl))) < 1e-4
        assert np.sum(np.abs(self.x_np_randn - np.asarray(self.x_ocl_randn))) < 1e-4

    def test_roll(self):
        assert np.sum(np.abs(bops.roll(self.x_np_randn,-10,1) - np.asarray(bops.roll(self.x_ocl_randn,-10,1)))) < 1e-4
        assert np.sum(np.abs(bops.roll(self.x_np_randn,5,0) - np.asarray(bops.roll(self.x_ocl_randn,5,0)))) < 1e-4
        assert np.sum(np.abs(bops.fftshift(self.x_np_randn) - np.asarray(bops.fftshift(self.x_ocl_randn)))) < 1e-4

    def test_reshape(self):
        new_shape = (shape[0] // 2, shape[1] * 2)
        assert np.sum(np.abs(bops.reshape(self.x_np_randn, new_shape) - np.asarray(bops.reshape(self.x_ocl_randn, new_shape)))) < 1e-4

#     def test_sums(self):
#         assert np.sum(np.abs(bops.sumb(self.x_np_randn) - np.asarray(bops.sumb(self.x_ocl_randn)))) < 1e-4
#         assert np.sum(np.abs(bops.sumb(self.x_np_randn, 0) - np.asarray(bops.sumb(self.x_ocl_randn, 0)))) < 1e-4

    def test_argmin(self):
        x = yp.zeros(shape, dtype, backend)
        x[10,5] = 1.0
        x[2,3] = -1.0

        assert yp.argmax(yp.asbackend(x, 'numpy')) == yp.argmax(yp.asbackend(x, 'arrayfire'))
        assert yp.argmin(yp.asbackend(x, 'numpy')) == yp.argmin(yp.asbackend(x, 'arrayfire'))

    def test_fill(self):
        x_ocl_tofill = bops.randn(shape, dtype, 'arrayfire')
        x_np_tofill = np.asarray(x_ocl_tofill)
        fill_value = 10
        bops.fill(x_np_tofill, fill_value)
        bops.fill(x_ocl_tofill, fill_value)
        assert np.sum(np.abs(np.asarray(x_ocl_tofill) - fill_value)) < 1e-4
        assert np.sum(np.abs(np.asarray(x_np_tofill) - fill_value)) < 1e-4

    def test_max_min(self):
        assert np.abs(bops.amax(self.x_np_randn) - np.asarray(bops.amax(self.x_ocl_randn))) < 1e-4
        assert np.abs(bops.amin(self.x_np_randn) - np.asarray(bops.amin(self.x_ocl_randn))) < 1e-4

    def test_matrix_multiply(self):
        assert np.sum(np.abs(bops.matmul(self.A_np, self.x_np_randn) - np.asarray(bops.matmul(self.A_ocl, self.x_ocl_randn)))) < 1e-4

    def test_tile(self):
        assert np.sum(np.abs(np.asarray(bops.tile(self.x_ocl_randn, (2,2))) - bops.tile(self.x_np_randn, (2,2)))) < 1e-4

    def test_vec(self):
        assert np.sum(np.abs(np.asarray(bops.vectorize(self.A_ocl))- bops.vectorize(np.asarray(self.A_ocl)))) < 1e-4

    def test_pad(self):
        pad_size = [self.x_np_randn.shape[i] + 10 for i in range(len(shape))]
        assert np.sum(np.abs(bops.pad(self.x_np_randn, pad_size, crop_start=(0,0)) - np.asarray(bops.pad(self.x_ocl_randn,pad_size, crop_start=(0,0))))) < 1e-4
        assert np.sum(np.abs(bops.pad(self.x_np_randn, pad_size, crop_start=(5,2)) - np.asarray(bops.pad(self.x_ocl_randn,pad_size, crop_start=(5,2))))) < 1e-4
        assert np.sum(np.abs(bops.pad(self.x_np_randn, pad_size, crop_start=(1,3), pad_value=4) - np.asarray(bops.pad(self.x_ocl_randn,  pad_size, crop_start=(1,3), pad_value=4)))) < 1e-4

    def test_crop(self):
        crop_size = [self.x_np_randn.shape[i] - 2 for i in range(len(shape))]
        assert np.sum(np.abs(bops.crop(self.x_np_randn, crop_size, crop_start=(0,0)) - np.asarray(bops.crop(self.x_ocl_randn, crop_size, crop_start=(0,0))))) < 1e-4

        x_full_np = bops.rand((20,20), dtype=dtype, backend='numpy')
        x_full_ocl = bops.changeBackend(x_full_np, 'arrayfire')
        crop_size = tuple(np.asarray(bops.shape(x_full_np)) // 2)
        crop_start = (2,2)

        x_crop_np = bops.crop(x_full_np, crop_size, crop_start=crop_start)
        x_crop_ocl = bops.crop(x_full_ocl, crop_size, crop_start=crop_start)

        assert np.sum(np.abs(bops.changeBackend(x_crop_ocl, 'numpy') - x_crop_np)) < 1e-4

    def test_indexing(self):
        q = bops.randn((10,10), backend='numpy', dtype='complex32')
        q_ocl = bops.changeBackend(q, 'arrayfire')
        q_ocl_np = bops.changeBackend(q_ocl, 'numpy')
        assert np.sum(np.abs(q - q_ocl_np)) < eps

        m = bops.rand((10,20), dtype, "numpy")
        m_ocl = bops.changeBackend(m, 'arrayfire')

        assert abs(m[0,1] - bops.scalar(m_ocl[0,1])) < eps
        assert abs(m[1,1] - bops.scalar(m_ocl[1,1])) < eps
        assert abs(bops.scalar(m_ocl[4,1]) - m[4,1]) < eps

        assert bops.scalar(self.x_np_rect[5,15]) == bops.scalar(bops.changeBackend(self.x_np_rect, 'arrayfire')[5,15])
        assert bops.scalar(self.x_ocl_rect[5,15]) == bops.scalar(bops.changeBackend(self.x_ocl_rect, 'numpy')[5,15])

    def test_shape(self):
        # Check that shape function is correct

        assert bops.shape(bops.changeBackend(self.x_ocl_rect, 'numpy')) == self.sz
        assert bops.shape(bops.changeBackend(self.x_np_rect, 'arrayfire')) == self.sz

    def test_matmul(self):
        matrix_size = (10,20)
        m = bops.rand(matrix_size, dtype, 'arrayfire')
        xm = bops.rand(matrix_size[1], dtype, 'arrayfire')
        assert np.sum(np.abs(bops.changeBackend(bops.matmul(m, xm), 'numpy') - bops.changeBackend(m, 'numpy').dot(bops.changeBackend(xm, 'numpy')))) < eps

        # Check matrix multiply (numpy to arrayfire)
        m_np = bops.rand(matrix_size, dtype, 'numpy')
        xm_np = bops.rand(matrix_size[1], dtype, 'numpy')
        m_ocl = bops.changeBackend(m_np, 'arrayfire')
        xm_ocl = bops.changeBackend(xm_np, 'arrayfire')
        assert np.sum(np.abs(bops.changeBackend(bops.matmul(m_ocl, xm_ocl), 'numpy') - m_np.dot(xm_np))) < eps
import ndoperators as ops
# from ndoperators import Operator
import numpy as np
import llops as yp
from llops import real, min, max, abs, numpy_function
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, unsupervised_wiener,
                                 denoise_tv_bregman, denoise_nl_means)
from skimage.filters import median as skmedian
# import prox_tv

import functools

# TODO: several of these functions have arguments that
# may speed up their computation?

@numpy_function
def tv(image, **kwargs):
    # send arg multichannel, weight
    filtered = prox_tv.tv1_2d(np.asarray(image), kwargs.pop('weight', 0.3), n_threads=8)
    return _check_validity(image, filtered)

@numpy_function
def tv_bregman(image, **kwargs):
    # send arg weight
    filtered = yp.filter._complex_filter(image, denoise_tv_bregman, dtype='float64', **kwargs)
    return _check_validity(image, filtered)


@numpy_function
def bilateral(image, **kwargs):
    filtered = denoise_bilateral(np.real(image).astype(np.double), multichannel=False)
    return _check_validity(image, filtered)

@numpy_function
def wavelet(image, **kwargs):
    # send arg multichannel
    filtered = yp.filter._complex_filter(image, denoise_wavelet, **kwargs)
    return _check_validity(image, filtered)

@numpy_function
def median_denoise(image, **kwargs):
    # optional size of patch
    switch = kwargs.pop('sk', False)
    if switch:
        # filtered = yp.filter._complex_filter(image, skmedian, **kwargs)
        filtered = skmedian(np.real(image), **kwargs)
    else:
        filtered = yp.filter.median(image, **kwargs)
    return _check_validity(image, filtered)

@numpy_function
def weiner(image, **kwargs):
    # doesn't really make sense
    if 'psf' not in kwargs:
        psf = yp.ones((5, 5)) / 25
    else:
        psf = kwargs['psf']
    filtered, _ = unsupervised_wiener(image, psf, is_real=False)
    return _check_validity(image, filtered)

@numpy_function
def nl_means(image, **kwargs):
    # optional arg patch size, fast_mode=True, still pretty slow
    filtered = yp.filter._complex_filter(image, denoise_nl_means, **kwargs)
    return _check_validity(image, filtered)

def _check_validity(image, filtered):
    if yp.any(yp.isnan(filtered)) and not yp.any(yp.isnan(image)):
        return image
    else:
        return filtered

'''
Copyright 2017 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Maximum matrix size
MAX_FULL_MATRIX_ELEMENT_SIZE = 1e5

# Warn if calling an expensive operation that can most likely be avoided
WARN_FOR_EXPENSIVE_OPERATIONS = False

# This file contains configuration infromation, and is imported by every file in the operators library
def init():
    # Valid datatypes
    global valid_dtypes
    valid_dtypes = ('complex32', 'complex64', 'int32', 'int64', 'float32', 'float64')

    # Default backend
    global default_backend
    default_backend = 'numpy'

    # Default datatype
    global default_dtype
    default_dtype = 'float32'

    # Defualt FFT Backend
    global default_fft_backend
    default_fft_backend = 'scipy'

    # Default arrayfire backend
    global default_arrayfire_backend
    default_arrayfire_backend = None

    # Determine valid backends based on imports
    global valid_backends
    valid_backends = ['numpy']
    try:
        import arrayfire
        valid_backends.append('arrayfire')

        # Set default arrayfire backend
        af_backends = arrayfire.get_available_backends()
        if 'cuda' in af_backends:
            default_arrayfire_backend = 'cuda'
        elif 'opencl' in af_backends:
            default_arrayfire_backend = 'opencl'
        else:
            default_arrayfire_backend = 'cpu'

        # set default backend
        arrayfire.set_backend(default_arrayfire_backend)
    except:
        pass

    # Determine valid fft backends based on imports
    global valid_fft_backends
    valid_fft_backends = ['numpy', 'scipy']
    try:
        import pyfftw
        valid_fft_backends.append('fftw')
        default_fft_backend = 'fftw'
    except:
        pass

def setDefaultBackend(new_backend):
    ''' Set default backend for all new operators '''
    global default_backend
    if new_backend in valid_backends:
        default_backend = new_backend
    else:
        raise ValueError("Backend %s is not available. Valid backends: %s" % (new_backend, valid_backends))

def setDefaultDatatype(new_dtype):
    ''' Set default datatype for all new operators '''
    global default_dtype
    if new_dtype in valid_dtypes:
        default_dtype = new_dtype
    else:
        raise ValueError("Backend %s is not available. Valid backends: %s" % (new_dtype, valid_dtypes))

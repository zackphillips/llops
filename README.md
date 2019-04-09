# Low-level Operators library

This library enables backend-independent computing using numpy and arrayfire.

## Supported Backends

-   `numpy`
-   `arrayfire` (which interfaces with OpenCL and CUDA, in addition to their own CPU implementation)

These backend labels are used as strings to define data backends across the library.

## Supported Datatypes

-   Integers (`uint32`, `uint64`, `int32`, `int64`)
-   Floating point (`float32`, `float64`, `complex32`, `complex64`)

These datatypes labels are used as strings to define data types across the library.

**Note that many GPUs do not support double-precision datatypes.**

**Complex number datatypes are defined as the precision of the real part (unlike numpy). For instance, np.complex64 and np.complex128 data will be known to this library as `complex32` and `complex64` respectively.**

# Install

## Arrayfire install

To install arrayfire on OSX follow these instructions:

1. Download and install the following two commands using homebrew:
```bash
brew install glfw
brew install fontconfig
```
2. Download and run the (https://arrayfire.com/)[installer].

3. Add the arrayfire lib to the system path
```bash
export DYLD_LIBRARY_PATH=/opt/arrayfire/lib:$DYLD_LIBRARY_PATH
```
You can also append this to `~/.bash_profile` or `~/.zshrc` to make it persistent if you're using bash or zsh, respectively.

4. Test that the arrayfire examples will compile and run:
```bash
cp -r /opt/arrayfire/share/ArrayFire/examples/helloworld /tmp
cd /tmp
clang++ -I/opt/arrayfire/include -L/opt/arrayfire/lib -lafcpu helloworld.cpp -o hello_cpu && ./hello_cpu
clang++ -I/opt/arrayfire/include -L/opt/arrayfire/lib -lafopencl helloworld.cpp -o hello_opencl && ./hello_opencl
echo "It worked!"
```

5. Install arrayfire-python
```bash
pip install arrayFire
```

## pyfftw install
To install pyfftw using macports, use the following:
```
sudo port install fftw
sudo port install fftw-3 fftw-3-long fftw-3-single

export LDFLAGS="-L/opt/local/lib"
export CFLAGS="-I/opt/local/include"
export DYLD_LIBRARY_PATH=/opt/local/lib
export AF_PATH=/opt/arrayfire
export AF_VERBOSE_LOADS=1
pip install pyfftw

```

To install pyfftw using homebrew, use the following:
```
brew install fftw

# Remove libpng (so pyfftw can link against native OSX libpng)
brew uninstall --ignore-dependencies libpng

# Define linker constants
export LDFLAGS="-L/usr/local/lib"
export CFLAGS="-I/usr/local/include"
export DYLD_LIBRARY_PATH=/usr/local/lib

# install missing dependency
pip install cython

# Install module
pip install pyfftw

# Reinstall libpng
brew install libpng

```
Macports is preferred because is should be more robust to Apple's changes to built-in libraries, but either will work.


# License

BSD

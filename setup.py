from setuptools import setup, find_packages

setup(name='llops',
      version='0.1',
      description='Low-level Operators with a backend selector',
      license='BSD',
      packages=find_packages(),
      py_modules=['llops'],
      install_requires=['numpy', 'psutil', 'matplotlib', 'scipy', 'scikit-image', 'imageio', 'pillow'])

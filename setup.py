from setuptools import setup, find_packages

setup(name='llops',
      author='Zack Phillios',
      author_email='zkphil@gmail.com',
      version='0.1',
      description='Low-level Operators with a backend selector',
      license='BSD',
      packages=find_packages(),
      py_modules=['llops'],
      package_data={'': ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.json']},
      install_requires=['numpy', 'psutil', 'matplotlib', 'scipy', 'scikit-image', 'imageio', 'pillow', 'pywavelets'])

from setuptools import setup, find_packages


setup(
    name='TensorFox',
    version='1.0',
    license='GNU',
    author="Felipe Bottega Diniz",
    author_email='felipebottega@gmail.com',
    packages=find_packages('modules'),
    package_dir={'': 'modules'},
    url='https://github.com/felipebottega/Tensor-Fox',
    keywords='Tensor Fox',
    install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'sklearn',
          'matplotlib',
          'numba',
          'sparse_dot_mkl',
      ],

)
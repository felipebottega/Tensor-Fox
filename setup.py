from setuptools import setup, find_packages
import pathlib


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='TensorFox',
    version='1.0.12',
    description='Tensor Fox is a high performance package of multilinear algebra and tensor routines, with focus on the Canonical Polyadic Decomposition (CPD), also called PARAFAC or CANDECOMP.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='GNU',
    author="Felipe Bottega Diniz",
    author_email='felipebottega@gmail.com',
    packages=find_packages('modules'),
    package_dir={'': 'modules'},
    url='https://github.com/felipebottega/Tensor-Fox',
    keywords='Tensor Fox Canonical Polyadic Decomposition CPD PARAFAC CANDECOMP Multilinear Learning',
    install_requires=[
          'numpy>=1.21.0',
          'pandas>=1.2.3',
          'scipy>=1.6.2',
          'scikit-learn>=0.24.1',
          'matplotlib>=3.3.4',
          'numba>=0.53.1',
          'IPython>=7.31.1',
          'sparse_dot_mkl>=0.7',
      ],

)
from setuptools import setup, find_packages
import pathlib

# import pdb; pdb.set_trace()
# learned from https://github.com/pypa/sampleproject/
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='TensorFox',
    version='1.0.1',
    description='Python3 package for Multilinear Algebra and Tensor routines.',
    long_description=long_description
    long_description_content_type='text/markdown'
    license='GNU',
    author="Felipe Bottega Diniz",
    author_email='felipebottega@gmail.com',
    packages=find_packages('modules'),
    package_dir={'': 'modules'},
    url='https://github.com/felipebottega/Tensor-Fox',
    keywords='Tensor Fox CPD PARAFAC CANDECOMP',
    install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'sklearn',
          'matplotlib',
          'numba',
          'IPython',
          'sparse_dot_mkl>=0.7,<0.8',
      ],

)
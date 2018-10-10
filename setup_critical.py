from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "Critical",
        ["Critical.pyx"],
        extra_compile_args = ["-O3", "-march=native", "-fopenmp", "-fforce-addr", "-funroll-all-loops", "--param=max-unrolled-insns=20"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="Critical",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
    
)

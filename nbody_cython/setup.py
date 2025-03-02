from setuptools import setup
import os
from Cython.Build import cythonize
import numpy

os.environ["CC"] = "gcc"

setup(
    ext_modules=cythonize("cythonfn.pyx", compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)

from distutils.core import setup

from Cython.Build import cythonize

setup(
    name="ccc",
    ext_modules=cythonize("ccc.pyx"),
)

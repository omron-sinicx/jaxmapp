from distutils.core import setup

from Cython.Build import cythonize

setup(
    name="check_continuous_collision",
    ext_modules=cythonize("check_continuous_collision.pyx"),
)

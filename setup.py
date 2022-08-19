from Cython.Distutils import build_ext
from setuptools import Extension, setup

setup()
ext_modules = [
    Extension(
        "jaxmapp.check_continuous_collision",
        sources=["cython_helper/check_continuous_collision.pyx"],
        extra_compile_args=["-O3"],
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "quemb.molbe._cpp.mymodule",
        ["src/quemb/molbe/_cpp/mymodule.cpp"],
        include_dirs=["third-party/eigen-3.3.9"],  # Your Eigen headers path
        cxx_std=17,
        extra_compile_args=[
            "-O3",
            "-march=native",
            "-ffast-math",
            "-fopenmp",
        ],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

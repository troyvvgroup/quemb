from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "quemb.molbe._cpp.eri_sparse_DF",
        ["src/quemb/molbe/_cpp/eri_sparse_DF.cpp"],
        include_dirs=["external/eigen"],
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

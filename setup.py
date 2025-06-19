from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "quemb.cpp.mymodule",
        ["src/quemb/molbe/cpp/mymodule.cpp"],
        include_dirs=["third_party/eigen"],  # Your Eigen headers path
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

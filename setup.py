# Read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="quemb",
    version="1.0",
    description="QuEmb: A framework for efficient simulation of large molecules, surfaces, and solids via Bootstrap Embedding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oimeitei/quemb",
    license="Apache 2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.7.0",
        "pyscf>=2.0.0",
        "matplotlib",
        # TODO this temporarily points to mcocdawc/libdmet_preview.
        # as soon as this PR: https://github.com/gkclab/libdmet_preview/pull/21
        # is merged, let it point to the original libdmet
        # https://github.com/gkclab/libdmet_preview
        "libdmet @ git+https://github.com/mcocdawc/libdmet_preview.git@add_fixes_for_BE",
    ],
)

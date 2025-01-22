# Read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="quemb",
    version="1.0",
    description="QuEmb: A framework for efficient simulation of large molecules, "
    "surfaces, and solids via Bootstrap Embedding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oimeitei/quemb",
    license="Apache 2.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.22.0",
        "scipy>=1.7.0",
        "pyscf>=2.0.0",
        "networkx",
        "matplotlib",
        "libdmet @ git+https://github.com/gkclab/libdmet_preview.git",
        "attrs",
        "cattrs",
        "pyyaml",
    ],
)

# QuEmb

QuEmb is a robust framework designed to implement the Bootstrap Embedding (BE) method,
efficiently treating electron correlation in molecules, surfaces, and solids. This repository contains
the Python implementation of the BE methods, including periodic bootstrap embedding.
The code leverages [PySCF](https://github.com/pyscf/pyscf) library for quantum chemistry calculations and utlizes Python's
multiprocessing module to enable parallel computations in high-performance computing environments.

QuEmb includes two libraries: `molbe` and `kbe`.
The `molbe` library implements BE for molecules and supramolecular complexes,
while the `kbe` library is designed to handle periodic systems such as surfaces and solids using periodic BE.


## Features

- **Fragment-based quantum embedding:** Utilizes flexible system partioning with overlapping regions to
improve quantum embedding techniques.
- **Periodic Bootstrap Embedding:** Extends BE method to treat periodic systems (1D & 2D systems)
using reciprocal space sums.
- **High accuracy and efficiency:** Capable of recovering ~99.9% of electron correlation energy.
- **Parallel computing:** Employ's Python multiprocessing module to perform parallel computations across multiple
processors.

## Installation

### Prerequisites

- Python 3.6 or higher
- PySCF library
- Numpy
- Scipy
- libDMET (required for periodic BE)
- [Wannier90](https://github.com/wannier-developers/wannier90)<sup>&&</sup> (to use Wannier functions)

<sup>&&</sup>Wannier90 code is interfaced via [libDMET](https://github.com/gkclab/libdmet_preview) in QuEmb</sub>

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/oimeitei/quemb.git
   cd quemb

2. Install QuEmb using one of the following approaches:
    ```bash
    pip install .
    ```
    or simply add `path/to/quemd` to `PYTHONPATH`
    ```bash
    export PYTHONPATH=/path/to/quemb:$PYTHONPATH
    ```

    For conda (or virtual environment) installations, after creating your environment, specify the path to mol-be source as a path file, as in:
    ```bash
    echo path/to/quemb > $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")/quemb.pth
    ```


## Basic Usage

```bash
# Molecular
from quemb.molbe import fragpart
from quemb.molbe import BE

# Periodic
#from quemb.kbe import fragpart
#from quemb.kbe import BE

# Perform pyscf HF/KHF calculations
# get mol: pyscf.gto.M or pyscf.pbc.gto.Cell
# get mf: pyscf.scf.RHF or pyscf.pbc.KRHF

# Define fragments
myFrag = fragpart(be_type='be2', mol=mol)

# Initialize BE
mybe = BE(mf, myFrag)

# Perform density matching in BE
mybe.optimize(solver='CCSD')
```
See documentation and `quemb/example` for more details.

## Documentation

Comprehensive documentation for QuEmb is available at `quemb/docs`. The documentation provides detailed infomation on installation, usage, API reference, and examples. To build the documentation locally, simply navigate to `docs` and build using `make html` or `make latexpdf`.

Alternatively, you can view the latest documentation online [here](https://quemb.readthedocs.io/).

## References

The methods implemented in this code are described in details in the following papers:
- OR Meitei, T Van Voorhis, Periodic bootstrap embedding, [JCTC 19 3123 2023](https://doi.org/10.1021/acs.jctc.3c00069)
- OR Meitei, T Van Voorhis, Electron correlation in 2D periodic systems, [arXiv:2308.06185](https://arxiv.org/abs/2308.06185)
- HZ Ye, HK Tran, T Van Voorhis, Bootstrap embedding for large molecular systems, [JCTC 16 5035 2020](https://doi.org/10.1021/acs.jctc.0c00438)

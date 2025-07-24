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

- Python `3.10 <= version < 3.13`
- PySCF library
- Numpy
- Scipy
- [chemcoord](https://chemcoord.readthedocs.io/)  (required for fragmentation)
- [libDMET](https://github.com/gkclab/libdmet_preview) (required for periodic BE)
- [Wannier90](https://github.com/wannier-developers/wannier90)<sup>##</sup> (to use Wannier functions)

<sup>##</sup> `Wannier90` code is optional and only necessary to use Wannier functions in periodic code. </sub>

The required dependencies, with the exception of the optional `Wannier90`,
are automatically installed by `pip`.

### Installation

One can just `pip install` directly from the Github repository:
```bash
pip install git+https://github.com/troyvvgroup/quemb
```

Alternatively one can manually clone and install as in:
```bash
git clone --recurse-submodules https://github.com/troyvvgroup/quemb
cd quemb
pip install .
```



## Basic Usage

```bash
# Molecular
from quemb.molbe import fragmentate
from quemb.molbe import BE

# Periodic
#from quemb.kbe import fragmentate
#from quemb.kbe import BE

# Perform pyscf HF/KHF calculations
# get mol: pyscf.gto.M or pyscf.pbc.gto.Cell
# get mf: pyscf.scf.RHF or pyscf.pbc.KRHF

# Define fragments
myFrag = fragmentate(n_BE=2, mol=mol)

# Initialize BE
mybe = BE(mf, myFrag)

# Perform density matching in BE
mybe.optimize(solver='CCSD')
```
See documentation and `quemb/example` for more details.

## Documentation

Comprehensive documentation for QuEmb is available at `quemb/docs`. The documentation provides detailed infomation on installation, usage, API reference, and examples. To build the documentation locally, simply navigate to `docs` and build using `make html` or `make latexpdf`.

Alternatively, you can view the latest documentation online [here](https://vanvoorhisgroup.mit.edu/quemb/).

## References

This code has been described in a software paper: 
- M Cho, OR Meitei, LP Weisburn, O Weser, S Weatherly, et. al, QuEmb: a toolbox for bootstrap embedding calculations of molecular and periodic systems, [JPCA 129 6538 2025](https://doi.org/10.1021/acs.jpca.5c02983)

The methods implemented in this code are described in further detail in the following papers:
- HZ Ye, HK Tran, T Van Voorhis, Bootstrap embedding for large molecular systems, [JCTC 16 5035 2020](https://doi.org/10.1021/acs.jctc.0c00438)
- HK Tran, LP Weisburn, M Cho, S Weatherly, HZ Ye, T Van Voorhis, Bootstrap embedding for molecules in extended basis sets, [JCTC 20 10912 2024](https://doi.org/10.1021/acs.jctc.4c01267)
- OR Meitei, T Van Voorhis, Periodic bootstrap embedding, [JCTC 19 3123 2023](https://doi.org/10.1021/acs.jctc.3c00069)
- OR Meitei, T Van Voorhis, Electron correlation in 2D periodic systems from periodic bootstrap embedding, [JPC Lett. 15 11992 2024](https://doi.org/10.1021/acs.jpclett.4c02686)


## Contributors

The contributors in alphabetic order were:
- Alexandra Alexiu
- Minsik Cho
- Beck Hanscam
- Oinam Romesh Meitei
- Aleksandr Trofimov
- Oskar Weser
- Shaun Weatherly
- Leah Weisburn
- Hong-Zhou Ye

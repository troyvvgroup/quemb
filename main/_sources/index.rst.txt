.. QuEmb documentation master file, created by
   sphinx-quickstart on Sun Jul 28 08:42:03 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*****
QuEmb
*****

QuEmb is a robust framework designed to implement the Bootstrap Embedding (BE) method,
efficiently treating electron correlation in molecules, surfaces, and solids. This repository contains
the Python implementation of the BE methods, including periodic bootstrap embedding.
The code leverages `PySCF <https://github.com/pyscf/pyscf>`_ library for quantum chemistry calculations and utlizes Python's
multiprocessing module to enable parallel computations in high-performance computing environments.

QuEmb includes two libraries: ``quemb.molbe`` and ``quemb.kbe``.
The ``quemb.molbe`` library implements BE for molecules and supramolecular complexes,
while the ``quemb.kbe`` library is designed to handle periodic systems such as surfaces and solids using periodic BE.

References
==========

1. OR Meitei, T Van Voorhis, Periodic bootstrap embedding, `JCTC 19 3123 2023 <https://doi.org/10.1021/acs.jctc.3c00069>`_
2. OR Meitei, T Van Voorhis, Electron correlation in 2D periodic systems, `arXiv:2308.06185 <https://arxiv.org/abs/2308.06185>`_
3. HZ Ye, HK Tran, T Van Voorhis, Bootstrap embedding for large molecular systems, `JCTC 16 5035 2020 <https://doi.org/10.1021/acs.jctc.0c00438>`_


.. toctree::
   :maxdepth: 1

   install
   api_reference
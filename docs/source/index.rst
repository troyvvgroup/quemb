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


Table of Contents
=================

.. toctree::
   :maxdepth: 1

   install
   api_reference
   bibliography


References
==========

The whole software package was published in :cite:`cho_quemb_2025`.

The method is based on the following works:

- First work on BE for model systems :cite:`welborn_bootstrap_2016,ricke_performance_2017`
- Molecular BE :cite:`ye_bootstrap_2019,ye_atom-based_2019,ye_bootstrap_2020,ye_accurate_2021`
- Periodic BE :cite:`meitei_periodic_2023,meitei_electron_2024`
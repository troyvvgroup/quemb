Solver Routines
***************


Orbital Localization
====================


Density Matching Error
======================

.. autofunction:: quemb.molbe.solver.solve_error

Interface to Quantum Chemistry Methods
======================================

.. autofunction:: quemb.molbe.solver.solve_mp2

.. autofunction:: quemb.molbe.solver.solve_ccsd

.. autofunction:: quemb.molbe.helper.get_scfObj

Schmidt Decomposition
=====================

Molecular Schmidt decomposition
-------------------------------

.. autofunction:: quemb.molbe.solver.schmidt_decomposition

Periodic Schmidt decomposition
------------------------------

.. autofunction:: quemb.kbe.solver.schmidt_decomp_svd

Handling Hamiltonian
====================

.. autofunction:: quemb.molbe.helper.get_eri

.. autofunction:: quemb.molbe.helper.get_core


Build molecular HF potential
----------------------------

.. autofunction:: quemb.molbe.helper.get_veff

Build perioidic HF potential
----------------------------

.. autofunction:: quemb.kbe.helper.get_veff


Handling Energies
=================

.. autofunction:: quemb.molbe.helper.get_frag_energy
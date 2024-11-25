Basic Usage
***********

There are only three steps to run BE calculations using QuEmb:
 1. Define fragments using ``fragpart``.
 2. Initialize BE using ``BE``.
 3. Density matching or chemical potential optimization using ``BE.optimize``

QuEmb requires molecular & Hartree-Fock calculation objects from pyscf. Refer to pyscf documentations: Molecules: `1 <https://pyscf.org/user/gto.html>`_ & `2 <https://pyscf.org/user/scf.html>`_, Solids:  `3 <https://pyscf.org/user/pbcgto.html>`_ &  `4 <https://pyscf.org/user/pbc/scf.html>`_


Simple example of BE calculation on molecular system::

  from quemb.molbe import fragpart, BE

  # Perform pyscf calculations to get mol, mf objects
  # See quemb/example/molbe_h8_density_matching.py
  # get mol: pyscf.gto.M
  # get mf: pyscf.scf.RHF

  # Define fragments
  myFrag = fragpart(be_type='be2', mol=mol)

  # Initialize BE
  mybe = BE(mf, myFrag)

  # Perform density matching in BE
  mybe.optimize(solver='CCSD')


Simple example of periodic BE calculation on 1D periodic system::

  from quemb.kbe import fragpart, BE

  # Perform pyscf pbc calculations to get cell, kmf, kpts
  # See quemb/example/kbe_polyacetylene.py
  # get cell: pyscf.pbc.gto.Cell
  # get kmf: pyscf.pbc.scf.KRHF
  # get kpts: 2D array of k-points

  nk = 3 # no. of kpoints
  # Define fragments
  myFrag = fragpart(be_type='be2', mol=cell, kpt=[1,1,nk])

  # Initialize BE
  mybe = BE(mf, myFrag, kpts=kpts)

  # Perform density matching in BE
  mybe.optimize(solver='CCSD')


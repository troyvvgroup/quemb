Installation
************

Prerequisites
-------------

 * Python >=10 or higher
 * PySCF library
 * Numpy
 * Scipy
 * `libDMET <https://github.com/gkclab/libdmet_preview>`__ (required for periodic BE)
 * `Wannier90 <https://github.com/wannier-developers/wannier90>`_ :sup:`##`

| :sup:`##` :code:`Wannier90` code is optional and only necessary to use Wannier functions in periodic code.

The required dependencies, with the exception of the optional :code:`Wannier90`,
are automatically installed by :bash:`pip`.


Installation
-------------

One can just :bash:`pip install` directly from the Github repository

.. code-block:: bash

  pip install git+https://https://github.com/troyvvgroup/quemb



Alternatively one can manually clone and install as in

.. code-block:: bash

  git clone https://https://github.com/troyvvgroup/quemb
  cd quemb
  pip install .

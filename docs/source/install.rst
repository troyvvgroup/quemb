Installation
************

Prerequisites
-------------

 * Python :code:`3.10 <= version < 3.13`
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

  git clone --recurse-submodules https://https://github.com/troyvvgroup/quemb
  cd quemb
  pip install .

Documentation
-------------

Comprehensive documentation for QuEmb is available at `quemb/docs`. The documentation provides detailed infomation on installation, usage, API reference, and examples. To build the documentation locally, simply navigate to `docs` and build using `make html` or `make latexpdf`.

You can download the `PDF version <_static/quemb.pdf>`_.

Option 1: Download the `PDF version <_static/quemb.pdf>`_ of the documentation.

Option 2: Comprehensive documentation for QuEmb is available at `quemb/docs`. The documentation provides detailed infomation on installation, usage, API reference, and examples.

Option 3: Build the documentation locally.

.. code-block:: bash
 
    cd docs
    make html

or 

.. code-block:: bash
   
   make latexpdf

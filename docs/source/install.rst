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

Documentation
-------------

To build the documentation locally, do

.. code-block:: bash

    cd docs
    make html

or

.. code-block:: bash

   cd docs
   make latexpdf


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



Known issues and troubleshooting
--------------------------------

On macOS, the system-provided Clang compiler does not support OpenMP out of the
box. To enable OpenMP, it is recommended to use `Homebrew <https://brew.sh/>`_
to install either GCC or LLVM/Clang with the OpenMP runtime.

**Option 1 - GCC (includes OpenMP support by default):**

.. code-block:: bash

    brew install gcc
    CXX=$(brew --prefix gcc)/bin/g++ pip install .

**Option 2 - LLVM/Clang with OpenMP runtime:**

.. code-block:: bash

    brew install llvm libomp
    CXX=$(brew --prefix llvm)/bin/clang++ pip install .


Optional dependencies
---------------------

If you want to use the ORCA backend for Hartree-Fock you need to install ORCA from
`here <https://www.faccts.de/customer/login?came_from=/customer>`_.
This requires a registration and is free for academic use.
In addition you need to install the python interface via:


.. code-block:: bash

    pip install orca-pi

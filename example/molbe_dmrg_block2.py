# A run through the `QuEmb` interface with `block2` for performing BE-DMRG.
# `block2` is a DMRG and sparse tensor network library developed by the
# Garnet-Chan group at Caltech: https://block2.readthedocs.io/en/latest/index.html

import os

import matplotlib.pyplot as plt
import numpy
from pyscf import cc, fci, gto, scf

from molbe import BE, fragpart

# We'll consider the dissociation curve for a 1D chain of 8 H-atoms:
num_points = 3
seps = numpy.linspace(0.60, 1.6, num=num_points)
fci_ecorr, ccsd_ecorr, ccsdt_ecorr, bedmrg_ecorr = [], [], [], []

# Specify a scratch directory for fragment DMRG files:
scratch = os.getcwd()

for a in seps:
    # Hartree-Fock serves as the starting point for all BE calculations:
    mol = gto.M()
    mol.atom = [["H", (0.0, 0.0, i * a)] for i in range(8)]
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    # Exact diagonalization (FCI) will provide the reference:
    mc = fci.FCI(mf)
    fci_ecorr.append(mc.kernel()[0] - mf.e_tot)

    # CCSD and CCSD(T) are good additional points of comparison:
    mycc = cc.CCSD(mf).run()
    et_correction = mycc.ccsd_t()
    ccsd_ecorr.append(mycc.e_tot - mf.e_tot)
    ccsdt_ecorr.append(mycc.e_tot + et_correction - mf.e_tot)

    # Define BE1 fragments. Prior to DMRG the localization of MOs
    # is usually necessary. While there doesn't appear to be
    # any clear advantage to using any one scheme over another,
    # the Pipek-Mezey scheme continues to be the most popular. With
    # BE-DMRG, localization takes place prior to fragmentation:
    fobj = fragpart(be_type="be1", mol=mol)
    mybe = BE(
        mf,
        fobj,
        lo_method="pipek-mezey",  # or 'lowdin', 'iao', 'boys'
        pop_method="lowdin",  # or 'meta-lowdin', 'mulliken', 'iao', 'becke'
    )

    # Next, run BE-DMRG with default parameters and maxM=100.
    mybe.oneshot(
        solver="block2",  # or 'DMRG', 'DMRGSCF', 'DMRGCI'
        scratch=scratch,  # Scratch dir for fragment DMRG
        maxM=100,  # Max fragment bond dimension
        force_cleanup=True,  # Remove all fragment DMRG tmpfiles
    )

    bedmrg_ecorr.append(mybe.ebe_tot - mf.e_tot)
    # Setting `force_cleanup=True` will clean the scratch directory after each
    # fragment DMRG calculation finishes. DMRG tempfiles can be quite large, so
    # be sure to keep an eye on them if `force_cleanup=False` (default).

    # NOTE: This does *not* delete the log files `dmrg.conf` and `dmrg.out`for each frag,
    # which can still be found in `/scratch/`.

# Finally, plot the resulting potential energy curves:
fig, ax = plt.subplots()

ax.plot(seps, fci_ecorr, "o-", linewidth=1, label="FCI")
ax.plot(seps, ccsd_ecorr, "o-", linewidth=1, label="CCSD")
ax.plot(seps, ccsdt_ecorr, "o-", linewidth=1, label="CCSD(T)")
ax.plot(seps, bedmrg_ecorr, "o-", linewidth=1, label="BE1-DMRG")
ax.legend()

plt.savefig(os.path.join(scratch, f"BEDMRG_H8_PES{num_points}.png"))

# (See ../quemb/example/figures/BEDMRG_H8_PES20.png for an example.)

# For larger fragments, you'll want greater control over the fragment
# DMRG calculations. Using the same setup as above for a single geometry:
mol = gto.M()
mol.atom = [["H", (0.0, 0.0, i * 1.2)] for i in range(8)]
mol.basis = "sto-3g"
mol.charge = 0
mol.spin = 0
mol.build()
fobj = fragpart(be_type="be2", mol=mol)
mybe = BE(mf, fobj, lo_method="pipek-mezey", pop_method="lowdin")

# We automatically construct the fragment DMRG schedules based on user keywords. The following
# input, for example, yields a 60 sweep schedule which uses the two-dot algorithm from sweeps 0-49,
# and the one-dot algo from 50-60. The fragment orbitals are also reordered according the Fiedler
# vector procedure, along with a few other tweaks:

mybe.optimize(
    solver="block2",  # or 'DMRG', 'DMRGSCF', 'DMRGCI'
    scratch=scratch,  # Scratch dir for fragment DMRG
    startM=20,  # Initial fragment bond dimension (1st sweep)
    maxM=200,  # Maximum fragment bond dimension
    max_iter=60,  # Max number of sweeps
    twodot_to_onedot=50,  # Sweep num to switch from two- to one-dot algo.
    max_mem=40,  # Max memory (in GB) allotted to fragment DMRG
    max_noise=1e-3,  # Max MPS noise introduced per sweep
    min_tol=1e-8,  # Tighest Davidson tolerance per sweep
    block_extra_keyword=["fiedler"],  # Specify orbital reordering algorithm
    force_cleanup=True,  # Remove all fragment DMRG tmpfiles
    only_chem=True,
)

# Or, alternatively, we can construct a full schedule by hand:
schedule = {
    "scheduleSweeps": [0, 10, 20, 30, 40, 50],  # Sweep indices
    "scheduleMaxMs": [25, 50, 100, 200, 500, 500],  # Sweep maxMs
    "scheduleTols": [1e-5, 1e-5, 1e-6, 1e-6, 1e-8, 1e-8],  # Sweep Davidson tolerances
    "scheduleNoises": [0.01, 0.01, 0.001, 0.001, 1e-4, 0.0],  # Sweep MPS noise
}

# and pass it to the fragment solver through `schedule_kwargs`:
mybe.optimize(
    solver="block2",
    scratch=scratch,
    schedule_kwargs=schedule,
    block_extra_keyword=["fiedler"],
    force_cleanup=True,
    only_chem=True,
)

# To make sure the calculation is proceeding as expected, make sure to check `[scratch]/dmrg.conf`
# and `[scratch]/dmrg.out`, which are the fragment DMRG inputs and outputs, respectively, used
# by `block2`.

# NOTE: Parameters in `schedule_kwargs` will overwrite any other DMRG kwargs.
# NOTE: The DMRG schedule kwargs and related syntax follows the standard notation used in block2.

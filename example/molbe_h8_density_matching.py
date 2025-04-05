# Illustrates a simple molecular BE calculation with BE
# density matching between edge & centers of fragments.

from pyscf import fci, gto, scf

from quemb.molbe import BE, fragmentate

# PySCF HF generated mol & mf (molecular desciption & HF object)
mol = gto.M(
    atom="""
H 0. 0. 0.
H 0. 0. 1.
H 0. 0. 2.
H 0. 0. 3.
H 0. 0. 4.
H 0. 0. 5.
H 0. 0. 6.
H 0. 0. 7.
""",
    basis="sto-3g",
    charge=0,
)

mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# Perform PySCF FCI to get reference energy
mc = fci.FCI(mf)
fci_ecorr = mc.kernel()[0] - mf.e_tot
print(f"*** FCI Correlation Energy: {fci_ecorr:>14.8f} Ha", flush=True)

# Perform BE calculations with different fragment schemes:

# Define BE2 fragments
fobj = fragmentate(be_type="be2", mol=mol)
mybe = BE(mf, fobj)
mybe.optimize(solver="FCI")

# Compute BE error
be_ecorr = mybe.ebe_tot - mybe.ebe_hf
err_ = (fci_ecorr - be_ecorr) * 100.0 / fci_ecorr
print(f"*** BE2 Correlation Energy Error (%) : {err_:>8.4f} %")

# Define BE3 fragments
fobj = fragmentate(be_type="be3", mol=mol)
mybe = BE(mf, fobj)
mybe.optimize(solver="FCI")

# Compute BE error
be_ecorr = mybe.ebe_tot - mybe.ebe_hf
err_ = (fci_ecorr - be_ecorr) * 100.0 / fci_ecorr
print(f"*** BE3 Correlation Energy Error (%) : {err_:>8.4f} %")

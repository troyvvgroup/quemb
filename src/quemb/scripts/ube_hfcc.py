# Author(s): Lea Kjaergaard

"""
Compute the full (isotropic + anisotropic) hyperfine coupling tensor from a
BE-UCCSD spin density.

Built from PySCF's stable, core one-electron integrals (mol.intor).
The tensor comes from a single set of second-derivative-of-1/r integrals,
int1e_ipiprinv + int1e_iprinvip, which together give <nabla nabla (1/r)>.
Via the Poisson equation, nabla^2(1/r) = -4*pi*delta(r), so this object's
trace channel already is the isotropic Fermi-contact delta-function operator,
and its traceless part is the spin-dipolar operator. This comes directly from
the integral construction.

Usage:
    python ube_hfcc.py --xyz molecule.xyz --rdm1a rdm1a.npy --rdm1b rdm1b.npy
                       --charge -2 --spin 5 --basis def2-svp [--atoms Fe C]

Example:
    python ube_hfcc.py --xyz model_II.xyz --rdm1a model_II_rdm1a_BE2.npy
                       --rdm1b model_II_rdm1b_BE2.npy --charge -2 --spin 5
"""

import argparse

import numpy as np
from pyscf import gto
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor


def _fcdip_integral(mol, atom_id):
    nao = mol.nao
    with mol.with_rinv_origin(mol.atom_coord(atom_id)):
        ipipv = mol.intor("int1e_ipiprinv", 9).reshape(3, 3, nao, nao)
        ipvip = mol.intor("int1e_iprinvip", 9).reshape(3, 3, nao, nao)
        h1ao = ipipv + ipvip
        h1ao = h1ao + h1ao.transpose(0, 1, 3, 2)
        trace = h1ao[0, 0] + h1ao[1, 1] + h1ao[2, 2]
        idx = np.arange(3)
        h1ao[idx, idx] -= trace
    return h1ao


def hfc_tensor(mol, atom_id, dma, dmb):
    """Full (Fermi-contact + spin-dipolar) hyperfine coupling tensor for
    one nucleus, in MHz.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    atom_id : int
        0-indexed atom for which to compute the tensor.
    dma, dmb : numpy.ndarray, shape (nao, nao)
        Alpha and beta density matrices in the AO basis.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        The total HFC tensor in MHz.
    """
    spindm = dma - dmb
    effspin = mol.spin * 0.5

    e_gyro = 0.5 * nist.G_ELECTRON
    nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
    au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
    fac = nist.ALPHA**2 / 2 / effspin * e_gyro * au2MHz
    nuc_gyro = get_nuc_g_factor(mol.atom_symbol(atom_id)) * nuc_mag

    h1_fcdip = _fcdip_integral(mol, atom_id)
    fcsd = np.einsum("xyij,ji->xy", h1_fcdip, spindm)

    return fac * nuc_gyro * fcsd


def hfc_principal_values(tensor):
    """Diagonalize a (3,3) HFC tensor. Returns (A_principal sorted
    ascending, A_iso = trace/3)."""
    evals = np.linalg.eigvalsh(0.5 * (tensor + tensor.T))
    a_iso = float(np.trace(tensor)) / 3.0
    return evals, a_iso


def compute_hfcc(mol, dma, dmb, atoms=None):
    """Compute the full (isotropic + anisotropic) HFC tensor for a set of
    atoms from alpha/beta density matrices in the AO basis.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    dma, dmb : numpy.ndarray, shape (nao, nao)
    atoms : list of int, optional. Default is all atoms.

    Returns
    -------
    list of dict with keys : index, symbol, tensor, evals, a_iso
    """
    if atoms is None:
        atoms = range(mol.natm)

    results = []
    for i in atoms:
        symbol = mol.atom_symbol(i)
        if get_nuc_g_factor(symbol) == 0:
            continue

        tensor = hfc_tensor(mol, i, dma, dmb)
        evals, a_iso = hfc_principal_values(tensor)
        results.append(
            {
                "index": i,
                "symbol": symbol,
                "tensor": tensor,
                "evals": evals,
                "a_iso": a_iso,
            }
        )

    return results


def print_hfcc_table(results, title="Hyperfine Coupling Constants"):
    print(f"\n{title}")
    print("=" * 56)
    print(
        f"{'Atom':>4} {'Symbol':>6} {'A_1':>10} {'A_2':>10} {'A_3':>10} {'A_iso':>10}"
    )
    print("-" * 56)
    for r in results:
        e = r["evals"]
        print(
            f"{r['index']:>4} {r['symbol']:>6} "
            f"{e[0]:>10.4f} {e[1]:>10.4f} {e[2]:>10.4f} {r['a_iso']:>10.4f}"
        )
    print("=" * 56)


def main():
    parser = argparse.ArgumentParser(
        description="Compute the full HFC tensor from a BE-UCCSD spin density"
    )
    parser.add_argument("--xyz", required=True, help="XYZ geometry file")
    parser.add_argument(
        "--name",
        default=None,
        help="Shortcut: sets --rdm1a to NAME_rdm1a.npy and --rdm1b to NAME_rdm1b.npy",
    )
    parser.add_argument(
        "--rdm1a", required=False, default=None, help="Alpha 1-RDM .npy file"
    )
    parser.add_argument(
        "--rdm1b", required=False, default=None, help="Beta 1-RDM .npy file"
    )
    parser.add_argument("--charge", required=True, type=int, help="Molecular charge")
    parser.add_argument(
        "--spin", required=True, type=int, help="2S (number of unpaired electrons)"
    )
    parser.add_argument(
        "--basis", default="def2-svp", help="Basis set (default: def2-svp)"
    )
    parser.add_argument(
        "--atoms", nargs="+", help="Atom symbols to print e.g. Fe C H (default: all)"
    )
    parser.add_argument(
        "--unit", default="angstrom", help="Coordinate unit (default: angstrom)"
    )
    args = parser.parse_args()
    if args.name is not None:
        if args.rdm1a is None:
            args.rdm1a = f"{args.name}_rdm1a.npy"
        if args.rdm1b is None:
            args.rdm1b = f"{args.name}_rdm1b.npy"

    # Build molecule
    mol = gto.M()
    with open(args.xyz) as f:
        lines = f.readlines()
    # Handle both raw xyz (no header) and standard xyz (2-line header)
    try:
        int(lines[0].strip())
        mol.atom = "".join(lines[2:])  # standard xyz with natom + comment lines
    except ValueError:
        mol.atom = "".join(lines)  # raw xyz with no header
    mol.basis = args.basis
    mol.charge = args.charge
    mol.spin = args.spin
    mol.unit = args.unit
    mol.build()

    print(f"Molecule: {args.xyz}", flush=True)
    print(f"Basis: {args.basis}, charge={args.charge}, spin={args.spin}", flush=True)
    print(f"nao: {mol.nao_nr()}, nelec: {mol.nelec}", flush=True)

    # Load RDMs
    rdm1a = np.load(args.rdm1a)
    rdm1b = np.load(args.rdm1b)
    spin_density = rdm1a - rdm1b

    print("\nRDM validation:", flush=True)
    S = mol.intor("int1e_ovlp")
    print(
        f"  Trace rdm1a: {np.trace(rdm1a @ S):.4f} (expected {mol.nelec[0]})",
        flush=True,
    )
    print(
        f"  Trace rdm1b: {np.trace(rdm1b @ S):.4f} (expected {mol.nelec[1]})",
        flush=True,
    )
    print(
        f"  Net spin:    {np.trace(spin_density @ S):.4f} (expected {mol.spin})",
        flush=True,
    )

    # Filter atoms by symbol if requested
    if args.atoms:
        atom_indices = [i for i in range(mol.natm) if mol.atom_symbol(i) in args.atoms]
        print(f"\nComputing HFCCs for atoms: {args.atoms}", flush=True)
    else:
        atom_indices = None
        print("\nComputing HFCCs for all atoms", flush=True)

    results = compute_hfcc(mol, rdm1a, rdm1b, atoms=atom_indices)
    print_hfcc_table(results, title=f"BE-UCCSD Hyperfine Coupling ({args.basis})")


if __name__ == "__main__":
    main()

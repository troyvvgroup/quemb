"""
Compute isotropic Fermi contact HFCCs from a BE-UCCSD spin density.

Usage:
    python ube_hfcc.py --xyz molecule.xyz --rdm1a rdm1a.npy --rdm1b rdm1b.npy
                       --charge -2 --spin 5 --basis def2-svp [--atoms Fe C]

Example:
    python ube_hfcc.py --xyz model_II.xyz --rdm1a model_II_rdm1a_BE2.npy
                       --rdm1b model_II_rdm1b_BE2.npy --charge -2 --spin 5

Author: Lea Kjaergaard Northcote
"""
import argparse
import numpy as np
from pyscf import gto
from pyscf.data import nist
from pyscf.data.gyro import get_nuc_g_factor


def compute_fermi_contact(mol, spin_density_ao, atoms=None):
    """
    Compute isotropic Fermi contact HFCCs from spin density in AO basis.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    spin_density_ao : numpy.ndarray, shape (nao, nao)
    atoms : list of int, optional. Default is all atoms.

    Returns
    -------
    list of dict with keys: index, symbol, a_iso_mhz, rho_spin
    """
    if atoms is None:
        atoms = range(mol.natm)

    nuc_mag = 0.5 * (nist.E_MASS / nist.PROTON_MASS)
    au2MHz = nist.HARTREE2J / nist.PLANCK * 1e-6
    e_gyro = 0.5 * nist.G_ELECTRON
    effspin = mol.spin * 0.5
    fac = nist.ALPHA**2 / 2 / effspin * e_gyro * au2MHz

    results = []
    for i in atoms:
        symbol = mol.atom_symbol(i)
        g_factor = get_nuc_g_factor(symbol)
        if g_factor == 0:
            continue

        nuc_gyro = g_factor * nuc_mag
        coords = mol.atom_coord(i).reshape(1, 3)
        ao = mol.eval_gto('GTOval', coords)
        h1fc = 4 * np.pi / 3 * np.einsum('ip,iq->pq', ao, ao)
        fc = np.einsum('ij,ji', h1fc, spin_density_ao)
        a_iso = fac * nuc_gyro * fc

        results.append({
            'index': i,
            'symbol': symbol,
            'a_iso_mhz': a_iso,
            'rho_spin': fc,
        })

    return results


def print_hfcc_table(results, title="Fermi Contact HFCCs"):
    print(f"\n{title}")
    print("=" * 50)
    print(f"{'Atom':>4} {'Symbol':>6} {'a_iso (MHz)':>14} {'rho_spin (au)':>14}")
    print("-" * 50)
    for r in results:
        print(f"{r['index']:>4} {r['symbol']:>6} "
              f"{r['a_iso_mhz']:>14.4f} {r['rho_spin']:>14.6f}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Compute Fermi contact HFCCs from BE-UCCSD spin density"
    )
    parser.add_argument("--xyz",     required=True,  help="XYZ geometry file")
    parser.add_argument("--name",    default=None,   help="Shortcut: sets --rdm1a to NAME_rdm1a.npy and --rdm1b to NAME_rdm1b.npy")
    parser.add_argument("--rdm1a",   required=False, default=None, help="Alpha 1-RDM .npy file")
    parser.add_argument("--rdm1b",   required=False, default=None, help="Beta 1-RDM .npy file")
    parser.add_argument("--charge",  required=True,  type=int, help="Molecular charge")
    parser.add_argument("--spin",    required=True,  type=int, help="2S (number of unpaired electrons)")
    parser.add_argument("--basis",   default="def2-svp", help="Basis set (default: def2-svp)")
    parser.add_argument("--atoms",   nargs="+",      help="Atom symbols to print e.g. Fe C H (default: all)")
    parser.add_argument("--unit",    default="angstrom", help="Coordinate unit (default: angstrom)")
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
        mol.atom = ''.join(lines[2:])  # standard xyz with natom + comment lines
    except ValueError:
        mol.atom = ''.join(lines)      # raw xyz with no header
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

    print(f"\nRDM validation:", flush=True)
    S = mol.intor('int1e_ovlp')
    print(f"  Trace rdm1a: {np.trace(rdm1a @ S):.4f} (expected {mol.nelec[0]})", flush=True)
    print(f"  Trace rdm1b: {np.trace(rdm1b @ S):.4f} (expected {mol.nelec[1]})", flush=True)
    print(f"  Net spin:    {np.trace(spin_density @ S):.4f} (expected {mol.spin})", flush=True)

    # Filter atoms by symbol if requested
    if args.atoms:
        atom_indices = [i for i in range(mol.natm)
                       if mol.atom_symbol(i) in args.atoms]
        print(f"\nComputing HFCCs for atoms: {args.atoms}", flush=True)
    else:
        atom_indices = None
        print(f"\nComputing HFCCs for all atoms", flush=True)
    
    results = compute_fermi_contact(mol, spin_density, atoms=atom_indices)
    print_hfcc_table(results, title=f"BE-UCCSD Fermi Contact HFCCs ({args.basis})")


if __name__ == "__main__":
    main()

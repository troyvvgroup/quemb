"""
This script tests the local gradient of C8 alkane chain with BE2/sto-3g

Author(s): Carina Luo
"""

import unittest

import numpy as np
from pyscf import cc, grad, gto, scf
from quemb.molbe import BE, fragmentate
from quemb.molbe.chemfrag import Fragmented
from quemb.molbe.chemfrag import ChemGenArgs


class Test_local_gradient(unittest.TestCase):
    def test_C8alkane_sto3g_BE2(self):
        def compute_energy(
            mf0,
            mol,
            frag_idx,
            fobj,
            eq_fobjs,
            guess_dm,
        ):
            mf = scf.RHF(mol)
            mf.kernel(dm0=guess_dm)

            mybe = BE(mf, fobj, eq_fobjs=eq_fobjs, gradient_orb_space="RDM-invariant")
            frag = mybe.Fobjs[frag_idx]

            mycc = cc.CCSD(frag._mf, mo_coeff=frag.mo_coeffs, mo_occ=frag._mf.mo_occ)
            mycc.verbose = 0
            mycc.incore_complete = True
            eris = mycc.ao2mo()
            eris.mo_energy = frag._mf.mo_energy
            eris.fock = np.diag(frag._mf.mo_energy)
            mycc.kernel(eris=eris)
            e_corr = mycc.e_corr

            e_total = mybe.ebe_hf

            return e_total, e_corr

        def compute_grad_component(
            mf0,
            atom_idx,
            xyz,
            coords,
            frag_idx,
            delta,
            mol,
            fobj,
            eq_fobjs,
            guess_dm,
        ):
            coords[atom_idx, xyz] += delta
            mol.set_geom_(coords, unit="Bohr")
            e_plus_hf, e_plus_corr = compute_energy(
                mf0,
                mol,
                frag_idx,
                fobj,
                eq_fobjs,
                guess_dm,
            )

            coords[atom_idx, xyz] -= 2 * delta
            mol.set_geom_(coords, unit="Bohr")
            e_minus_hf, e_minus_corr = compute_energy(
                mf0,
                mol,
                frag_idx,
                fobj,
                eq_fobjs,
                guess_dm,
            )

            grad_hf = (e_plus_hf - e_minus_hf) / (2 * delta)
            grad_ccsd = grad_hf + (e_plus_corr - e_minus_corr) / (2 * delta)

            return grad_ccsd, grad_hf

        def compute_gradients(
            mol,
            mf0,
            frag_per_atom,
            fobj,
            eq_fobjs,
            guess_dm,
            delta=1e-4,
        ):
            gradient_ccsd = np.zeros_like(mol.atom_coords())
            gradient_hf = np.zeros_like(mol.atom_coords())

            coords = mol.atom_coords()
            for atom_idx, frag_idx in enumerate(frag_per_atom):
                for xyz in range(3):
                    grad_ccsd, grad_hf_val = compute_grad_component(
                        mf0,
                        atom_idx,
                        xyz,
                        coords,
                        frag_idx,
                        delta,
                        mol,
                        fobj,
                        eq_fobjs,
                        guess_dm,
                    )

                    coords[atom_idx, xyz] += delta
                    gradient_ccsd[atom_idx, xyz] = grad_ccsd
                    gradient_hf[atom_idx, xyz] = grad_hf_val

            return gradient_ccsd, gradient_hf

        def get_eq_fobjs(mybe):
            eq_fobjs = []
            for each_fragment in mybe.Fobjs:  # for each reference fragment
                Dhf = (
                    mybe.lmo_coeff[:, : mybe.Nocc] @ mybe.lmo_coeff[:, : mybe.Nocc].T
                )  # get the density matrix in the LMO basis
                D_SO = (
                    each_fragment.TAfull_lo_eo.T @ Dhf @ each_fragment.TAfull_lo_eo
                )  # get the density matrix in the embedding basis

                eigvals, eigvecs = np.linalg.eigh(D_SO)  # diagonalize
                eigvals = eigvals[::-1]
                eigvecs = eigvecs[:, ::-1]
                each_fragment.eigvecs = eigvecs

                occ_idx = np.arange(mybe.Nocc)
                virt_idx = np.arange(mybe.Nocc, eigvecs.shape[1])
                each_fragment.TA_occ = each_fragment.TAfull_lo_eo @ eigvecs[:, occ_idx]
                each_fragment.TA_virt = (
                    each_fragment.TAfull_lo_eo @ eigvecs[:, virt_idx]
                )

                # for schmidt invariant corresponding orbitals
                each_fragment.TA_lo_eo_frag = each_fragment.TA_lo_eo[
                    :, : each_fragment.n_f
                ]
                each_fragment.TA_lo_eo_bath = each_fragment.TA_lo_eo[
                    :, each_fragment.n_f :
                ]
                eq_fobjs.append(each_fragment)
            return eq_fobjs

        def rms_diff(a, b):
            return np.sqrt(np.mean((a - b) ** 2))

        mol = gto.M(
            atom="""
        C -6.50082061 0.42809441 -0.33125845
        H -6.89795577 1.27644728 -0.88166622
        H -6.89851790 0.46366805 0.67973292
        H -6.86086259 -0.48110058 -0.80561333
        C -4.95959048 0.46121645 -0.31682643
        H -4.58969047 0.39421197 -1.33946393
        H -4.58933583 -0.41358544 0.21548967
        C -4.40521586 1.73906799 0.35030230
        H -4.77564149 2.60941023 -0.18158514
        H -4.77784375 1.80181541 1.37035032
        C -2.86136706 1.77084329 0.36858865
        H -2.49136107 1.70460430 -0.65370521
        H -2.48801590 0.89674287 0.89860579
        C -2.30516475 3.04621978 1.03180202
        H -2.67809827 3.92327809 0.50081696
        H -2.67799737 3.11148497 2.05341202
        C -0.76077219 3.07612798 1.04978690
        H -0.38828821 3.01244561 0.02672814
        H -0.38871447 2.20394310 1.58089808
        C -0.20115324 4.35409399 1.71443939
        H -0.57312305 5.22874412 1.18222956
        H -0.57410325 4.42002077 2.73518330
        C 1.33740594 4.38392965 1.72949319
        H 1.73459871 4.34800504 0.71920427
        H 1.73574942 3.53287480 2.27926264
        H 1.69946137 5.29112978 2.20278235
        """,
            basis="sto-3g",
            charge=0,
            unit="Angstrom",
        )

        n_BE = 2
        treat_H_different = True

        args = ChemGenArgs(treat_H_different=treat_H_different)
        fragmented = Fragmented.from_mole(
            mol, n_BE=n_BE, treat_H_different=treat_H_different
        )
        cpf = fragmented.frag_structure.centers_per_frag
        apf = fragmented.frag_structure.atoms_per_frag

        cpf_h = []
        for fidx, frag in enumerate(apf):
            cpf_h_f = []
            for atm in frag:
                if atm in cpf[fidx]:
                    cpf_h_f.append(atm)
                elif any(atm in f for f in cpf):
                    break
                else:
                    cpf_h_f.append(atm)
            cpf_h.append(cpf_h_f)

        frag_per_atom = [
            next(i for i, f in enumerate(cpf_h) if atom_idx in f)
            for atom_idx in range(mol.natm)
        ]

        mf = scf.RHF(mol)  # .density_fit(auxbasis="weigend")
        mf.kernel()
        guess_dm = mf.make_rdm1()

        hf_ref_grad_obj = grad.RHF(mf)
        gradient_hf_ref = hf_ref_grad_obj.kernel()

        mycc = cc.CCSD(mf)
        mycc.kernel()
        grad_ccsd = mycc.nuc_grad_method()
        gradient_ref = grad_ccsd.kernel()

        fobj = fragmentate(
            mol=mol, frag_type="chemgen", n_BE=n_BE, additional_args=args
        )
        mybe = BE(mf, fobj, gradient_orb_space="Unmodified")
        mybe.oneshot()

        eq_fobjs = get_eq_fobjs(mybe)
        print(f"the type of eq_fobjs is {type(eq_fobjs)}")

        gradient, gradient_hf = compute_gradients(
            mol,
            mf,
            frag_per_atom,
            fobj,
            eq_fobjs,
            guess_dm,
            delta=1e-4,
        )

        rms_diff_ccsd = rms_diff(gradient_ref, gradient)
        rms_diff_hf = rms_diff(gradient_hf_ref, gradient_hf)

        expected = np.loadtxt("gradient.txt")
        expected_hf = np.loadtxt("gradient_hf.txt")
        assert np.allclose(gradient,expected)
        assert np.allclose(gradient_hf,expected_hf)

if __name__ == "__main__":
    unittest.main()

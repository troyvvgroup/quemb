# Author(s): Minsik Cho, Leah Weisburn

import numpy, os
from pyscf import gto, scf
import time

from pyscf.lib import chkfile


def libint2pyscf(
    xyzfile,
    hcore,
    basis,
    hcore_skiprows=1,
    use_df=False,
    unrestricted=False,
    spin=0,
    charge=0,
):
    """Build a pyscf Mole and RHF/UHF object using the given xyz file
    and core Hamiltonian (in libint standard format)
    c.f.
    In libint standard format, the basis sets appear in the order
    atom#   n   l   m
    0       1   0   0   1s
    0       2   0   0   2s
    0       2   1   -1  2py
    0       2   1   0   2pz
    0       2   1   1   2px
    ...
    In pyscf, the basis sets appear in the order
    atom #  n   l   m
    0       1   0   0   1s
    0       2   0   0   2s
    0       2   1   1   2px
    0       2   1   -1  2py
    0       2   1   0   2pz
    ...
    For higher angular momentum, both use [-l, -l+1, ..., l-1, l] ordering.


    Parameters
    ----------
    xyzfile : string
        Path to the xyz file
    hcore : string
        Path to the core Hamiltonian
    basis : string
        Name of the basis set
    hcore_skiprows : int, optional
        # of first rows to skip from the core Hamiltonian file, by default 1
    use_df : bool, optional
        If true, use density-fitting to evaluate the two-electron integrals
    unrestricted : bool, optional
        If true, use UHF bath
    spin : int, optional
        2S, Difference between the number of alpha and beta electrons
    charge : int, optional
        Total charge of the system

    Returns
    -------
    (pyscf.gto.Mole, pyscf.scf.RHF or pyscf.scf.UHF)
    """
    # Check input validity
    assert os.path.exists(xyzfile), "Input xyz file does not exist"
    assert os.path.exists(hcore), "Input core Hamiltonian file does not exist"

    mol = gto.M(atom=xyzfile, basis=basis, spin=spin, charge=charge)
    hcore_libint = numpy.loadtxt(hcore, skiprows=hcore_skiprows)

    libint2pyscf = []
    for labelidx, label in enumerate(mol.ao_labels()):
        # pyscf: px py pz // 1 -1 0
        # libint: py pz px // -1 0 1
        if "p" not in label.split()[2]:
            libint2pyscf.append(labelidx)
        else:
            if "x" in label.split()[2]:
                libint2pyscf.append(labelidx + 2)
            elif "y" in label.split()[2]:
                libint2pyscf.append(labelidx - 1)
            elif "z" in label.split()[2]:
                libint2pyscf.append(labelidx - 1)

    hcore_pyscf = hcore_libint[numpy.ix_(libint2pyscf, libint2pyscf)]

    mol.incore_anyway = True
    if use_df:
        mf = scf.UHF(mol).density_fit() if unrestricted else scf.RHF(mol).density_fit()
        from pyscf import df

        mydf = df.DF(mol).build()
        mf.with_df = mydf
    else:
        mf = scf.UHF(mol) if unrestricted else scf.RHF(mol)
    mf.get_hcore = lambda *args: hcore_pyscf

    return mol, mf


def be2fcidump(be_obj, fcidump_prefix, basis):
    """Construct FCIDUMP file for each fragment in a given BE object
    Assumes molecular, restricted BE calculation

    Parameters
    ----------
    be_obj : molbe.mbe.BE
        BE object
    fcidump_prefix : string
        Prefix for path & filename to the output fcidump files
        Each file is named [fcidump_prefix]_f0, ...
    basis : string
        'embedding' to get the integrals in the embedding basis
        'fragment_mo' to get the integrals in the fragment MO basis
    """
    from pyscf import ao2mo
    from pyscf.tools import fcidump
    import h5py

    for fidx, frag in enumerate(be_obj.Fobjs):
        # Read in eri
        read = h5py.File(frag.eri_file, "r")
        eri = read[frag.dname][()]  # 2e in embedding basis
        read.close()
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = numpy.einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = numpy.einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx),
            h1e,
            h2e,
            frag.TA.shape[1],
            frag.nsocc,
            ms=0,
        )


def ube2fcidump(be_obj, fcidump_prefix, basis):
    """Construct FCIDUMP file for each fragment in a given BE object
    Assumes molecular, restricted BE calculation

    Parameters
    ----------
    be_obj : molbe.mbe.BE
        BE object
    fcidump_prefix : string
        Prefix for path & filename to the output fcidump files
        Each file is named [fcidump_prefix]_f0, ...
    basis : string
        'embedding' to get the integrals in the embedding basis
        'fragment_mo' to get the integrals in the fragment MO basis
    """
    from pyscf import ao2mo
    from pyscf.tools import fcidump
    import h5py

    for fidx, frag in enumerate(be_obj.Fobjs_a):
        # Read in eri
        read = h5py.File(frag.eri_file, "r")
        eri = read[frag.dname][()]  # 2e in embedding basis
        read.close()
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = numpy.einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = numpy.einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx) + "a",
            h1e,
            h2e,
            frag.TA.shape[1],
            frag.nsocc,
            ms=0,
        )

    for fidx, frag in enumerate(be_obj.Fobjs_b):
        # Read in eri
        read = h5py.File(frag.eri_file, "r")
        eri = read[frag.dname][()]  # 2e in embedding basis
        read.close()
        eri = ao2mo.restore(1, eri, frag.nao)
        if basis == "embedding":
            h1e = frag.fock
            h2e = eri
        elif basis == "fragment_mo":
            frag.scf()  # make sure that we have mo coefficients
            h1e = numpy.einsum(
                "ij,ia,jb->ab", frag.fock, frag.mo_coeffs, frag.mo_coeffs, optimize=True
            )
            h2e = numpy.einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                eri,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                frag.mo_coeffs,
                optimize=True,
            )
        else:
            raise Exception("Basis should be either embedding or fragment_mo")

        fcidump.from_integrals(
            fcidump_prefix + "f" + str(fidx) + "b",
            h1e,
            h2e,
            frag.TA.shape[1],
            frag.nsocc,
            ms=0,
        )


def be2puffin(
    xyzfile,
    basis,
    hcore=None,
    libint_inp=False,
    pts_and_charges=None,
    jk=None,
    use_df=False,
    charge=0,
    spin=0,
    nproc=1,
    ompnum=1,
    be_type="be1",
    df_aux_basis=None,
    frozen_core=True,
    localization_method="lowdin",
    localization_basis=None,
    unrestricted=False,
    from_chk=False,
    checkfile=None,
    ecp=None,
):
    """Front-facing API bridge tailored for SCINE Puffin
    Returns the CCSD oneshot energies
    - QM/MM notes: Using QM/MM alongside big basis sets, especially with a frozen
    core, can cause localization and numerical stability problems. Use with
    caution. Additional work to this end on localization, frozen core, ECPs,
    and QM/MM in this capacity is ongoing.
    - If running unrestricted QM/MM calculations, with ECPs, in a large basis set,
    do not freeze the core. Using an ECP for heavy atoms improves the localization
    numerics, but this is not yet compatible with frozen core on the rest of the atoms.

    Parameters
    ----------
    xyzfile : string
        Path to the xyz file
    basis : string
        Name of the basis set
    hcore : numpy.array
        Two-dimensional array of the core Hamiltonian
    libint_inp : bool
        True for hcore provided in Libint format. Else, hcore input is in PySCF format
        Default is False, i.e., hcore input is in PySCF format
    pts_and_charges : tuple of numpy.array
        QM/MM (points, charges). Use pyscf's QM/MM instead of starting Hamiltonian
    jk : numpy.array
        Coulomb and Exchange matrices (pyscf will calculate this if not given)
    use_df : bool, optional
        If true, use density-fitting to evaluate the two-electron integrals
    charge : int, optional
        Total charge of the system
    spin : int, optional
        Total spin of the system, pyscf definition
    nproc : int, optional
    ompnum: int, optional
        Set number of processors and ompnum for the jobs
    frozen_core: bool, optional
        Whether frozen core approximation is used or not, by default True
    localization_method: string, optional
        For now, lowdin is best supported for all cases. IAOs to be expanded
        By default 'lowdin'
    localization_basis: string, optional
        IAO minimal-like basis, only nead specification with IAO localization
        By default None
    unrestricted: bool, optional
        Unrestricted vs restricted HF and CCSD, by default False
    from_chk: bool, optional
        Run calculation from converged RHF/UHF checkpoint. By default False
    checkfile: string, optional
        if not None:
        - if from_chk: specify the checkfile to run the embedding calculation
        - if not from_chk: specify where to save the checkfile
        By default None
    ecp: string, optional
        specify the ECP for any atoms, accompanying the basis set
        syntax: {'Atom_X': 'ECP_for_X'; 'Atom_Y': 'ECP_for_Y'}
        By default None
    """
    from .fragment import fragpart
    from .mbe import BE
    from .ube import UBE

    # Check input validity
    assert os.path.exists(xyzfile), "Input xyz file does not exist"

    mol = gto.M(atom=xyzfile, basis=basis, charge=charge, spin=spin)

    if not from_chk:
        if hcore is None:  # from point charges OR with no external potential
            hcore_pyscf = None
        else:  # from starting Hamiltonian in Libint format
            if libint_inp == True:
                libint2pyscf = []
                for labelidx, label in enumerate(mol.ao_labels()):
                    # pyscf: px py pz // 1 -1 0
                    # libint: py pz px // -1 0 1
                    if "p" not in label.split()[2]:
                        libint2pyscf.append(labelidx)
                    else:
                        if "x" in label.split()[2]:
                            libint2pyscf.append(labelidx + 2)
                        elif "y" in label.split()[2]:
                            libint2pyscf.append(labelidx - 1)
                        elif "z" in label.split()[2]:
                            libint2pyscf.append(labelidx - 1)

                hcore_pyscf = hcore[numpy.ix_(libint2pyscf, libint2pyscf)]
            else:
                # Input hcore is in PySCF format
                hcore_pyscf = hcore
        if not jk is None:
            jk_pyscf = (
                jk[0][
                    numpy.ix_(libint2pyscf, libint2pyscf, libint2pyscf, libint2pyscf)
                ],
                jk[1][
                    numpy.ix_(libint2pyscf, libint2pyscf, libint2pyscf, libint2pyscf)
                ],
            )

        mol.incore_anyway = True
        if unrestricted:
            if use_df and jk is None:
                print("UHF and df are incompatible: use_df = False")
                use_df = False
            if hcore is None:
                if pts_and_charges:
                    from pyscf import qmmm

                    print(
                        "Using QM/MM Point Charges: Assuming QM structure in Angstrom and MM Coordinates in Bohr !!!"
                    )
                    mf1 = scf.UHF(mol).set(
                        max_cycle=200
                    )  # using SOSCF is more reliable
                    # mf1 = scf.UHF(mol).set(max_cycle = 200, level_shift = (0.3, 0.2))
                    # using level shift helps, but not always. level_shift and
                    # scf.addons.dynamic_level_shift do not seem to work with QM/MM
                    # note: from the SCINE database, the structure is in Angstrom but the MM point charges
                    # are in Bohr !!
                    mf = qmmm.mm_charge(
                        mf1, pts_and_charges[0], pts_and_charges[1], unit="bohr"
                    ).newton()  # mf object, coordinates, charges
                else:
                    mf = scf.UHF(mol).set(max_cycle=200, level_shift=(0.3, 0.2))
            else:
                mf = scf.UHF(mol).set(max_cycle=200).newton()
        else:  # restricted
            if pts_and_charges:  # running QM/MM
                from pyscf import qmmm

                print(
                    "Using QM/MM Point Charges: Assuming QM structure in Angstrom and MM Coordinates in Bohr !!!"
                )
                mf1 = scf.RHF(mol).set(max_cycle=200)
                mf = qmmm.mm_charge(
                    mf1, pts_and_charges[0], pts_and_charges[1], unit="bohr"
                ).newton()
                print(
                    "Setting use_df to false and jk to none: have not tested DF and QM/MM from point charges at the same time"
                )
                use_df = False
                jk = None
            elif use_df and jk is None:
                from pyscf import df

                mf = scf.RHF(mol).density_fit(auxbasis=df_aux_basis)
            else:
                mf = scf.RHF(mol)

        if not hcore is None:
            mf.get_hcore = lambda *args: hcore_pyscf
        if not jk is None:
            mf.get_jk = lambda *args: jk_pyscf

        if checkfile:
            print("Saving checkfile to:", checkfile)
            mf.chkfile = checkfile
        time_pre_mf = time.time()
        mf.kernel()
        time_post_mf = time.time()
        if mf.converged == True:
            print("Reference HF Converged", flush=True)
        else:
            print("Reference HF Unconverged -- stopping the calculation", flush=True)
            sys.exit()
        if use_df:
            print(
                "Using auxillary basis in density fitting: ",
                mf.with_df.auxmol.basis,
                flush=True,
            )
            print("DF auxillary nao_nr", mf.with_df.auxmol.nao_nr(), flush=True)
        print("Time for mf kernel to run: ", time_post_mf - time_pre_mf, flush=True)

    elif from_chk:
        print("Running from chkfile", checkfile, flush=True)
        scf_result_dic = chkfile.load(checkfile, "scf")
        if unrestricted:
            mf = scf.UHF(mol)
        else:
            mf = scf.RHF(mol)
        print("Running from chkfile not tested with density fitting: DF set to None")
        mf.with_df = None
        mf.__dict__.update(scf_result_dic)
        time_post_mf = time.time()
        print("Chkfile electronic energy:", mf.energy_elec(), flush=True)
        print("Chkfile e_tot:", mf.e_tot, flush=True)

    # Finished initial reference HF: now, fragmentation step

    fobj = fragpart(
        be_type=be_type, frag_type="autogen", mol=mol, frozen_core=frozen_core
    )
    time_post_fragpart = time.time()
    print(
        "Time for fragmentation to run: ", time_post_fragpart - time_post_mf, flush=True
    )

    # Run embedding setup

    if unrestricted:
        mybe = UBE(mf, fobj, lo_method="lowdin")
        solver = "UCCSD"
    else:
        mybe = BE(mf, fobj, lo_method="lowdin")
        solver = "CCSD"

    # Run oneshot embedding and return system energy

    mybe.oneshot(
        solver=solver, nproc=nproc, ompnum=ompnum, calc_frag_energy=True, clean_eri=True
    )
    return mybe.ebe_tot


def print_energy(ecorr, e_V_Kapprox, e_F_dg, e_hf):
    # Print energy results
    print("-----------------------------------------------------", flush=True)
    print(" BE ENERGIES with cumulant-based expression", flush=True)

    print("-----------------------------------------------------", flush=True)
    print(" E_BE = E_HF + Tr(F del g) + Tr(V K_approx)", flush=True)
    print(" E_HF            : {:>14.8f} Ha".format(e_hf), flush=True)
    print(" Tr(F del g)     : {:>14.8f} Ha".format(e_F_dg), flush=True)
    print(" Tr(V K_aprrox)  : {:>14.8f} Ha".format(e_V_Kapprox), flush=True)
    print(" E_BE            : {:>14.8f} Ha".format(ecorr + e_hf), flush=True)
    print(" Ecorr BE        : {:>14.8f} Ha".format(ecorr), flush=True)
    print("-----------------------------------------------------", flush=True)

    print(flush=True)

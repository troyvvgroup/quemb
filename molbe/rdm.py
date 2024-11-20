# Author(s): Oinam Romesh Meitei

import numpy, sys, scipy


def rdm1_fullbasis(
    self,
    return_ao=True,
    only_rdm1=False,
    only_rdm2=False,
    return_lo=False,
    return_RDM2=True,
    print_energy=False,
):
    """
    Compute the one-particle and two-particle reduced density matrices (RDM1 and RDM2).

    Parameters:
    -----------
    return_ao : bool, optional
        Whether to return the RDMs in the AO basis. Default is True.
    only_rdm1 : bool, optional
        Whether to compute only the RDM1. Default is False.
    only_rdm2 : bool, optional
        Whether to compute only the RDM2. Default is False.
    return_lo : bool, optional
        Whether to return the RDMs in the localized orbital (LO) basis. Default is False.
    return_RDM2 : bool, optional
        Whether to return the two-particle RDM (RDM2). Default is True.
    print_energy : bool, optional
        Whether to print the energy contributions. Default is False.

    Returns:
    --------
    rdm1AO : numpy.ndarray
        The one-particle RDM in the AO basis.
    rdm2AO : numpy.ndarray
        The two-particle RDM in the AO basis (if return_RDM2 is True).
    rdm1LO : numpy.ndarray
        The one-particle RDM in the LO basis (if return_lo is True).
    rdm2LO : numpy.ndarray
        The two-particle RDM in the LO basis (if return_lo and return_RDM2 are True).
    rdm1MO : numpy.ndarray
        The one-particle RDM in the molecular orbital (MO) basis (if return_ao is False).
    rdm2MO : numpy.ndarray
        The two-particle RDM in the MO basis (if return_ao is False and return_RDM2 is True).
    """
    from pyscf import scf, ao2mo

    # Copy the molecular orbital coefficients
    C_mo = self.C.copy()
    nao, nmo = C_mo.shape

    # Initialize density matrices for atomic orbitals (AO)
    rdm1AO = numpy.zeros((nao, nao))
    rdm2AO = numpy.zeros((nao, nao, nao, nao))

    for fobjs in self.Fobjs:
        if return_RDM2:
            # Adjust the one-particle reduced density matrix (RDM1)
            drdm1 = fobjs.__rdm1.copy()
            drdm1[numpy.diag_indices(fobjs.nsocc)] -= 2.0

            # Compute the two-particle reduced density matrix (RDM2) and subtract non-connected component
            dm_nc = numpy.einsum(
                "ij,kl->ijkl", drdm1, drdm1, dtype=numpy.float64, optimize=True
            ) - 0.5 * numpy.einsum(
                "ij,kl->iklj", drdm1, drdm1, dtype=numpy.float64, optimize=True
            )
            fobjs.__rdm2 -= dm_nc

        # Generate the projection matrix
        cind = [fobjs.fsites[i] for i in fobjs.efac[1]]
        Pc_ = (
            fobjs.TA.T
            @ self.S
            @ self.W[:, cind]
            @ self.W[:, cind].T
            @ self.S
            @ fobjs.TA
        )

        if not only_rdm2:
            # Compute RDM1 in the localized orbital (LO) basis and transform to AO basis
            rdm1_eo = fobjs.mo_coeffs @ fobjs.__rdm1 @ fobjs.mo_coeffs.T
            rdm1_center = Pc_ @ rdm1_eo
            rdm1_ao = fobjs.TA @ rdm1_center @ fobjs.TA.T
            rdm1AO += rdm1_ao

        if not only_rdm1:
            # Transform RDM2 to AO basis
            rdm2s = numpy.einsum(
                "ijkl,pi,qj,rk,sl->pqrs",
                fobjs.__rdm2,
                *([fobjs.mo_coeffs] * 4),
                optimize=True,
            )
            rdm2_ao = numpy.einsum(
                "xi,ijkl,px,qj,rk,sl->pqrs",
                Pc_,
                rdm2s,
                fobjs.TA,
                fobjs.TA,
                fobjs.TA,
                fobjs.TA,
                optimize=True,
            )
            rdm2AO += rdm2_ao

    if not only_rdm1:
        # Symmetrize RDM2 and add the non-cumulant part if requested
        rdm2AO = (rdm2AO + rdm2AO.T) / 2.0
        if return_RDM2:
            nc_AO = (
                numpy.einsum(
                    "ij,kl->ijkl", rdm1AO, rdm1AO, dtype=numpy.float64, optimize=True
                )
                - numpy.einsum(
                    "ij,kl->iklj", rdm1AO, rdm1AO, dtype=numpy.float64, optimize=True
                )
                * 0.5
            )
            rdm2AO = nc_AO + rdm2AO

        # Transform RDM2 to the molecular orbital (MO) basis if needed
        if not return_ao:
            CmoT_S = self.C.T @ self.S
            rdm2MO = numpy.einsum(
                "ijkl,pi,qj,rk,sl->pqrs",
                rdm2AO,
                CmoT_S,
                CmoT_S,
                CmoT_S,
                CmoT_S,
                optimize=True,
            )

        # Transform RDM2 to the localized orbital (LO) basis if needed
        if return_lo:
            CloT_S = self.W.T @ self.S
            rdm2LO = numpy.einsum(
                "ijkl,pi,qj,rk,sl->pqrs",
                rdm2AO,
                CloT_S,
                CloT_S,
                CloT_S,
                CloT_S,
                optimize=True,
            )

    if not only_rdm2:
        # Symmetrize RDM1
        rdm1AO = (rdm1AO + rdm1AO.T) / 2.0

        # Transform RDM1 to the MO basis if needed
        if not return_ao:
            rdm1MO = self.C.T @ self.S @ rdm1AO @ self.S @ self.C

        # Transform RDM1 to the LO basis if needed
        if return_lo:
            rdm1LO = self.W.T @ self.S @ rdm1AO @ self.S @ self.W

    if return_RDM2 and print_energy:
        # Compute and print energy contributions
        Eh1 = numpy.einsum("ij,ij", self.hcore, rdm1AO, optimize=True)
        eri = ao2mo.restore(1, self.mf._eri, self.mf.mo_coeff.shape[1])
        E2 = 0.5 * numpy.einsum("pqrs,pqrs", eri, rdm2AO, optimize=True)
        print(flush=True)
        print("-----------------------------------------------------", flush=True)
        print(" BE ENERGIES with cumulant-based expression", flush=True)

        print("-----------------------------------------------------", flush=True)

        print(" 1-elec E        : {:>15.8f} Ha".format(Eh1), flush=True)
        print(" 2-elec E        : {:>15.8f} Ha".format(E2), flush=True)
        E_tot = Eh1 + E2 + self.E_core + self.enuc
        print(" E_BE            : {:>15.8f} Ha".format(E_tot), flush=True)
        print(
            " Ecorr BE        : {:>15.8f} Ha".format((E_tot) - self.ebe_hf), flush=True
        )
        print("-----------------------------------------------------", flush=True)
        print(flush=True)

    if only_rdm1:
        if return_ao:
            return rdm1AO
        else:
            return rdm1MO
    if only_rdm2:
        if return_ao:
            return rdm2AO
        else:
            return rdm2MO

    if return_lo and return_ao:
        return (rdm1AO, rdm2AO, rdm1LO, rdm2LO)
    if return_lo and not return_ao:
        return (rdm1MO, rdm2MO, rdm1LO, rdm2LO)

    if return_ao:
        return rdm1AO, rdm2AO
    if not return_ao:
        return rdm1MO, rdm2MO


def compute_energy_full(
    self, approx_cumulant=False, use_full_rdm=False, return_rdm=True
):
    """
    Compute the total energy using rdms in the full basis.

    Parameters
    ----------
    approx_cumulant : bool, optional
        If True, use an approximate cumulant for the energy computation. Default is False.
    use_full_rdm : bool, optional
        If True, use the full two-particle RDM for energy computation. Default is False.
    return_rdm : bool, optional
        If True, return the computed reduced density matrices (RDMs). Default is True.

    Returns
    -------
    tuple of numpy.ndarray or None
        If `return_rdm` is True, returns a tuple containing the one-particle and two-particle
        reduced density matrices (RDM1 and RDM2). Otherwise, returns None.

    Notes
    -----
    This function computes the total energy in the full basis, with options to use
    approximate or true cumulants, and to return the reduced density matrices (RDMs). The
    energy components are printed as part of the function's output.
    """

    from pyscf import scf, ao2mo

    # Compute the one-particle reduced density matrix (RDM1) and the cumulant (Kumul) in the full basis
    rdm1f, Kumul, rdm1_lo, rdm2_lo = self.rdm1_fullbasis(
        return_lo=True, return_RDM2=False
    )

    if not approx_cumulant:
        # Compute the true two-particle reduced density matrix (RDM2) if not using approximate cumulant
        Kumul_T = self.rdm1_fullbasis(only_rdm2=True)

    if return_rdm:
        # Construct the full RDM2 from RDM1
        RDM2_full = (
            numpy.einsum(
                "ij,kl->ijkl", rdm1f, rdm1f, dtype=numpy.float64, optimize=True
            )
            - numpy.einsum(
                "ij,kl->iklj", rdm1f, rdm1f, dtype=numpy.float64, optimize=True
            )
            * 0.5
        )

        # Add the cumulant part to RDM2
        if not approx_cumulant:
            RDM2_full += Kumul_T
        else:
            RDM2_full += Kumul

    # Compute the change in the one-particle density matrix (delta_gamma)
    del_gamma = rdm1f - self.hf_dm

    # Compute the effective potential
    veff = scf.hf.get_veff(self.mol, rdm1f, hermi=0)

    # Compute the one-electron energy
    Eh1 = numpy.einsum("ij,ij", self.hcore, rdm1f, optimize=True)

    # Compute the energy due to the effective potential
    EVeff = numpy.einsum("ij,ij", veff, rdm1f, optimize=True)

    # Compute the change in the one-electron energy
    Eh1_dg = numpy.einsum("ij,ij", self.hcore, del_gamma, optimize=True)

    # Compute the change in the effective potential energy
    Eveff_dg = numpy.einsum("ij,ij", self.hf_veff, del_gamma, optimize=True)

    # Restore the electron repulsion integrals (ERI)
    eri = ao2mo.restore(1, self.mf._eri, self.mf.mo_coeff.shape[1])

    # Compute the cumulant part of the two-electron energy
    EKumul = numpy.einsum("pqrs,pqrs", eri, Kumul, optimize=True)

    if not approx_cumulant:
        # Compute the true two-electron energy if not using approximate cumulant
        EKumul_T = numpy.einsum("pqrs,pqrs", eri, Kumul_T, optimize=True)

    if use_full_rdm and return_rdm:
        # Compute the full two-electron energy using the full RDM2
        E2 = numpy.einsum("pqrs,pqrs", eri, RDM2_full, optimize=True)

    # Compute the approximate BE total energy
    EKapprox = self.ebe_hf + Eh1_dg + Eveff_dg + EKumul / 2.0
    self.ebe_tot = EKapprox

    if not approx_cumulant:
        # Compute the true BE total energy if not using approximate cumulant
        EKtrue = Eh1 + EVeff / 2.0 + EKumul_T / 2.0 + self.enuc + self.E_core
        self.ebe_tot = EKtrue

    # Print energy results
    print("-----------------------------------------------------", flush=True)
    print(" BE ENERGIES with cumulant-based expression", flush=True)

    print("-----------------------------------------------------", flush=True)
    print(" E_BE = E_HF + Tr(F del g) + Tr(V K_approx)", flush=True)
    print(" E_HF            : {:>14.8f} Ha".format(self.ebe_hf), flush=True)
    print(" Tr(F del g)     : {:>14.8f} Ha".format(Eh1_dg + Eveff_dg), flush=True)
    print(" Tr(V K_aprrox)  : {:>14.8f} Ha".format(EKumul / 2.0), flush=True)
    print(" E_BE            : {:>14.8f} Ha".format(EKapprox), flush=True)
    print(" Ecorr BE        : {:>14.8f} Ha".format(EKapprox - self.ebe_hf), flush=True)

    if not approx_cumulant:
        print(flush=True)
        print(" E_BE = Tr(F[g] g) + Tr(V K_true)", flush=True)
        print(" Tr(h1 g)        : {:>14.8f} Ha".format(Eh1), flush=True)
        print(" Tr(Veff[g] g)   : {:>14.8f} Ha".format(EVeff / 2.0), flush=True)
        print(" Tr(V K_true)    : {:>14.8f} Ha".format(EKumul_T / 2.0), flush=True)
        print(" E_BE            : {:>14.8f} Ha".format(EKtrue), flush=True)
        if use_full_rdm and return_rdm:
            print(
                " E(g+G)          : {:>14.8f} Ha".format(
                    Eh1 + 0.5 * E2 + self.E_core + self.enuc
                ),
                flush=True,
            )
        print(
            " Ecorr BE        : {:>14.8f} Ha".format(EKtrue - self.ebe_hf), flush=True
        )
        print(flush=True)
        print(" True - approx   : {:>14.4e} Ha".format(EKtrue - EKapprox))
    print("-----------------------------------------------------", flush=True)

    print(flush=True)

    # Return the RDMs if requested
    if return_rdm:
        return (rdm1f, RDM2_full)

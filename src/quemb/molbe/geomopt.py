from pyscf import scf, lib
from pyscf.tools import finite_diff
import numpy as np

def energy_hf(mol):
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    return mf.e_tot

class Energy(lib.StreamObject):
    def __init__(self, mol, energy_func, displacement=1e-4):
        self.mol = mol
        self.energy_func = energy_func
        self.e_tot = None
        self.displacement = displacement

        # for finite_diff module assertions only, no effect
        self.conv_tol = 1e-12  
        self.converged = True

    def kernel(self, mol=None):
        if mol is not None:
            self.mol = mol

        self.e_tot = self.energy_func(self.mol)
        return self.e_tot

    def as_scanner(self):
        parent = self

        class Scanner(lib.SinglePointScanner, lib.StreamObject):

            def __call__(self, mol):

                parent.mol = mol
                parent.e_tot = parent.energy_func(mol)
                
                self.mol = mol
                self.e_tot = parent.e_tot

                return self.e_tot

        scanner = Scanner()
        scanner.mol = parent.mol

        # for finite_diff module assertions only, no effect
        scanner.conv_tol = 1e-12
        scanner.converged = True
        
        return scanner


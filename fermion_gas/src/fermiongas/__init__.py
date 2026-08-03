"""fermiongas -- trapped few-fermion mixtures via SAPT-style perturbation
theory, benchmarked against exact diagonalization.

Standalone (numpy/scipy). All integrals are plain numpy arrays so the same
tensors can later be handed to psi4 or any FCI/CC/SAPT backend.

Public API
----------
    TrapDVR, harmonic_oscillator        -- single-species 1D trap (DVR)
    contact_eri, inter_species_eri      -- contact 4-index integrals
    TwoComponentFCI, PT2Result          -- two-component FCI + SAPT-style PT2
    dispersion                          -- fixed-order & RPA dispersion
    rhf_contact, RHFResult              -- Hartree-Fock reference
    ccsd, CCSDResult                    -- psi4numpy-style spin-orbital CCSD
    write_fcidump                       -- export Hamiltonian for external codes
"""

from .trap import TrapDVR, harmonic_oscillator
from .contact import contact_eri, inter_species_eri
from .twocomponent import TwoComponentFCI, PT2Result
from .rpa import dispersion
from .hartreefock import rhf_contact, RHFResult
from .ccsd import ccsd, CCSDResult
from .fcidump import write_fcidump

__all__ = [
    "TrapDVR",
    "harmonic_oscillator",
    "contact_eri",
    "inter_species_eri",
    "TwoComponentFCI",
    "PT2Result",
    "dispersion",
    "rhf_contact",
    "RHFResult",
    "ccsd",
    "CCSDResult",
    "write_fcidump",
]

__version__ = "0.1.0"

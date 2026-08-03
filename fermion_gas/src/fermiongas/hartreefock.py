"""Closed-shell Hartree-Fock reference for the balanced two-component contact
gas (equal masses, N_up = N_dn = n_pair).

The contact interaction acts only between opposite spins, and same-spin
fermions do not interact, so the Fock operator has NO exchange term and NO
same-spin contribution -- each up electron feels only the Hartree field of the
down density (and vice versa):

    F = h + J[D],   J_pq = sum_rs (pq|rs) D_rs,   D = sum_{i occ} C_.i C_.i^T

with h = diag(non-interacting trap energies) and (pq|rs) = g * integral
phi_p phi_q phi_r phi_s dx (fully symmetric contact tensor). Providing canonical
HF orbitals (diagonal Fock) is what the CCSD module needs as its reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact import contact_eri


@dataclass
class RHFResult:
    e_hf: float           # total HF energy (absolute, incl. trap one-body)
    eps: np.ndarray       # canonical HF orbital energies, shape (n_orb,)
    C: np.ndarray         # HF coefficients in the DVR-orbital basis (n_orb, n_orb)
    mo_orbitals: np.ndarray   # HF orbitals sampled on the grid (n_grid, n_orb)
    eri_mo: np.ndarray    # (pq|rs) in the HF MO basis (n_orb,)*4, g folded in
    n_pair: int           # occupied orbitals per spin


def rhf_contact(
    energies: np.ndarray,
    orbitals: np.ndarray,
    dx: float,
    n_pair: int,
    g: float,
    max_iter: int = 200,
    e_conv: float = 1e-11,
) -> RHFResult:
    """Self-consistent closed-shell HF for the balanced contact gas."""
    energies = np.asarray(energies, float)
    n = energies.size
    h = np.diag(energies)
    eri = contact_eri(orbitals, dx, g)   # (pq|rs), g included

    C = np.eye(n)                        # trap orbitals are the initial guess
    D = C[:, :n_pair] @ C[:, :n_pair].T
    e_old = 0.0
    for _ in range(max_iter):
        J = np.einsum("pqrs,rs->pq", eri, D, optimize=True)
        F = h + J
        eps, C = np.linalg.eigh(F)
        D = C[:, :n_pair] @ C[:, :n_pair].T
        # E = 2 Tr(hD) + sum (pq|rs) D_pq D_rs   (up<->dn Hartree, no exchange)
        e_hf = 2.0 * np.einsum("pq,pq->", h, D) + np.einsum(
            "pqrs,pq,rs->", eri, D, D, optimize=True
        )
        if abs(e_hf - e_old) < e_conv:
            break
        e_old = e_hf

    mo_orbitals = orbitals @ C
    eri_mo = contact_eri(mo_orbitals, dx, g)
    return RHFResult(
        e_hf=float(e_hf), eps=eps, C=C, mo_orbitals=mo_orbitals,
        eri_mo=eri_mo, n_pair=n_pair,
    )

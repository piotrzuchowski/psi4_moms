r"""Resummed (RPA / ring) dispersion for the two-component contact gas.

Fixed-order 2nd-order dispersion is the leading term of an infinite series of
inter-spin "ring" diagrams. Summing them to all orders is the direct-RPA
correlation energy restricted to the inter-species channel (the only channel
here, since same-spin fermions do not contact-interact):

    E_disp^(20)  = -(1/2pi) \int_0^inf dw  Tr[ F_up(w) B F_dn(w) B^T ]
    E_disp^RPA   =  (1/2pi) \int_0^inf dw  Tr ln( 1 - M(w) ),   M = F_up B F_dn B^T

with the particle-hole response  F_sigma(w)_{ia} = 2 dE_ia / (dE_ia^2 + w^2)
and coupling matrix  B_{ia,jb} = g (ia|jb).

Because M(w) is similar to a symmetric positive-semidefinite matrix, its
eigenvalues are real and >= 0. The RPA integral is finite only while the
largest eigenvalue stays below 1; lambda_max(w) -> 1 is the collective
(pairing) instability -- the exact point beyond which the perturbation series,
at any order, cannot reach the true ground state. We monitor it.

The omega integral uses the substitution w = c tan(theta), theta in (0, pi/2),
with Gauss-Legendre nodes.
"""

from __future__ import annotations

import numpy as np


def _ph_space(energies: np.ndarray, n_occ: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Particle-hole list for a filled sea of n_occ orbitals.

    Returns (i_idx, a_idx, dE) with dE = eps_a - eps_i > 0.
    """
    occ = np.arange(n_occ)
    vir = np.arange(n_occ, energies.size)
    i_idx = np.repeat(occ, vir.size)
    a_idx = np.tile(vir, occ.size)
    dE = energies[a_idx] - energies[i_idx]
    return i_idx, a_idx, dE


def _coupling_matrix(eri: np.ndarray, ph_up, ph_dn, g: float) -> np.ndarray:
    """B_{ia,jb} = g (ia|jb) = g * eri[i,a,j,b]."""
    iu, au, _ = ph_up
    jd, bd, _ = ph_dn
    # B_{ia,jb} = g (ia|jb) via advanced indexing over the ph pairs
    B = g * eri[iu[:, None], au[:, None], jd[None, :], bd[None, :]]
    return B


def _quad_nodes(n: int, c: float) -> tuple[np.ndarray, np.ndarray]:
    r"""Gauss-Legendre nodes/weights for \int_0^inf f(w) dw via w = c tan(theta)."""
    t, w = np.polynomial.legendre.leggauss(n)          # on [-1, 1]
    theta = 0.25 * np.pi * (t + 1.0)                   # -> (0, pi/2)
    jac = 0.25 * np.pi * w                             # dtheta weight
    omega = c * np.tan(theta)
    domega = c / np.cos(theta) ** 2                    # dw/dtheta
    return omega, jac * domega


def dispersion(
    energies: np.ndarray,
    eri: np.ndarray,
    n_up: int,
    n_dn: int,
    g: float,
    n_omega: int = 48,
    c: float | None = None,
) -> dict:
    """Return fixed-order and RPA dispersion plus the RPA stability monitor.

    Keys: 'e20' (fixed order), 'rpa' (resummed), 'lambda_max' (largest
    eigenvalue of M over the frequency grid; >= 1 means RPA has gone unstable).
    """
    ph_up = _ph_space(energies, n_up)
    ph_dn = _ph_space(energies, n_dn)
    dE_up = ph_up[2]
    dE_dn = ph_dn[2]
    B = _coupling_matrix(eri, ph_up, ph_dn, g)         # (n_ph_up, n_ph_dn)

    if c is None:
        c = float(np.median(np.concatenate([dE_up, dE_dn])))
    omegas, weights = _quad_nodes(n_omega, c)

    e20 = 0.0
    e_rpa = 0.0
    lam_max = 0.0
    unstable = False
    for w, wt in zip(omegas, weights):
        f_up = 2.0 * dE_up / (dE_up**2 + w**2)
        f_dn = 2.0 * dE_dn / (dE_dn**2 + w**2)
        # M = F_up B F_dn B^T   (n_ph_up x n_ph_up); f_dn scales B's columns
        M = (f_up[:, None] * B) @ (f_dn[None, :] * B).T
        e20 += -wt * np.trace(M) / (2.0 * np.pi)
        # eigenvalues of M (real, symmetric-similar): use the symmetric form
        s = np.sqrt(f_up)
        Msym = (s[:, None] * B) @ (f_dn[None, :] * B).T * s[None, :]
        ev = np.linalg.eigvalsh(Msym)
        lam_max = max(lam_max, float(ev[-1]))
        if ev[-1] >= 1.0:
            unstable = True
            continue  # ln(1-lambda) diverges; skip this node's RPA contribution
        e_rpa += wt * np.sum(np.log1p(-ev)) / (2.0 * np.pi)

    return {
        "e20": e20,
        "rpa": e_rpa if not unstable else np.nan,
        "lambda_max": lam_max,
        "unstable": unstable,
    }

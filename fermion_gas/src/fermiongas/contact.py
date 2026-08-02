"""Contact-interaction integrals on a shared DVR grid.

A zero-range (delta) interaction g * delta(x - x') turns into the fully
symmetric four-index tensor

    (i j | k l) = g * integral phi_i(x) phi_j(x) phi_k(x) phi_l(x) dx

evaluated by grid quadrature. Two flavors:

  * intra-species  (single orbital set)          -> contact_eri(orb, dx, g)
  * inter-species  (species A pair, species B pair) -> inter_species_eri(...)

The inter-species tensor is the genuinely new object for the two-species /
SAPT program: it couples the two Fermi vacua and is the only place the two
species talk to each other.

Note on statistics: a single species of *spin-polarized* fermions does not
feel the contact interaction at all (the antisymmetric spatial wavefunction
vanishes at coincidence). The intra-species tensor is therefore only physical
for a two-component (two spin state) species; it is provided for completeness
and for building full-CI benchmarks.
"""

from __future__ import annotations

import numpy as np


def contact_eri(orbitals: np.ndarray, dx: float, g: float = 1.0) -> np.ndarray:
    """Single-species contact tensor (ij|kl) = g * int phi_i phi_j phi_k phi_l dx.

    Parameters
    ----------
    orbitals:
        (n_grid, n_orb) array of trap orbitals sampled on the grid.
    dx:
        Grid spacing (quadrature weight).
    g:
        Contact coupling strength.

    Returns
    -------
    (n_orb, n_orb, n_orb, n_orb) array, fully symmetric in all four indices.
    """
    return g * dx * np.einsum(
        "gi,gj,gk,gl->ijkl", orbitals, orbitals, orbitals, orbitals, optimize=True
    )


def inter_species_eri(
    orb_a: np.ndarray, orb_b: np.ndarray, dx: float, g_ab: float = 1.0
) -> np.ndarray:
    """Inter-species contact tensor coupling species A and B on a shared grid.

        (i_A j_A | k_B l_B) = g_ab * int phi^A_i phi^A_j phi^B_k phi^B_l dx

    Symmetric under i<->j and k<->l separately (but not A<->B unless the
    orbital sets are identical). Both orbital sets must live on the SAME grid
    with the SAME spacing dx -- essential when A and B have different masses
    (e.g. Li vs Er) and hence very different spatial extents.

    Returns
    -------
    (n_A, n_A, n_B, n_B) array.
    """
    if orb_a.shape[0] != orb_b.shape[0]:
        raise ValueError(
            "species A and B must be sampled on the same grid: "
            f"got {orb_a.shape[0]} vs {orb_b.shape[0]} grid points"
        )
    return g_ab * dx * np.einsum(
        "gi,gj,gk,gl->ijkl", orb_a, orb_a, orb_b, orb_b, optimize=True
    )

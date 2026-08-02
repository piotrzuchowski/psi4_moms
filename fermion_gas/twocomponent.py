"""Two-component (spin-up / spin-down) 1D fermions in a shared trap, contact
interaction between opposite spins only.

The many-body problem is solved two ways from the SAME interaction operator:

  * FCI    -- exact diagonalization in the full determinant basis;
  * PT2    -- Rayleigh-Schroedinger through 2nd order about the non-interacting
              trap reference, with the 2nd-order term split into
                E_ind  (single excitations = induction)   and
                E_disp (up-down double excitations = dispersion),
              i.e. the SAPT polarization decomposition
                E_int = E1[elst] + E2[ind + disp] + ...

Building both from one operator guarantees the FCI-vs-PT comparison is a clean
statement about the perturbation series, not about two different codes.

Determinants are stored as integer bitmasks (bit p set = orbital p occupied),
separately for the up and down sectors. The convention places the whole up
string to the left of the down string, so an up-transition and a down-
transition act independently and their fermion signs simply multiply.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _excite(mask: int, s: int, p: int) -> tuple[int, int] | None:
    """Apply a_p^dagger a_s to a bitmask determinant.

    Returns (sign, new_mask) or None if the move annihilates the state
    (s empty, or p already occupied with p != s).
    """
    if not (mask >> s) & 1:
        return None
    sign = -1 if bin(mask & ((1 << s) - 1)).count("1") & 1 else 1
    m = mask & ~(1 << s)
    if (m >> p) & 1:
        return None
    if bin(m & ((1 << p) - 1)).count("1") & 1:
        sign = -sign
    m |= 1 << p
    return sign, m


def _occ(mask: int) -> list[int]:
    out = []
    p = 0
    while mask:
        if mask & 1:
            out.append(p)
        mask >>= 1
        p += 1
    return out


@dataclass
class PT2Result:
    e0: float          # non-interacting reference energy
    e1: float          # 1st order  = electrostatics (Hartree)
    e_ind: float       # 2nd order, singles = induction
    e_disp: float      # 2nd order, up-down doubles = dispersion

    @property
    def e2(self) -> float:
        return self.e_ind + self.e_disp

    @property
    def total(self) -> float:
        return self.e0 + self.e1 + self.e2

    @property
    def e_int(self) -> float:
        """Interaction energy (relative to the non-interacting reference)."""
        return self.e1 + self.e2


class TwoComponentFCI:
    """Full determinant basis + interaction operator for N_up + N_dn fermions
    in ``n_orb`` trap orbitals, coupled by a contact interaction of unit
    strength (the coupling g is applied afterwards, since H = H0 + g * V)."""

    def __init__(
        self,
        energies: np.ndarray,
        eri: np.ndarray,
        n_up: int,
        n_dn: int,
    ) -> None:
        self.eps = np.asarray(energies, float)
        self.eri = np.asarray(eri, float)
        self.n_orb = self.eps.size
        self.n_up = n_up
        self.n_dn = n_dn
        self._build_basis()
        self._build_operators()

    # -- basis --------------------------------------------------------------

    def _build_basis(self) -> None:
        def masks(n_part):
            out = []
            for combo in combinations(range(self.n_orb), n_part):
                m = 0
                for o in combo:
                    m |= 1 << o
                out.append(m)
            return out

        self.up_masks = masks(self.n_up)
        self.dn_masks = masks(self.n_dn)
        self.dim = len(self.up_masks) * len(self.dn_masks)
        # Reference = lowest-energy determinant (orbitals are energy-ordered).
        self.up_ref = self.up_masks[0]
        self.dn_ref = self.dn_masks[0]
        self._up_index = {m: i for i, m in enumerate(self.up_masks)}
        self._dn_index = {m: i for i, m in enumerate(self.dn_masks)}

    def _idx(self, u_mask: int, d_mask: int) -> int:
        return self._up_index[u_mask] * len(self.dn_masks) + self._dn_index[d_mask]

    # -- operators ----------------------------------------------------------

    def _diag_h0(self, u_mask: int, d_mask: int) -> float:
        return float(self.eps[_occ(u_mask)].sum() + self.eps[_occ(d_mask)].sum())

    def _build_operators(self) -> None:
        """Non-interacting diagonal H0 and the unit-strength interaction V."""
        n = self.dim
        self.h0_diag = np.empty(n)
        for u in self.up_masks:
            for d in self.dn_masks:
                self.h0_diag[self._idx(u, d)] = self._diag_h0(u, d)

        # H_int = sum_{pqrs} <phi_p phi_q phi_r phi_s> a+_{p up} a+_{q dn} a_{r dn} a_{s up}
        # up part: a+_p a_s  (annihilate s, create p);  dn part: a+_q a_r.
        V = np.zeros((n, n))
        eri = self.eri
        for u in self.up_masks:
            occ_u = _occ(u)
            for d in self.dn_masks:
                occ_d = _occ(d)
                col = self._idx(u, d)
                for s in occ_u:
                    for p in range(self.n_orb):
                        up_move = _excite(u, s, p)
                        if up_move is None:
                            continue
                        sgn_u, u2 = up_move
                        for r in occ_d:
                            for q in range(self.n_orb):
                                dn_move = _excite(d, r, q)
                                if dn_move is None:
                                    continue
                                sgn_d, d2 = dn_move
                                row = self._idx(u2, d2)
                                V[row, col] += eri[p, q, r, s] * sgn_u * sgn_d
        self.V = V
        self.ref = self._idx(self.up_ref, self.dn_ref)

    # -- solvers ------------------------------------------------------------

    def fci(self, g: float) -> float:
        """Exact ground-state energy for coupling g (lowest eigenvalue only)."""
        H = np.diag(self.h0_diag) + g * self.V
        try:
            from scipy.linalg import eigh
            return float(eigh(H, subset_by_index=(0, 0), eigvals_only=True)[0])
        except Exception:
            return float(np.linalg.eigvalsh(H)[0])

    def pt2(self, g: float) -> PT2Result:
        """Rayleigh-Schroedinger through 2nd order about the trap reference,
        with the SAPT-style ind/disp split."""
        ref = self.ref
        e0 = self.h0_diag[ref]
        e1 = g * self.V[ref, ref]

        col = g * self.V[:, ref]
        e_ind = 0.0
        e_disp = 0.0
        for j in range(self.dim):
            if j == ref:
                continue
            v = col[j]
            if v == 0.0:
                continue
            denom = e0 - self.h0_diag[j]
            contrib = v * v / denom
            # classify j vs reference by excitation character
            u_j = self.up_masks[j // len(self.dn_masks)]
            d_j = self.dn_masks[j % len(self.dn_masks)]
            n_up_h = bin(self.up_ref & ~u_j).count("1")
            n_dn_h = bin(self.dn_ref & ~d_j).count("1")
            if n_up_h + n_dn_h == 1:
                e_ind += contrib          # single excitation -> induction
            else:
                e_disp += contrib         # up-down double     -> dispersion
        return PT2Result(e0=e0, e1=e1, e_ind=e_ind, e_disp=e_disp)


if __name__ == "__main__":
    # Quick self-test: g -> 0 consistency and 1+1 vs 2+2 smoke test.
    from trap import TrapDVR
    from contact import contact_eri

    trap = TrapDVR(n_orb=8)
    eri = contact_eri(trap.orbitals, trap.dx)

    for (nu, nd) in [(1, 1), (2, 2)]:
        solver = TwoComponentFCI(trap.energies, eri, nu, nd)
        e_ni = solver.h0_diag[solver.ref]
        for g in (0.0, 0.2, -0.2):
            fci = solver.fci(g)
            pt = solver.pt2(g)
            print(
                f"N={nu}+{nd}  g={g:+.2f}  "
                f"E_int(FCI)={fci - e_ni:+.5f}  "
                f"E_int(PT2)={pt.e_int:+.5f}  "
                f"[elst={pt.e1:+.4f} ind={pt.e_ind:+.4f} disp={pt.e_disp:+.4f}]"
            )
        print()

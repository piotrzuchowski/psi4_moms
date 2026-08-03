"""Spin-orbital CCSD, psi4numpy-style, fed the fermiongas integrals.

This is the standard Stanton-Gauss-Watts-Bartlett (1991) spin-orbital CCSD
formulation used in the psi4numpy Coupled-Cluster tutorials. The ONLY change
from those tutorials is the integral source: instead of
``psi4.core.MintsHelper`` we build the antisymmetrized two-electron integrals
from a fermiongas Hartree-Fock reference (``hartreefock.rhf_contact``). The
amplitude equations are unchanged, so results are directly comparable to a
psi4numpy CCSD run on the same Hamiltonian.

Spin ordering is "spin-blocked": spin-orbitals 0..n-1 are the alpha (up)
spatial orbitals, n..2n-1 the beta (down) ones. Because the contact interaction
is spin-independent and local, same-spin antisymmetrized integrals vanish
automatically and only up-down terms survive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hartreefock import rhf_contact


@dataclass
class CCSDResult:
    e_hf: float
    e_corr: float
    converged: bool
    n_iter: int

    @property
    def e_total(self) -> float:
        return self.e_hf + self.e_corr


def _antisymmetrized_spin_orbitals(eps: np.ndarray, eri_mo: np.ndarray):
    """Build spin-orbital Fock diagonal and <PQ||RS> from spatial HF data.

    <PQ||RS> = (spatial) * ( d(sP,sR) d(sQ,sS) - d(sP,sS) d(sQ,sR) ),
    using that the contact spatial integral is fully symmetric.
    """
    n = eps.size
    nso = 2 * n
    spat = np.arange(nso) % n
    spin = np.arange(nso) // n

    f_so = np.concatenate([eps, eps])  # alpha block then beta block

    E = eri_mo[np.ix_(spat, spat, spat, spat)]
    s = spin
    dPR = (s[:, None, None, None] == s[None, None, :, None])
    dQS = (s[None, :, None, None] == s[None, None, None, :])
    dPS = (s[:, None, None, None] == s[None, None, None, :])
    dQR = (s[None, :, None, None] == s[None, None, :, None])
    G = E * (dPR * dQS).astype(float) - E * (dPS * dQR).astype(float)  # <PQ||RS>

    # Reorder spin-orbitals by energy so the occupied ones come first
    # (spin-blocked ordering interleaves occ/virt across the alpha/beta blocks).
    perm = np.argsort(f_so, kind="stable")
    f_so = f_so[perm]
    G = G[np.ix_(perm, perm, perm, perm)]
    return f_so, G


def ccsd(
    ref: "RHFResult",
    max_iter: int = 200,
    e_conv: float = 1e-10,
    damp: float = 0.0,
) -> CCSDResult:
    """Spin-orbital CCSD on a fermiongas RHF reference.

    Parameters
    ----------
    ref : RHFResult from hartreefock.rhf_contact
    damp : optional linear damping in [0,1) on the amplitudes (helps near the
           strong-coupling/pairing regime where CCSD is stiff).
    """
    eps_so, G = _antisymmetrized_spin_orbitals(ref.eps, ref.eri_mo)
    nocc = 2 * ref.n_pair
    nso = eps_so.size
    o = slice(0, nocc)
    v = slice(nocc, nso)

    eo = eps_so[o]
    ev = eps_so[v]
    Dia = eo[:, None] - ev[None, :]
    Dijab = (eo[:, None, None, None] + eo[None, :, None, None]
             - ev[None, None, :, None] - ev[None, None, None, :])

    Voovv = G[o, o, v, v]
    t1 = np.zeros((nocc, nso - nocc))
    t2 = Voovv / Dijab

    def energy(t1, t2):
        e = np.einsum("ijab,ijab->", Voovv, t2, optimize=True) * 0.25
        e += 0.5 * np.einsum("ijab,ia,jb->", Voovv, t1, t1, optimize=True)
        return e

    def tau(t1, t2):
        return t2 + np.einsum("ia,jb->ijab", t1, t1, optimize=True) \
                  - np.einsum("ib,ja->ijab", t1, t1, optimize=True)

    def tau_t(t1, t2):
        return t2 + 0.5 * (np.einsum("ia,jb->ijab", t1, t1, optimize=True)
                           - np.einsum("ib,ja->ijab", t1, t1, optimize=True))

    Goovv = G[o, o, v, v]
    Govov = G[o, v, o, v]
    Goooo = G[o, o, o, o]
    Gvvvv = G[v, v, v, v]
    Gooov = G[o, o, o, v]
    Govvv = G[o, v, v, v]
    Gvovv = G[v, o, v, v]
    Govvo = G[o, v, v, o]
    Govoo = G[o, v, o, o]
    Gvvvo = G[v, v, v, o]
    Goovo = G[o, o, v, o]

    e_old = energy(t1, t2)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        tt = tau_t(t1, t2)
        ta = tau(t1, t2)

        # --- two-body-like intermediates (Stanton eqs. 3-8) ---
        Fae = (-0.5 * np.einsum("mnaf,mnef->ae", tt, Goovv, optimize=True)
               + np.einsum("mf,mafe->ae", t1, Govvv, optimize=True))
        Fmi = (0.5 * np.einsum("inef,mnef->mi", tt, Goovv, optimize=True)
               + np.einsum("ne,mnie->mi", t1, Gooov, optimize=True))
        Fme = np.einsum("nf,mnef->me", t1, Goovv, optimize=True)

        Wmnij = (Goooo
                 + np.einsum("je,mnie->mnij", t1, Gooov, optimize=True)
                 - np.einsum("ie,mnje->mnij", t1, Gooov, optimize=True)
                 + 0.25 * np.einsum("ijef,mnef->mnij", ta, Goovv, optimize=True))
        tmp = np.einsum("mb,amef->abef", t1, Gvovv, optimize=True)
        Wabef = (Gvvvv - tmp + tmp.transpose(1, 0, 2, 3)
                 + 0.25 * np.einsum("mnab,mnef->abef", ta, Goovv, optimize=True))
        Wmbej = (Govvo
                 + np.einsum("jf,mbef->mbej", t1, Govvv, optimize=True)
                 - np.einsum("nb,mnej->mbej", t1, Goovo, optimize=True)
                 - np.einsum("jnfb,mnef->mbej", t2 * 0.5 + np.einsum("jf,nb->jnfb", t1, t1, optimize=True),
                             Goovv, optimize=True))

        # --- T1 (Stanton eq. 1) ---
        t1new = (np.einsum("ie,ae->ia", t1, Fae, optimize=True)
                 - np.einsum("ma,mi->ia", t1, Fmi, optimize=True)
                 + np.einsum("imae,me->ia", t2, Fme, optimize=True)
                 - np.einsum("nf,naif->ia", t1, G[o, v, o, v], optimize=True)
                 - 0.5 * np.einsum("imef,maef->ia", t2, Govvv, optimize=True)
                 - 0.5 * np.einsum("mnae,nmei->ia", t2, G[o, o, v, o], optimize=True))
        t1new /= Dia

        # --- T2 (Stanton eq. 2) ---
        Pab = lambda X: X - X.transpose(0, 1, 3, 2)
        Pij = lambda X: X - X.transpose(1, 0, 2, 3)

        t2new = Goovv.copy()
        tmp = np.einsum("ijae,be->ijab", t2, Fae - 0.5 * np.einsum("mb,me->be", t1, Fme, optimize=True), optimize=True)
        t2new += Pab(tmp)
        tmp = np.einsum("imab,mj->ijab", t2, Fmi + 0.5 * np.einsum("je,me->mj", t1, Fme, optimize=True), optimize=True)
        t2new -= Pij(tmp)
        t2new += 0.5 * np.einsum("mnab,mnij->ijab", ta, Wmnij, optimize=True)
        t2new += 0.5 * np.einsum("ijef,abef->ijab", ta, Wabef, optimize=True)
        tmp = (np.einsum("imae,mbej->ijab", t2, Wmbej, optimize=True)
               - np.einsum("ie,ma,mbej->ijab", t1, t1, Govvo, optimize=True))
        t2new += Pij(Pab(tmp))
        tmp = np.einsum("ie,abej->ijab", t1, Gvvvo, optimize=True)
        t2new += Pij(tmp)
        tmp = np.einsum("ma,mbij->ijab", t1, Govoo, optimize=True)
        t2new -= Pab(tmp)
        t2new /= Dijab

        if damp > 0.0:
            t1 = damp * t1 + (1 - damp) * t1new
            t2 = damp * t2 + (1 - damp) * t2new
        else:
            t1, t2 = t1new, t2new

        e_new = energy(t1, t2)
        if abs(e_new - e_old) < e_conv:
            converged = True
            e_old = e_new
            break
        e_old = e_new

    return CCSDResult(e_hf=ref.e_hf, e_corr=float(e_old),
                      converged=converged, n_iter=it)

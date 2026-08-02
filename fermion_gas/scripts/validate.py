"""Consolidated sanity checks for the fermiongas package. Run:

    python scripts/validate.py

Covers:
  1-4. Analytic harmonic-oscillator checks for the DVR + contact integrals.
  5.   Two-species smoke test (mass imbalance on a shared grid).
  6.   Two-component FCI vs PT2 consistency at small coupling.
  7.   RPA dispersion reduces to fixed-order, which matches the explicit sum.
"""

import numpy as np

from fermiongas import TrapDVR, contact_eri, inter_species_eri, TwoComponentFCI
from fermiongas.rpa import dispersion, _ph_space, _coupling_matrix


def main() -> None:
    ok = True
    trap = TrapDVR(n_orb=12)

    # 1) Orbital energies -> 0.5, 1.5, 2.5, ...
    expected = 0.5 + np.arange(trap.n_orb)
    err = np.max(np.abs(trap.energies - expected))
    print("1) HO energies (first 6):", np.round(trap.energies[:6], 6))
    print(f"   max |E_dvr - (n+1/2)| = {err:.2e}")
    ok &= err < 1e-6

    # 2) Contact self-integral (00|00) = 1/sqrt(2 pi).
    eri = contact_eri(trap.orbitals, trap.dx)
    analytic = 1.0 / np.sqrt(2.0 * np.pi)
    print(f"2) (00|00) dvr = {eri[0,0,0,0]:.8f}   analytic = {analytic:.8f}")
    ok &= abs(eri[0, 0, 0, 0] - analytic) < 1e-6

    # 3) Parity zero.
    print(f"3) (00|01) = {eri[0,0,0,1]:.2e}  (should be ~0)")
    ok &= abs(eri[0, 0, 0, 1]) < 1e-8

    # 4) Full four-index symmetry.
    sym = max(
        np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))),
        np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))),
        np.max(np.abs(eri - eri.transpose(3, 1, 2, 0))),
    )
    print(f"4) contact-tensor asymmetry = {sym:.2e}")
    ok &= sym < 1e-12

    # 5) Two-species shared grid (mass 10).
    heavy = TrapDVR(mass=10.0, n_orb=6)
    err_h = np.max(np.abs(heavy.energies - (0.5 + np.arange(6)) / np.sqrt(10.0)))
    v = inter_species_eri(trap.orbitals, heavy.orbitals, trap.dx)
    print(f"5) heavy(m=10) energy err = {err_h:.2e}; inter tensor {v.shape}")
    ok &= err_h < 1e-5

    # 6) Two-component FCI vs PT2 at small coupling.
    trap8 = TrapDVR(n_orb=8)
    eri8 = contact_eri(trap8.orbitals, trap8.dx)
    solver = TwoComponentFCI(trap8.energies, eri8, 2, 2)
    e_ni = solver.h0_diag[solver.ref]
    g = 0.1
    d = abs((solver.fci(g) - e_ni) - solver.pt2(g).e_int)
    print(f"6) 2+2 at g={g}: |E_int(FCI) - PT2| = {d:.2e}")
    ok &= d < 1e-3

    # 7) RPA -> fixed-order -> explicit sum.
    res = dispersion(trap8.energies, eri8, 2, 2, 0.1, n_omega=64)
    ph_up = _ph_space(trap8.energies, 2)
    ph_dn = _ph_space(trap8.energies, 2)
    B = _coupling_matrix(eri8, ph_up, ph_dn, 0.1)
    e20_sum = -np.sum(B**2 / (ph_up[2][:, None] + ph_dn[2][None, :]))
    print(f"7) disp: e20(quad)={res['e20']:.6e}  e20(sum)={e20_sum:.6e}  "
          f"rpa={res['rpa']:.6e}")
    ok &= abs(res["e20"] - e20_sum) < 1e-8

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()

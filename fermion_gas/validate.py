"""Sanity checks for the standalone DVR + contact-integral core against
analytic harmonic-oscillator results. Run:  python validate.py
"""

import numpy as np

from trap import TrapDVR
from contact import contact_eri, inter_species_eri


def main() -> None:
    trap = TrapDVR(n_orb=12)  # default 1/2 x^2 trap, mass 1
    ok = True

    # 1) Orbital energies should be 0.5, 1.5, 2.5, ... (hbar omega = 1).
    expected = 0.5 + np.arange(trap.n_orb)
    err = np.max(np.abs(trap.energies - expected))
    print("1) HO energies (first 6):", np.round(trap.energies[:6], 6))
    print("   max |E_dvr - (n+1/2)| =", f"{err:.2e}")
    ok &= err < 1e-6

    # 2) Contact self-integral (00|00) = int phi_0^4 dx.
    #    phi_0 = pi^-1/4 exp(-x^2/2)  ->  phi_0^4 = pi^-1 exp(-2 x^2)
    #    int = pi^-1 * sqrt(pi/2) = 1/sqrt(2 pi) = 0.3989422804...
    eri = contact_eri(trap.orbitals, trap.dx)
    analytic = 1.0 / np.sqrt(2.0 * np.pi)
    print(f"\n2) (00|00) dvr = {eri[0,0,0,0]:.8f}   analytic = {analytic:.8f}")
    ok &= abs(eri[0, 0, 0, 0] - analytic) < 1e-6

    # 3) Parity: (00|01) must vanish (odd integrand) for the symmetric trap.
    print(f"3) (00|01) = {eri[0,0,0,1]:.2e}  (should be ~0 by parity)")
    ok &= abs(eri[0, 0, 0, 1]) < 1e-8

    # 4) Full four-index symmetry of the contact tensor.
    sym = max(
        np.max(np.abs(eri - eri.transpose(1, 0, 2, 3))),
        np.max(np.abs(eri - eri.transpose(0, 1, 3, 2))),
        np.max(np.abs(eri - eri.transpose(2, 3, 0, 1))),
        np.max(np.abs(eri - eri.transpose(3, 1, 2, 0))),
    )
    print(f"4) max asymmetry of contact tensor = {sym:.2e}  (should be ~0)")
    ok &= sym < 1e-12

    # 5) Two-species smoke test: a heavier species (mass 10) shares the grid.
    #    Its level spacing is omega = 1/sqrt(m), so E_n = (n+1/2)/sqrt(10).
    heavy = TrapDVR(mass=10.0, n_orb=6)
    expected_h = (0.5 + np.arange(6)) / np.sqrt(10.0)
    err_h = np.max(np.abs(heavy.energies - expected_h))
    print(f"\n5) heavy (m=10) energies err = {err_h:.2e}  spacing = "
          f"{heavy.energies[1]-heavy.energies[0]:.4f}  (expect {1/np.sqrt(10):.4f})")
    ok &= err_h < 1e-5

    v = inter_species_eri(trap.orbitals, heavy.orbitals, trap.dx)
    print(f"   inter-species tensor shape = {v.shape}  (0000) = {v[0,0,0,0]:.6f}")

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")


if __name__ == "__main__":
    main()

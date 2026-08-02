"""Fixed-order vs resummed (RPA) dispersion against exact correlation.

Three curves for the beyond-mean-field correlation energy E_corr = E_int - E1
of the two-component 1D Fermi gas:

  * FCI          -- exact:  E_int(FCI) - E1[elst]
  * fixed order  -- induction + 2nd-order dispersion   (SAPT PT2)
  * RPA          -- induction + ring-resummed dispersion

The RPA stability eigenvalue lambda_max(g) is overlaid: where it approaches 1
the resummation (and any finite-order PT) breaks down -- the collective
instability that marks the edge of the perturbative regime.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fermiongas import TrapDVR, contact_eri, TwoComponentFCI, dispersion

# --- system ---------------------------------------------------------------
N_UP, N_DN = 3, 3
N_ORB = 8
G_GRID = np.linspace(-2.5, 5.0, 31)

trap = TrapDVR(n_orb=N_ORB)
eri = contact_eri(trap.orbitals, trap.dx)
solver = TwoComponentFCI(trap.energies, eri, N_UP, N_DN)
e_ni = solver.h0_diag[solver.ref]

fci_corr, fixed_corr, rpa_corr, lam = [], [], [], []
disp_check_ok = True
for g in G_GRID:
    pt = solver.pt2(g)
    e_int_fci = solver.fci(g) - e_ni
    disp = dispersion(trap.energies, eri, N_UP, N_DN, g, n_omega=48)

    # consistency: fixed-order dispersion two independent ways
    if abs(pt.e_disp - disp["e20"]) > 1e-6 * (1 + abs(pt.e_disp)):
        disp_check_ok = False

    fci_corr.append(e_int_fci - pt.e1)          # exact beyond-mean-field
    fixed_corr.append(pt.e_ind + disp["e20"])   # induction + fixed disp
    rpa_val = pt.e_ind + disp["rpa"]            # induction + RPA disp
    rpa_corr.append(rpa_val)
    lam.append(disp["lambda_max"])

fci_corr = np.array(fci_corr)
fixed_corr = np.array(fixed_corr)
rpa_corr = np.array(rpa_corr)
lam = np.array(lam)

print("fixed-order dispersion matches determinant PT2:",
      "OK" if disp_check_ok else "MISMATCH")

# --- figure ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.axhline(0, color="0.85", lw=0.8)
ax.axvline(0, color="0.85", lw=0.8)
ax.plot(G_GRID, fci_corr, "k-", lw=2.4, label=r"FCI  $E_{\mathrm{int}}-E^{(1)}$ (exact)")
ax.plot(G_GRID, fixed_corr, "--", color="#d62728", lw=1.9,
        label=r"fixed order  ind + disp$^{(20)}$")
ax.plot(G_GRID, rpa_corr, "-", color="#1f77b4", lw=1.9,
        label=r"RPA  ind + disp$^{\mathrm{RPA}}$")
ax.set_xlabel(r"contact coupling  $g$")
ax.set_ylabel(r"correlation energy  $E_{\mathrm{corr}}$  (trap units)")
ax.set_title(f"Fixed-order vs resummed dispersion  (N={N_UP}+{N_DN}, {N_ORB} orbitals)")
ax.legend(frameon=False, fontsize=9, loc="lower left")

# overlay the RPA stability eigenvalue
ax2 = ax.twinx()
ax2.plot(G_GRID, lam, ":", color="#7f7f7f", lw=1.6)
ax2.axhline(1.0, color="#7f7f7f", lw=0.9, ls=(0, (1, 1)))
ax2.set_ylabel(r"RPA  $\lambda_{\max}$  (dotted; =1 : instability)", color="#5f5f5f")
ax2.tick_params(axis="y", labelcolor="#5f5f5f")
ax2.set_ylim(0, max(1.15, float(np.nanmax(lam)) * 1.1))

fig.tight_layout()
out = "rpa_vs_fixed.png"
fig.savefig(out, dpi=150)
print("wrote", out)

print("\n   g     FCI_corr   fixed     RPA      lam_max")
for k in range(0, len(G_GRID), 3):
    print(f" {G_GRID[k]:+.2f}  {fci_corr[k]:+.5f}  {fixed_corr[k]:+.5f}  "
          f"{rpa_corr[k]:+.5f}  {lam[k]:.3f}")

"""ED vs 2nd-order perturbation theory for the two-component 1D Fermi gas.

Produces the core figure: interaction energy E_int(g) from exact
diagonalization (FCI) compared with the SAPT polarization expansion truncated
at 1st order (electrostatics) and 2nd order (elst + induction + dispersion),
plus the component breakdown. Shows where fixed-order PT peels away from the
truth on both the repulsive (fermionization) and attractive (pairing) wings.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fermiongas import TrapDVR, contact_eri, TwoComponentFCI

# --- system ---------------------------------------------------------------
N_UP, N_DN = 2, 2
N_ORB = 8
G_GRID = np.linspace(-3.0, 3.0, 49)

trap = TrapDVR(n_orb=N_ORB)
eri = contact_eri(trap.orbitals, trap.dx)
solver = TwoComponentFCI(trap.energies, eri, N_UP, N_DN)
e_ni = solver.h0_diag[solver.ref]  # non-interacting reference

fci, first, second = [], [], []
elst, ind, disp = [], [], []
for g in G_GRID:
    fci.append(solver.fci(g) - e_ni)
    pt = solver.pt2(g)
    first.append(pt.e1)
    second.append(pt.e_int)
    elst.append(pt.e1)
    ind.append(pt.e_ind)
    disp.append(pt.e_disp)

fci = np.array(fci)
first = np.array(first)
second = np.array(second)

# --- figure ---------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

ax1.axhline(0, color="0.8", lw=0.8)
ax1.axvline(0, color="0.8", lw=0.8)
ax1.plot(G_GRID, fci, "k-", lw=2.4, label="FCI (exact)")
ax1.plot(G_GRID, first, "--", color="#1f77b4", lw=1.8, label=r"1st order (elst)")
ax1.plot(G_GRID, second, "-.", color="#d62728", lw=1.8, label=r"2nd order (elst+ind+disp)")
ax1.set_xlabel(r"contact coupling  $g$")
ax1.set_ylabel(r"interaction energy  $E_{\mathrm{int}}$  (trap units)")
ax1.set_title(f"Two-component Fermi gas  N={N_UP}+{N_DN},  {N_ORB} orbitals")
ax1.legend(frameon=False, fontsize=9, loc="upper left")
ax1.text(-2.9, ax1.get_ylim()[0] * 0.82, "attractive\n(pairing)", fontsize=8,
         color="0.4", ha="left", va="bottom")
ax1.text(2.9, ax1.get_ylim()[1] * 0.82, "repulsive\n(fermionization)", fontsize=8,
         color="0.4", ha="right", va="top")

ax2.axhline(0, color="0.8", lw=0.8)
ax2.axvline(0, color="0.8", lw=0.8)
ax2.plot(G_GRID, elst, color="#1f77b4", lw=1.8, label="electrostatics  $E^{(1)}$")
ax2.plot(G_GRID, ind, color="#2ca02c", lw=1.8, label="induction  $E^{(2)}_{ind}$")
ax2.plot(G_GRID, disp, color="#ff7f0e", lw=1.8, label="dispersion  $E^{(2)}_{disp}$")
ax2.plot(G_GRID, second, "k-.", lw=1.4, label="PT2 total")
ax2.set_xlabel(r"contact coupling  $g$")
ax2.set_ylabel(r"SAPT component  (trap units)")
ax2.set_title("SAPT polarization decomposition")
ax2.legend(frameon=False, fontsize=9, loc="lower left")

fig.tight_layout()
out = "ed_vs_pt2.png"
fig.savefig(out, dpi=150)
print("wrote", out)

# --- also print a small table of the breakdown ---------------------------
print("\n   g      E_int(FCI)   PT2 total    elst      ind       disp     |err|")
for g in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0):
    pt = solver.pt2(g)
    f = solver.fci(g) - e_ni
    print(f" {g:+.2f}   {f:+.5f}    {pt.e_int:+.5f}   {pt.e1:+.4f}  "
          f"{pt.e_ind:+.4f}  {pt.e_disp:+.4f}   {abs(f-pt.e_int):.4f}")

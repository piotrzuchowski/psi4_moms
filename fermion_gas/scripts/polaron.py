"""The Fermi-polaron limit: one impurity (spin-down) in a sea of N spin-up
fermions, via the same DVR + contact + FCI/PT machinery.

Shows the polaron energy decomposed into the SAPT terms (mean-field /
induction / dispersion) versus exact FCI, together with the quasiparticle
residue  Z = |<Phi_0|Psi>|^2  (weight of the bare-impurity state in the
dressed ground state). Z -> 0 on the attractive side marks the
polaron -> molecule crossover: a particle-particle (pairing) event that
finite-order perturbation theory -- and particle-hole RPA -- cannot follow.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fermiongas import TrapDVR, contact_eri, TwoComponentFCI

# --- system: N-up sea + single down impurity ------------------------------
N_SEA = 4
N_ORB = 10
G_GRID = np.linspace(-4.0, 3.0, 43)

trap = TrapDVR(n_orb=N_ORB)
eri = contact_eri(trap.orbitals, trap.dx)
solver = TwoComponentFCI(trap.energies, eri, N_SEA, 1)  # N_up = sea, N_dn = 1
e_ni = solver.h0_diag[solver.ref]
ref = solver.ref
H0 = np.diag(solver.h0_diag)

e_fci, z_res = [], []
e_elst, e_ind, e_disp, e_pt2 = [], [], [], []
for g in G_GRID:
    # full ground state (need the eigenvector for the residue)
    w, V = np.linalg.eigh(H0 + g * solver.V)
    e_fci.append(w[0] - e_ni)
    z_res.append(float(V[ref, 0] ** 2))

    pt = solver.pt2(g)
    e_elst.append(pt.e1)
    e_ind.append(pt.e_ind)
    e_disp.append(pt.e_disp)
    e_pt2.append(pt.e_int)

e_fci = np.array(e_fci)
e_pt2 = np.array(e_pt2)
z_res = np.array(z_res)

# --- figure ---------------------------------------------------------------
fig, (axE, axZ) = plt.subplots(1, 2, figsize=(11, 4.4))

axE.axhline(0, color="0.85", lw=0.8)
axE.axvline(0, color="0.85", lw=0.8)
axE.plot(G_GRID, e_fci, "k-", lw=2.4, label="FCI polaron energy")
axE.plot(G_GRID, e_elst, "--", color="#1f77b4", lw=1.7, label=r"mean field  $E^{(1)}$")
axE.plot(G_GRID, e_pt2, "-.", color="#d62728", lw=1.8, label=r"PT2  (mf+ind+disp)")
axE.set_xlabel(r"impurity-sea coupling  $g$")
axE.set_ylabel(r"polaron energy  $E_p$  (trap units)")
axE.set_title(f"Fermi polaron: 1 impurity in {N_SEA}-fermion sea")
axE.legend(frameon=False, fontsize=9, loc="upper left")

axZ.axvline(0, color="0.85", lw=0.8)
axZ.plot(G_GRID, z_res, color="#2ca02c", lw=2.2)
axZ.fill_between(G_GRID, 0, z_res, color="#2ca02c", alpha=0.12)
axZ.set_xlabel(r"impurity-sea coupling  $g$")
axZ.set_ylabel(r"quasiparticle residue  $Z=|\langle\Phi_0|\Psi\rangle|^2$")
axZ.set_title("Residue: bare-impurity weight in dressed state")
axZ.set_ylim(0, 1.02)
axZ.annotate("polaron\n(well dressed)", xy=(1.0, 0.9), fontsize=8, color="0.4", ha="center")
axZ.annotate("molecule\ncrossover", xy=(-3.4, 0.25), fontsize=8, color="0.4", ha="center")

fig.tight_layout()
out = "polaron.png"
fig.savefig(out, dpi=150)
print("wrote", out)

print("\n   g     E_p(FCI)   PT2      mf       ind      disp      Z")
for k in range(0, len(G_GRID), 4):
    print(f" {G_GRID[k]:+.2f}  {e_fci[k]:+.5f}  {e_pt2[k]:+.5f}  {e_elst[k]:+.4f}  "
          f"{e_ind[k]:+.4f}  {e_disp[k]:+.4f}  {z_res[k]:.3f}")

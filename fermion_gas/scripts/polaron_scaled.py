"""Experiment-comparable polaron curves: E_p/E_F and residue Z versus a
dimensionless coupling, so the toy 1D model can be read against measured
Fermi-polaron data (which is reported in units of E_F and 1/(k_F a)).

Conventions (hbar = m = 1):
  E_F  = highest occupied sea single-particle level  (Fermi energy)
  k_F  = sqrt(2 E_F)
  1D scattering length for V = g delta(x):  a_1D = 2/|g|  (reduced mass 1/2)
  dimensionless coupling shown two ways:  eta = g / E_F   and  1/(k_F a_1D)

Honest caveat baked into the plot: unlike 3D, the 1D contact gas has NO
unitarity/resonance -- an impurity-sea dimer exists for any attraction and its
binding grows as -g^2/4. So E_p/E_F does not saturate at a universal value; it
keeps diving. The 3D unitary attractive-polaron value (-0.6 E_F) is drawn only
as an orientation landmark, not as something the 1D model should reproduce.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from fermiongas import TrapDVR, contact_eri, TwoComponentFCI

# --- system ---------------------------------------------------------------
N_SEA = 4
N_ORB = 10
G_GRID = np.linspace(-6.0, 4.0, 41)

trap = TrapDVR(n_orb=N_ORB)
eri = contact_eri(trap.orbitals, trap.dx)
solver = TwoComponentFCI(trap.energies, eri, N_SEA, 1)
e_ni = solver.h0_diag[solver.ref]
ref = solver.ref
H0 = np.diag(solver.h0_diag)

E_F = float(trap.energies[N_SEA - 1])   # Fermi energy of the sea
k_F = np.sqrt(2.0 * E_F)

eta, inv_kfa = [], []
ep_ef, z_res = [], []
for g in G_GRID:
    w, v = eigh(H0 + g * solver.V, subset_by_index=(0, 0))
    ep_ef.append((w[0] - e_ni) / E_F)
    z_res.append(float(v[ref, 0] ** 2))
    eta.append(g / E_F)
    inv_kfa.append(-g / (2.0 * k_F))    # 1/(k_F a_1D), positive on attractive side

eta = np.array(eta)
ep_ef = np.array(ep_ef)
z_res = np.array(z_res)

# --- figure ---------------------------------------------------------------
fig, (axE, axZ) = plt.subplots(1, 2, figsize=(11, 4.4))

axE.axhline(0, color="0.85", lw=0.8)
axE.axvline(0, color="0.85", lw=0.8)
axE.axhline(-0.6, color="#d62728", lw=1.0, ls=(0, (4, 3)))
axE.text(eta.min() * 0.98, -0.55, "3D unitary attractive polaron  $-0.6\\,E_F$",
         color="#d62728", fontsize=8, va="bottom")
axE.plot(eta, ep_ef, "k-", lw=2.4)
axE.set_xlabel(r"dimensionless coupling  $g/E_F$")
axE.set_ylabel(r"polaron energy  $E_p / E_F$")
axE.set_title(f"Polaron energy (1 impurity, {N_SEA}-fermion sea, $E_F={E_F:.1f}$)")

axZ.axvline(0, color="0.85", lw=0.8)
axZ.plot(eta, z_res, color="#2ca02c", lw=2.4)
axZ.fill_between(eta, 0, z_res, color="#2ca02c", alpha=0.12)
axZ.set_xlabel(r"dimensionless coupling  $g/E_F$")
axZ.set_ylabel(r"quasiparticle residue  $Z$")
axZ.set_title("Residue vs coupling")
axZ.set_ylim(0, 1.02)

fig.tight_layout()
out = "polaron_scaled.png"
fig.savefig(out, dpi=150)
print("wrote", out)

print(f"\nE_F = {E_F:.3f}   k_F = {k_F:.3f}")
print("   g     g/E_F   1/(kF a1D)   E_p/E_F     Z")
for k in range(0, len(G_GRID), 4):
    print(f" {G_GRID[k]:+.2f}  {eta[k]:+.3f}   {inv_kfa[k]:+.3f}     "
          f"{ep_ef[k]:+.4f}   {z_res[k]:.3f}")

"""Method comparison for the two-component contact gas, in the spirit of the
Grining et al. benchmarks: mean-field (HF), SAPT PT2, psi4numpy-style CCSD, and
exact FCI, all on the same Hamiltonian.

Interaction energy E_int = E - E_nonint is plotted for each method versus the
coupling g. CCSD -- which resums both particle-hole AND particle-particle
(ladder/pairing) diagrams -- should track FCI far better than fixed-order PT2,
especially on the attractive side where the pairing channel matters (the same
channel the particle-hole RPA in figure_rpa.py missed).

Also writes an FCIDUMP so the identical Hamiltonian can be cross-checked with
pyscf or any external CC code.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fermiongas import (
    TrapDVR, contact_eri, TwoComponentFCI,
    rhf_contact, ccsd, write_fcidump,
)

# --- system ---------------------------------------------------------------
N_PAIR = 2          # N_up = N_dn = 2
N_ORB = 8
G_GRID = np.linspace(-3.0, 3.0, 25)

trap = TrapDVR(n_orb=N_ORB)
eri = contact_eri(trap.orbitals, trap.dx)
solver = TwoComponentFCI(trap.energies, eri, N_PAIR, N_PAIR)
e_ni = solver.h0_diag[solver.ref]

hf, pt2, cc, fci = [], [], [], []
cc_conv = []
for g in G_GRID:
    ref = rhf_contact(trap.energies, trap.orbitals, trap.dx, N_PAIR, g)
    c = ccsd(ref)
    hf.append(ref.e_hf - e_ni)
    pt2.append(solver.pt2(g).e_int)
    cc.append(c.e_total - e_ni)
    fci.append(solver.fci(g) - e_ni)
    cc_conv.append(c.converged)

hf = np.array(hf); pt2 = np.array(pt2); cc = np.array(cc); fci = np.array(fci)

# --- figure ---------------------------------------------------------------
fig, (axE, axErr) = plt.subplots(1, 2, figsize=(11, 4.4))

axE.axhline(0, color="0.85", lw=0.8); axE.axvline(0, color="0.85", lw=0.8)
axE.plot(G_GRID, fci, "k-", lw=2.6, label="FCI (exact)")
axE.plot(G_GRID, cc, "o", color="#1f77b4", ms=4, label="CCSD (psi4numpy-style)")
axE.plot(G_GRID, pt2, "--", color="#d62728", lw=1.7, label="SAPT PT2")
axE.plot(G_GRID, hf, ":", color="#7f7f7f", lw=1.7, label="HF (mean field)")
axE.set_xlabel(r"contact coupling  $g$")
axE.set_ylabel(r"interaction energy  $E_{\mathrm{int}}$")
axE.set_title(f"Method comparison (N={N_PAIR}+{N_PAIR}, {N_ORB} orbitals)")
axE.legend(frameon=False, fontsize=9, loc="upper left")

axErr.axhline(0, color="0.85", lw=0.8); axErr.axvline(0, color="0.85", lw=0.8)
axErr.semilogy(G_GRID, np.abs(cc - fci) + 1e-16, "o-", color="#1f77b4", ms=3, label="|CCSD - FCI|")
axErr.semilogy(G_GRID, np.abs(pt2 - fci) + 1e-16, "s--", color="#d62728", ms=3, label="|PT2 - FCI|")
axErr.semilogy(G_GRID, np.abs(hf - fci) + 1e-16, "^:", color="#7f7f7f", ms=3, label="|HF - FCI|")
axErr.set_xlabel(r"contact coupling  $g$")
axErr.set_ylabel(r"error vs FCI")
axErr.set_title("Error from exact")
axErr.legend(frameon=False, fontsize=9, loc="lower left")

fig.tight_layout()
fig.savefig("ccsd_compare.png", dpi=150)
print("wrote ccsd_compare.png")

# --- FCIDUMP export (Hamiltonian at g = 1 as an example) ------------------
write_fcidump(trap.energies, contact_eri(trap.orbitals, trap.dx, 1.0),
              n_elec=2 * N_PAIR, filename="contact_gas.FCIDUMP")
print("wrote contact_gas.FCIDUMP (g=1)")

print("\n   g      HF        PT2       CCSD       FCI     |CCSD-FCI| |PT2-FCI|")
for k in range(0, len(G_GRID), 3):
    print(f" {G_GRID[k]:+.2f}  {hf[k]:+.4f}  {pt2[k]:+.4f}  {cc[k]:+.5f}  "
          f"{fci[k]:+.5f}  {abs(cc[k]-fci[k]):.2e}  {abs(pt2[k]-fci[k]):.2e}")
if not all(cc_conv):
    print("NOTE: CCSD did not converge at some g:",
          [f"{g:+.2f}" for g, ok in zip(G_GRID, cc_conv) if not ok])

# Source modules reference

The importable library lives in `src/fermiongas/` and has four modules. All
quantities use natural trap units: `hbar = 1`, particle mass is a free
parameter, energies in units of the trap quantum, lengths in oscillator units.
Every integral / operator is returned as a plain numpy array.

Import surface:

```python
from fermiongas import (
    TrapDVR, harmonic_oscillator,   # trap.py
    contact_eri, inter_species_eri, # contact.py
    TwoComponentFCI, PT2Result,     # twocomponent.py
    dispersion,                     # rpa.py
    rhf_contact, RHFResult,         # hartreefock.py
    ccsd, CCSDResult,               # ccsd.py
    write_fcidump,                  # fcidump.py
)
```

---

## `trap.py` — single-species trap orbitals

### `harmonic_oscillator(x) -> ndarray`
Default trap potential `V(x) = 1/2 x^2` (frequency `omega = 1`).

### `class TrapDVR`
One species of fermions in an arbitrary 1D trap, solved by **Colbert-Miller
sinc-DVR** on a uniform grid. The kinetic matrix is the exact infinite-grid
form

```
T_ij = 1/(2 m dx^2) * { pi^2/3               i == j
                      { 2 (-1)^(i-j)/(i-j)^2  i != j
```

and `H = T + diag(V(x))` is diagonalized; the lowest `n_orb` states are kept
(high-lying DVR states are grid artifacts).

Constructor:

```python
TrapDVR(potential=None,   # callable V(x); defaults to 1/2 x^2
        mass=1.0,         # particle mass (any fermion)
        x_min=-10.0, x_max=10.0, dx=0.1,   # uniform grid
        n_orb=20)         # number of lowest orbitals to keep
```

Attributes: `grid`, `dx`, `energies` (shape `(n_orb,)`), `orbitals`
(shape `(n_grid, n_orb)`, columns are `phi_i(x_grid)` normalized so the grid
integral of `phi_i^2` is ~1), `mass`, `n_orb`.

Method: `density(occ)` — `sum_i<occ |phi_i(x)|^2`, a filled sea of `occ`
spin-polarized fermions.

Note: this is a clean-room rewrite of the original PyBEST `ContactInteraction1D`
model Hamiltonian, dropping the PyBEST dependency **and** the parity screen that
silently zeroed integrals for *asymmetric* traps. Asymmetric potentials are now
handled correctly.

---

## `contact.py` — contact interaction integrals

A zero-range interaction `g * delta(x - x')` becomes the fully symmetric tensor
`(ij|kl) = g * integral phi_i phi_j phi_k phi_l dx`, evaluated by grid
quadrature.

### `contact_eri(orbitals, dx, g=1.0) -> ndarray`
Single-species tensor, shape `(n_orb,)*4`, symmetric in all four indices.

### `inter_species_eri(orb_a, orb_b, dx, g_ab=1.0) -> ndarray`
Inter-species tensor `(i_A j_A | k_B l_B)`, shape `(n_A, n_A, n_B, n_B)`,
symmetric under `i<->j` and `k<->l`. Both orbital sets must be sampled on the
**same grid** (essential when A and B have different masses). This is the
object that couples the two Fermi vacua.

Statistics note: a single species of *spin-polarized* fermions does not feel the
contact interaction (the antisymmetric spatial wavefunction vanishes at
coincidence). The intra-species tensor is physical only for a two-component
(two spin state) species, or for building FCI benchmarks.

---

## `twocomponent.py` — FCI and SAPT-style PT2

Two-component (spin up / spin down) fermions in a shared trap, contact
interaction between opposite spins only. **FCI and PT2 are built from one
interaction operator**, so the comparison is a clean statement about the
perturbation series (not two different codes).

### `class TwoComponentFCI(energies, eri, n_up, n_dn)`
Builds the full determinant basis (bitmask determinants per spin sector) and the
operators `H = H0 + g * V`, where `H0` is the diagonal non-interacting energy and
`V` the unit-strength contact interaction

```
H_int = sum_{pqrs} <phi_p phi_q phi_r phi_s>
                   a+_{p up} a+_{q dn} a_{r dn} a_{s up}
```

Methods:

- `fci(g) -> float` — exact ground-state energy at coupling `g` (lowest
  eigenvalue; uses scipy `eigh` subset when available).
- `pt2(g) -> PT2Result` — Rayleigh-Schrodinger through 2nd order about the
  non-interacting trap reference, with the 2nd-order term split by excitation
  character:
  - **single excitations -> induction**,
  - **up-down double excitations -> dispersion**.

Useful attributes: `dim`, `ref` (reference determinant index), `h0_diag`, `V`.

### `class PT2Result`
Fields `e0, e1, e_ind, e_disp`, with properties `e2` (= ind + disp), `total`
(= e0 + e1 + e2), and `e_int` (= e1 + e2, the interaction energy relative to the
non-interacting reference). Here `e1` is the SAPT electrostatics term.

**Quasiparticle residue.** The FCI eigenvector's weight on the reference
determinant, `Z = |<Phi_0|Psi>|^2`, is computed in `scripts/polaron.py` by
taking the lowest eigenpair and squaring the reference component.

---

## `rpa.py` — fixed-order and resummed dispersion

Dispersion is the leading term of an infinite series of inter-spin **ring**
diagrams. This module builds both, via the imaginary-frequency response
integral.

```
E_disp^(20) = -(1/2pi) int_0^inf dw  Tr[ F_up(w) B F_dn(w) B^T ]     (fixed order)
E_disp^RPA  =  (1/2pi) int_0^inf dw  Tr ln( 1 - M(w) ),  M = F_up B F_dn B^T
```

with particle-hole response `F_sigma(w)_{ia} = 2 dE_ia / (dE_ia^2 + w^2)` and
coupling matrix `B_{ia,jb} = g (ia|jb)`. The frequency integral uses a
Gauss-Legendre rule under `w = c tan(theta)`.

### `dispersion(energies, eri, n_up, n_dn, g, n_omega=48, c=None) -> dict`
Returns:

- `e20` — fixed-order dispersion (matches the explicit
  `-sum |B|^2 / (dE_up + dE_dn)` and the determinant-PT2 value),
- `rpa` — ring-resummed dispersion (reduces to `e20` as `g -> 0`), or `nan` if
  the RPA kernel has gone unstable,
- `lambda_max` — largest eigenvalue of the RPA kernel over the frequency grid;
  **`>= 1` signals the collective (density-channel) instability**, the edge of
  the resummation's validity,
- `unstable` — bool flag.

Private helpers (used by `scripts/validate.py`): `_ph_space`,
`_coupling_matrix`, `_quad_nodes`.

**Finding.** For this contact system the ph-RPA resummation *over-correlates*
and flags a spurious instability; the physical breakdown lives in the
particle-particle (pairing) channel, not the particle-hole (density) rings.
See the top-level README for the full discussion.

---

## `hartreefock.py` — closed-shell HF reference

### `rhf_contact(energies, orbitals, dx, n_pair, g, ...) -> RHFResult`
Self-consistent closed-shell Hartree-Fock for the **balanced** two-component
contact gas (`N_up = N_dn = n_pair`, equal masses). Because same-spin fermions
do not contact-interact and opposite-spin interaction has no exchange, the Fock
operator is simply

```
F = h + J[D],   J_pq = sum_rs (pq|rs) D_rs,   D = C[:, :n_pair] C[:, :n_pair]^T
```

`RHFResult` fields: `e_hf` (total HF energy, absolute), `eps` (canonical HF
orbital energies), `C` (HF coefficients in the DVR-orbital basis), `mo_orbitals`
(HF orbitals on the grid = `orbitals @ C`), `eri_mo` (`(pq|rs)` in the HF MO
basis), `n_pair`. This is the reference CCSD consumes.

Limitation: closed-shell / balanced only. Imbalanced (`N_up != N_dn`) would
need UHF/ROHF.

---

## `ccsd.py` — psi4numpy-style spin-orbital CCSD

### `ccsd(ref, max_iter=200, e_conv=1e-10, damp=0.0) -> CCSDResult`
The **Stanton-Gauss-Watts-Bartlett (1991) spin-orbital CCSD** equations exactly
as in the psi4numpy Coupled-Cluster tutorials, but fed a fermiongas
`RHFResult` instead of `psi4.core.MintsHelper`. Same-spin antisymmetrized
contact integrals cancel automatically, so only up-down terms enter. Spin-
orbitals are reordered by energy so occupied come first.

`CCSDResult` fields: `e_hf`, `e_corr`, `converged`, `n_iter`, plus property
`e_total = e_hf + e_corr`. `damp` in `[0,1)` linearly mixes old/new amplitudes,
which helps in the stiff strong-coupling regime.

Validation: `e_total` equals FCI to ~1e-8 for 1+1 (CCSD exact for two fermions);
tracks FCI to ~1e-4 for 2+2 at weak/moderate coupling. It **fails to converge**
at strong repulsion (fermionization), correctly signalling the single-reference
breakdown.

---

## `fcidump.py` — Hamiltonian export

### `write_fcidump(energies, eri, n_elec, filename, ms2=0, core_energy=0.0)`
Write a standard **FCIDUMP** (one-body diagonal `energies`, two-body chemist
`(ij|kl)` with 8-fold-unique storage, core energy). Lets external codes (pyscf,
psi4numpy CCSD, DMRG, ...) consume the identical contact-gas Hamiltonian for
cross-checks. The contact interaction is spin-independent, so the spatial-orbital
dump is written directly and a standard antisymmetric solver reproduces the
same-spin cancellation on its own.

---

## Extending toward psi4 / new physics

- Integrals are bare numpy arrays -> wrap into a psi4 `Matrix`/`Tensor` to reuse
  psi4's FCI/CC/SAPT solvers.
- `inter_species_eri` already supports **mass-imbalanced two-species** integrals
  on a shared grid (e.g. Li / Cr) — the basis for tuning the induction/dispersion
  ratio via the trap-frequency (mass) ratio.
- A **three-species** extension (e.g. Li-up / Li-down / Cr) would unlock the
  first genuinely non-additive term: three-body (Axilrod-Teller-Muto-like)
  dispersion, which two species cannot produce.

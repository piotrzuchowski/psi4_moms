# fermiongas

Trapped few-fermion mixtures treated with **SAPT-style perturbation theory**,
benchmarked against **exact diagonalization (FCI)**. Standalone (numpy/scipy);
every integral comes out as a plain numpy array, so the same tensors can later
be handed to psi4 or any FCI/CC/SAPT backend.

---

## The idea

Two distinguishable sets of trapped fermions (spin-up / spin-down, or two
species such as Li / Cr) have a zeroth-order state that is a **product of two
Slater determinants** — one per "Fermi vacuum". That is exactly the starting
point of Symmetry-Adapted Perturbation Theory (SAPT) for two molecules. Because
the two sets are **distinguishable, there is no inter-set exchange**, so SAPT
collapses to its polarization skeleton:

```
E_int  =  E1[electrostatics]  +  E2[induction + dispersion]  +  ...
```

with **no exchange terms**. This package builds the trap orbitals, the contact
interaction integrals, the exact FCI ground state, and the SAPT decomposition,
so one can ask quantitatively **where the perturbation series works and where it
breaks** — using FCI as ground truth.

Physical dictionary:

| SAPT term         | Cold-atom meaning                                             |
|-------------------|--------------------------------------------------------------|
| electrostatics    | mean-field (Hartree) energy between the two densities        |
| induction         | each cloud's density relaxing in the static mean field of the other (mean-field self-consistency) |
| dispersion        | correlated co-fluctuation of the two clouds (the *mediated / induced* interaction) |
| (exchange)        | **absent** — the two sets are distinguishable                |

In the **single-impurity** limit this is exactly the **Fermi polaron**: the
polaron energy splits into mean-field shift + static dressing (induction) +
dynamical dressing (dispersion), and the FCI eigenvector gives the
quasiparticle residue `Z`.

---

## Install

```bash
pip install -e .            # core (numpy, scipy)
pip install -e ".[figures]" # also matplotlib, for the scripts/
```

Then:

```python
from fermiongas import TrapDVR, contact_eri, TwoComponentFCI, dispersion
```

Run the checks:

```bash
python scripts/validate.py
```

---

## Layout

```
fermion_gas/
  src/fermiongas/       importable library (see docs/source_modules.md)
    trap.py             Colbert-Miller sinc-DVR for one species in a 1D trap
    contact.py          contact 4-index integrals (intra- and inter-species)
    twocomponent.py     two-component FCI + SAPT-style PT2 (ind/disp split)
    rpa.py              fixed-order and RPA-resummed dispersion + stability
    hartreefock.py      closed-shell HF reference for the contact gas
    ccsd.py             psi4numpy-style spin-orbital CCSD on the integrals
    fcidump.py          export the Hamiltonian as a standard FCIDUMP
  scripts/              drivers that produce the figures/results below
    validate.py         analytic + consistency checks
    make_figure.py      ED vs 1st/2nd-order PT (balanced gas)
    figure_rpa.py       fixed-order vs RPA dispersion + instability monitor
    polaron.py          polaron energy decomposition + residue Z
    polaron_scaled.py   polaron E_p/E_F and Z vs dimensionless coupling
    ccsd_compare.py     HF / PT2 / CCSD / FCI method comparison
  docs/
    source_modules.md   per-module reference for the library
  resources/            reference literature (PDFs, gitignored) + original integrals.py
```

`docs/source_modules.md` documents the four library modules in detail.

---

## What was studied, and what we found

### 1. Two-component gas: ED vs SAPT perturbation theory (`make_figure.py`)

Spin-up / spin-down fermions sharing a trap, contact interaction between
opposite spins. FCI vs 1st-order (electrostatics) vs 2nd-order (elst + ind +
disp), swept over coupling `g`.

- **1st order (mean field) is qualitatively wrong** on both wings.
- **2nd order tracks FCI** across a wide range; adding induction + dispersion
  recovers the curvature and the attractive/repulsive asymmetry (the latter
  coming entirely from the always-stabilizing `E2`).
- **Dispersion dominates induction** — the correlated term beats the
  mean-field-relaxation term, as in molecular van der Waals.

### 2. Resummed dispersion: RPA is the *wrong channel* (`figure_rpa.py`)

Compared fixed-order dispersion with a particle-hole ring (direct-RPA)
resummation, on the 3+3 gas.

- **Fixed-order SAPT dispersion is remarkably robust** — it tracks FCI from
  `g = -2.5` to `+5`.
- **Naive ph-RPA is worse, not better**: it over-correlates and develops a
  *spurious* density-channel instability (largest kernel eigenvalue -> 1).
- **Lesson:** the physical strong-coupling breakdown is in the
  **particle-particle (pairing/BCS) channel**, which density-ring RPA cannot
  see. The correct resummation for attractive contact fermions is the
  **ladder / T-matrix** channel. This confirms a general caveat: SAPT(DFT)-style
  density-response dispersion is not automatically right for gapless/contact
  fermions — the interaction channel matters.

### 3. Fermi-polaron limit (`polaron.py`, `polaron_scaled.py`)

One impurity in an N-fermion sea.

- Polaron energy splits cleanly into **mean-field shift + induction (static
  dressing) + dispersion (dynamical dressing)**, with dispersion ~2x induction.
- **PT2 begins to overbind at strong attraction** (`g < ~-3.5`) — the visible
  onset of the particle-particle (ladder) regime.
- **Residue `Z` stays > 0.8** across the studied window: the impurity remains a
  well-defined quasiparticle. The polaron->molecule crossover (`Z -> 0`) is
  **not reached** in a finite 1D trap / truncated basis — it needs stronger
  coupling and more orbitals.
- `polaron_scaled.py` reports `E_p/E_F` and `Z` versus a **dimensionless
  coupling**, directly comparable to measured Fermi-polaron data. It also makes
  the **1D-vs-3D difference visible**: unlike 3D, the 1D contact gas has **no
  unitarity/resonance**, so `E_p/E_F` does not saturate at the universal
  `-0.6 E_F` — it keeps diving as `-g^2/4` (an impurity-sea dimer exists for any
  attraction). The repulsive branch, by contrast, is quantitatively
  experiment-like.

### 4. Method ladder: HF / PT2 / CCSD / FCI (`ccsd_compare.py`)

The full quantum-chemistry ladder on one Hamiltonian, in the spirit of the
Grining et al. benchmarks. CCSD is the **psi4numpy spin-orbital algorithm** (the
Stanton equations from the psi4numpy Coupled-Cluster tutorials), fed *our*
contact integrals via a Hartree-Fock reference (`hartreefock.py`) instead of
`psi4.core.MintsHelper`.

- **Validation:** CCSD equals FCI to `~1e-8` for 1+1 (CCSD is exact for two
  fermions).
- **CCSD dominates on the attractive / weak-repulsive side** — 1-2 orders of
  magnitude closer to FCI than PT2. This is the payoff of the
  **particle-particle (pairing/ladder) diagrams** CCSD resums: it captures
  exactly the channel the particle-hole RPA in study 2 *missed*.
- **CCSD breaks down at strong repulsion** (`g > ~2.2`): it overshoots and then
  **fails to converge** — the fermionization/Tonks regime, where the
  single-determinant reference is qualitatively wrong (static correlation) and
  single-reference CCSD is known to diverge. A *new* breakdown boundary,
  distinct from where PT2 and RPA fail; there the humble PT2 is actually more
  stable than the diverging CCSD.
- `fcidump.py` exports the identical Hamiltonian as a standard **FCIDUMP**, so
  it can be cross-checked with pyscf or any external CC code.

**Connecting integrals to CCSD.** Native psi4's compiled CC modules read
DPD integrals from their own transform pipeline and are impractical to feed a
custom Hamiltonian; the psi4numpy route works because its amplitude equations
need only numpy tensors -- which is exactly what this package produces.

---

## Caveats (read before over-interpreting)

- **1D contact, not 3D.** No scattering resonance / unitarity in 1D; the
  attractive side differs *in kind* from 3D experiments. Bridge to experiment
  on the **repulsive branch** and via **dimensionless observables** (`E_p/E_F`,
  `Z`), not via the coupling constant.
- **Finite basis.** Tight-dimer (molecule) formation is capped by the number of
  orbitals, so `Z`-collapse is not reachable at the scales used here.
- **Contact-interaction renormalization.** With a bare delta interaction each
  order of PT carries a cutoff-dependent piece; the finite basis is the cutoff.
  Absolute numbers are basis-limited, but the **FCI-vs-PT comparison is
  basis-exact** (both use the same orbitals and operator).

---

## Context

The dispersion / induction terms are the cold-atom **mediated (induced)
interaction** — the same physics as the RKKY interaction and the Fermi-polaron
dressing cloud. The strong-coupling regime probed by experiments (e.g. the
Feshbach-tuned Fermi-polaron work of Zaccanti and collaborators, centered on
unitarity) is precisely the non-perturbative window where fixed-order SAPT
fails and the particle-particle ladder is required — consistent with what the
`figure_rpa.py` and `polaron.py` results show here.

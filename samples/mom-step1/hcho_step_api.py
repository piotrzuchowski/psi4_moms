"""Formaldehyde HOMO -> LUMO excited state via STEP (PsiAPI).

Runs an RHF ground state followed by a UHF STEP excited-state SCF
(State-Targeted Energy Projection, Carter-Fenk & Herbert,
JCTC 16, 5067 (2020), Eq. 6).  Prints:

  * Ground-state and excited-state total energies
  * STEP excitation energy in eV
  * Orbital-overlap diagnostic: confirms the RHF LUMO ended up in the
    STEP alpha OCCupied block and the RHF HOMO in the VIRtual block
    (implementation-correctness check; method-independent)
  * UHF <S^2> observed and the spin-contamination residual dS
"""

import numpy as np
import psi4


def spin_contamination(wfn, S_ao: np.ndarray) -> dict:
    """Return dict {exact, observed, dS} for a UHF wavefunction.

    Mirrors Psi4's internal HF::compute_spin_contamination formula:
      dN = ||C_a_occ^T S C_b_occ||_F^2,
      dS = min(na, nb) - dN,
      <S^2>_exact    = |na-nb|/2 * (|na-nb|/2 + 1),
      <S^2>_observed = <S^2>_exact + dS.
    """
    Ca = np.asarray(wfn.Ca())
    Cb = np.asarray(wfn.Cb())
    na = wfn.nalpha()
    nb = wfn.nbeta()
    M  = Ca[:, :na].T @ S_ao @ Cb[:, :nb]
    dN = float((M * M).sum())
    dS = min(na, nb) - dN
    nm = (na - nb) / 2.0
    s2_exact    = abs(nm) * (abs(nm) + 1.0)
    s2_observed = s2_exact + dS
    return {"exact": s2_exact, "observed": s2_observed, "dS": dS}


def main() -> None:
    psi4.set_output_file("hcho_step_api.out", append=False)
    psi4.set_memory("500 MB")

    hcho = psi4.geometry(
        """
        0 1
        C  0.000000  0.000000  0.000000
        O  0.000000  0.000000  1.208000
        H  0.000000  0.943102 -0.544500
        H  0.000000 -0.943102 -0.544500
        symmetry c1
        """
    )

    psi4.set_options(
        {
            "basis":         "6-31G",
            "scf_type":      "pk",
            "e_convergence": 1.0e-8,
            "d_convergence": 1.0e-8,
            "guess":         "sad",
            "reference":     "rhf",
        }
    )

    # --- 1. RHF ground state (reference orbitals for diagnostics) ---
    E_gs, wfn_rhf = psi4.energy("scf", return_wfn=True, molecule=hcho)
    print(f"\nRHF ground state        : {E_gs:20.12f} Eh")

    Ca_rhf   = np.asarray(wfn_rhf.Ca())
    S_ao     = np.asarray(wfn_rhf.S())
    nalpha   = wfn_rhf.nalpha()
    phi_HOMO = Ca_rhf[:, nalpha - 1]
    phi_LUMO = Ca_rhf[:, nalpha]

    # --- 2. UHF STEP excited state ---
    psi4.set_options(
        {
            "reference":         "uhf",
            "guess":             "read",
            "mom_start":         5,
            "mom_occ":           [8],
            "mom_vir":           [9],
            "mom_step":          True,
            "mom_step_epsilon":  0.1,
        }
    )
    E_step, wfn_step = psi4.energy("scf", return_wfn=True, molecule=hcho)

    HA2EV = 27.2113957
    print(f"UHF STEP n->pi*         : {E_step:20.12f} Eh")
    print(f"STEP excitation energy  : {(E_step - E_gs) * HA2EV:12.3f} eV")

    # --- 3. Orbital-overlap diagnostic ---
    Ca_step  = np.asarray(wfn_step.Ca())
    nalpha_s = wfn_step.nalpha()

    ovl_LUMO = Ca_step.T @ S_ao @ phi_LUMO
    best_occ_LUMO = float(np.max(np.abs(ovl_LUMO[:nalpha_s])))
    best_vir_LUMO = float(np.max(np.abs(ovl_LUMO[nalpha_s:])))

    ovl_HOMO = Ca_step.T @ S_ao @ phi_HOMO
    best_occ_HOMO = float(np.max(np.abs(ovl_HOMO[:nalpha_s])))
    best_vir_HOMO = float(np.max(np.abs(ovl_HOMO[nalpha_s:])))

    print("\nOrbital-overlap diagnostic (vs. RHF reference):")
    print(f"  RHF LUMO -> STEP alpha:  |occ|_max = {best_occ_LUMO:.4f}   "
          f"|vir|_max = {best_vir_LUMO:.4f}  (want occ >> vir)")
    print(f"  RHF HOMO -> STEP alpha:  |occ|_max = {best_occ_HOMO:.4f}   "
          f"|vir|_max = {best_vir_HOMO:.4f}  (want vir >> occ)")

    # --- 4. Spin contamination ---
    s2 = spin_contamination(wfn_step, S_ao)
    print(f"\n<S^2>:  exact    = {s2['exact']:.4f}")
    print(f"        observed = {s2['observed']:.4f}")
    print(f"        dS       = {s2['dS']:.4f}    "
          f"(broken-symmetry UHF singlet typically has <S^2> ~ 1)")

    # Ziegler spin-purified energy estimate (Eq. 8 in Carter-Fenk & Herbert):
    # E_sing ~ 2 E_mix - E_trip.  We don't compute E_trip here, but flag the
    # value of <S^2> so the user knows whether spin-purification is advisable.
    if s2["observed"] > 0.1:
        print("  Spin contamination is substantial; consider a spin-purified")
        print("  estimate E_sing ~ 2*E_mix - E_trip (Ziegler multiplet split).")


if __name__ == "__main__":
    main()

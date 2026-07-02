"""MOM-SAPT: SAPT0 interaction energy with an electronically excited monomer.

Formaldehyde (excited n->pi* via MOM) ... water (ground state).

The new SAPT option ``sapt_mom_monomer`` (NONE / A / B) routes ONE monomer's
SCF inside the SAPT0 driver through the MOM/IMOM/STEP excited-state machinery
(set up with the usual mom_start / mom_occ / mom_vir keywords), while the
partner monomer and the dimer SCF remain ground-state Aufbau.  The delta-HF
correction is skipped -- a ground-state dimer SCF is inconsistent with an
excited monomer -- so SAPT TOTAL ENERGY is the plain sum
Elst10,r + Exch10 + (Ind20,r + Exch-Ind20,r) + (Disp20 + Exch-Disp20).

Open-shell SAPT0 requires the dimer-centered basis (the default
sapt_basis='dimer'), so mom_occ/mom_vir indices refer to the ghost-augmented
monomer SCF; occupied indices match the isolated molecule, but check that
the target virtual is not reordered by the ghost functions.

Run:  python mom_sapt_api.py
"""

import psi4

HA2EV = psi4.constants.hartree2ev
HA2KCAL = psi4.constants.hartree2kcalmol


def main() -> None:
    psi4.set_output_file("mom_sapt_api.out", append=False)
    psi4.set_memory("500 MB")

    dimer = psi4.geometry(
        """
        0 1
        C  0.000000  0.000000  0.000000
        O  0.000000  0.000000  1.208000
        H  0.000000  0.943102 -0.544500
        H  0.000000 -0.943102 -0.544500
        --
        0 1
        O  0.000000  0.000000  3.400000
        H  0.000000  0.755000  3.980000
        H  0.000000 -0.755000  3.980000
        units angstrom
        symmetry c1
        """
    )

    psi4.set_options(
        {
            "basis":         "cc-pvdz",
            "scf_type":      "df",
            "reference":     "uhf",
            "e_convergence": 1.0e-8,
            "d_convergence": 1.0e-8,
            "guess":         "sad",
        }
    )

    def components():
        return {t: psi4.variable(f"SAPT {t} ENERGY")
                for t in ("ELST", "EXCH", "IND", "DISP", "TOTAL")}

    # --- 1. Ground-state SAPT0 (reference) ---
    psi4.energy("sapt0", molecule=dimer)
    gs = components()
    psi4.core.clean()

    # --- 2. MOM-SAPT: HCHO (monomer A) excited HOMO(8) -> LUMO(9) ---
    psi4.set_options(
        {
            "sapt_mom_monomer": "A",
            "mom_start":        5,
            "mom_occ":          [8],
            "mom_vir":          [9],
        }
    )
    psi4.energy("sapt0", molecule=dimer)
    ex = components()
    exc = psi4.variable("SAPT MOM EXCITATION ENERGY")

    # --- 3. Report ---
    print(f"\nMonomer A (HCHO) n->pi* excitation: {exc:.6f} Eh = {exc * HA2EV:.3f} eV\n")
    print(f"{'term':>6s} {'ground [mEh]':>14s} {'excited [mEh]':>14s} {'shift [mEh]':>13s}")
    for t in ("ELST", "EXCH", "IND", "DISP", "TOTAL"):
        g, e = gs[t] * 1000.0, ex[t] * 1000.0
        print(f"{t:>6s} {g:14.4f} {e:14.4f} {e - g:13.4f}")
    print(f"\nGround-state E_int : {gs['TOTAL'] * HA2KCAL:10.4f} kcal/mol (with dHF)")
    print(f"Excited-state E_int: {ex['TOTAL'] * HA2KCAL:10.4f} kcal/mol (no dHF)")
    print("\nNote: the ground-state SAPT0 total includes delta-HF, the MOM-SAPT")
    print("total does not; compare component-by-component for a clean picture.")


if __name__ == "__main__":
    main()

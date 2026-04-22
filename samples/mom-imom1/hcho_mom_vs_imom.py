"""Formaldehyde HOMO -> LUMO excited-state UHF at 6-31G, MOM vs IMOM.

Runs the same excited-state determinant twice -- once with classical
sliding-reference MOM, once with IMOM (initial-MOM) -- parses the SCF
iterations from each run's output file, and produces a comparison plot.

Artifacts produced in the current working directory:
    rhf_ground.out    -- ground-state RHF of neutral HCHO
    uhf_mom.out       -- excited-state UHF with classical MOM
    rhf_guess.out     -- second RHF pass (to re-seed guess for IMOM)
    uhf_imom.out      -- excited-state UHF with IMOM
    mom_vs_imom.png   -- the plot
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import psi4


ITER_LINE = re.compile(
    r"@\S+\s+iter\s+(\S+):\s+"            # tag & iter label (digit or SAD/MOM)
    r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"   # total energy
    r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s+"   # Delta E
    r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)"      # density RMS
)


def parse_scf_iterations(path: str):
    """Return list of dicts {iter, energy, dE, rms} from a Psi4 output file."""
    rows = []
    with open(path) as fh:
        for line in fh:
            m = ITER_LINE.search(line)
            if not m:
                continue
            label, energy, de, rms = m.groups()
            # Map guess labels (SAD, MOM) to a best-effort numeric index
            try:
                it = int(label)
            except ValueError:
                # Guess iterations and MOM-start markers: use -1 so they sort
                # before iteration 1.  We'll drop them from the plot.
                it = -1
            rows.append({
                "iter":   it,
                "label":  label,
                "energy": float(energy),
                "dE":     float(de),
                "rms":    float(rms),
            })
    return rows


def run_hcho_mom_imom():
    """Drive the three SCFs, splitting output across separate files."""
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

    common = {
        "basis":         "jun-cc-pVDZ",
        "scf_type":      "pk",
        "e_convergence": 1.0e-8,
        "d_convergence": 1.0e-8,
    }

    # 1. RHF ground state
    psi4.set_output_file("rhf_ground.out", append=False)
    psi4.set_options({**common, "reference": "rhf", "guess": "sad"})
    e_rhf, wfn_rhf = psi4.energy("scf", molecule=hcho, return_wfn=True)

    # 2. UHF HOMO -> LUMO excited, classical MOM
    psi4.set_output_file("uhf_mom.out", append=False)
    psi4.set_options({
        **common,
        "reference":   "uhf",
        "guess":       "read",
        "mom_start":   5,
        "mom_occ":     [8],
        "mom_vir":     [9],
        "mom_initial": False,
    })
    e_mom, wfn_mom = psi4.energy("scf", molecule=hcho, return_wfn=True)

    # 3. Re-seed guess (second RHF pass) so IMOM reads the same orbitals
    psi4.set_output_file("rhf_guess.out", append=False)
    psi4.core.clean()
    psi4.set_options({**common, "reference": "rhf", "guess": "sad"})
    _ = psi4.energy("scf", molecule=hcho)

    # 4. UHF HOMO -> LUMO excited, IMOM
    psi4.set_output_file("uhf_imom.out", append=False)
    psi4.set_options({
        **common,
        "reference":   "uhf",
        "guess":       "read",
        "mom_start":   5,
        "mom_occ":     [8],
        "mom_vir":     [9],
        "mom_initial": True,
    })
    e_imom, wfn_imom = psi4.energy("scf", molecule=hcho, return_wfn=True)

    return (e_rhf, wfn_rhf), (e_mom, wfn_mom), (e_imom, wfn_imom)


def orbital_overlap_diagnostic(wfn_ref, wfn_final, ref_orb_idx: int,
                               label: str) -> dict:
    """Compute overlaps between a reference orbital and the final alpha
    occupied set.

    Parameters
    ----------
    wfn_ref : psi4.core.Wavefunction
        Wavefunction whose Ca column `ref_orb_idx` defines the reference MO
        (0-indexed; for HOMO use nalpha-1, for LUMO use nalpha).
    wfn_final : psi4.core.Wavefunction
        Wavefunction after the MOM/IMOM run.  Must be C1 symmetry.
    ref_orb_idx : int
        0-indexed column in `wfn_ref.Ca()` to use as reference orbital.
    label : str
        Name to print (e.g. "MOM", "IMOM").

    Returns
    -------
    dict with keys {"overlaps", "best_occ", "best_occ_overlap", "best_vir",
                    "best_vir_overlap"}.
    """
    Ca_ref   = wfn_ref.Ca().to_array()     # (nbf, nmo)
    Ca_final = wfn_final.Ca().to_array()
    S        = wfn_ref.S().to_array()      # AO overlap, basis-invariant

    phi_ref = Ca_ref[:, ref_orb_idx]        # (nbf,)
    # Overlap vector: S-inner-product of phi_ref with every final alpha MO
    overlaps = Ca_final.T @ S @ phi_ref     # (nmo,)

    nocc = wfn_final.nalpha()
    occ_abs = np.abs(overlaps[:nocc])
    vir_abs = np.abs(overlaps[nocc:])
    best_occ = int(np.argmax(occ_abs))
    best_vir = int(np.argmax(vir_abs)) + nocc

    info = {
        "overlaps":         overlaps,
        "best_occ":         best_occ + 1,           # 1-indexed for display
        "best_occ_overlap": float(occ_abs[best_occ]),
        "best_vir":         best_vir + 1,
        "best_vir_overlap": float(np.abs(overlaps[best_vir])),
    }

    print(f"\n  [{label}] Reference orbital: RHF MO #{ref_orb_idx + 1}")
    print(f"    Best match in final alpha OCCupied set:  "
          f"MO #{info['best_occ']}  |<ref|S|phi>| = {info['best_occ_overlap']:.6f}")
    print(f"    Best match in final alpha VIRtual  set:  "
          f"MO #{info['best_vir']}  |<ref|S|phi>| = {info['best_vir_overlap']:.6f}")
    return info


def plot(rows_mom, rows_imom, out_path: str, mom_start: int = 5):
    """Plot energy and |dE| per iteration for MOM and IMOM on one figure."""

    # Drop the guess (iter < 1) rows for plotting
    mom  = [r for r in rows_mom  if r["iter"] >= 1]
    imom = [r for r in rows_imom if r["iter"] >= 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # --- Top: total energy ---
    ax1.plot([r["iter"] for r in mom],
             [r["energy"] for r in mom],
             "o-", label="MOM (sliding ref.)", color="tab:blue")
    ax1.plot([r["iter"] for r in imom],
             [r["energy"] for r in imom],
             "s--", label="IMOM (frozen ref.)", color="tab:red")
    ax1.axvline(mom_start, color="gray", linestyle=":", alpha=0.7,
                label=f"MOM_START = {mom_start}")
    ax1.set_ylabel("UHF energy / Eh")
    ax1.set_title("HCHO / 6-31G  HOMO→LUMO excited state — SCF convergence")
    ax1.legend(loc="best", frameon=False)
    ax1.grid(True, alpha=0.3)

    # --- Bottom: |dE| on log scale ---
    ax2.semilogy([r["iter"] for r in mom],
                 [max(abs(r["dE"]), 1.0e-14) for r in mom],
                 "o-", label="MOM", color="tab:blue")
    ax2.semilogy([r["iter"] for r in imom],
                 [max(abs(r["dE"]), 1.0e-14) for r in imom],
                 "s--", label="IMOM", color="tab:red")
    ax2.axvline(mom_start, color="gray", linestyle=":", alpha=0.7)
    ax2.set_xlabel("SCF iteration")
    ax2.set_ylabel("|ΔE| / Eh  (log)")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def main() -> None:
    psi4.set_memory("500 MB")

    (e_rhf, wfn_rhf), (e_mom, wfn_mom), (e_imom, wfn_imom) = run_hcho_mom_imom()
    HA2EV = 27.2113957

    print("\n" + "=" * 60)
    print(f"{'Method':<32s}{'E / Eh':>20s}")
    print("-" * 60)
    print(f"{'RHF ground state':<32s}{e_rhf:>20.12f}")
    print(f"{'UHF n->pi* (classical MOM)':<32s}{e_mom:>20.12f}")
    print(f"{'UHF n->pi* (IMOM)':<32s}{e_imom:>20.12f}")
    print("-" * 60)
    print(f"{'dE (MOM)':<32s}{(e_mom - e_rhf) * HA2EV:>17.3f} eV")
    print(f"{'dE (IMOM)':<32s}{(e_imom - e_rhf) * HA2EV:>17.3f} eV")
    print(f"{'|E(MOM) - E(IMOM)|':<32s}"
          f"{abs(e_mom - e_imom) * 1000.0:>17.3f} mEh")
    print("=" * 60)

    # --- Overlap diagnostic ------------------------------------------
    # The essential test: did the initially-promoted LUMO (RHF MO #9,
    # 0-indexed 8) survive into the final UHF alpha *occupied* set?
    # For a successful MOM/IMOM run, some final occupied alpha MO should
    # have near-unity overlap with the initial LUMO.
    # If the overlap lives in the VIRTUAL set instead, the run collapsed
    # back to the ground state.
    nalpha = wfn_rhf.nalpha()          # 8 for HCHO
    lumo_idx = nalpha                  # 0-indexed: column #8 = MO #9 (LUMO)
    homo_idx = nalpha - 1              # 0-indexed: column #7 = MO #8 (HOMO)

    print("\n" + "=" * 60)
    print("  Orbital-overlap diagnostic")
    print("  Reference orbitals taken from the initial RHF ground state.")
    print("=" * 60)
    info_mom_lumo  = orbital_overlap_diagnostic(
        wfn_rhf, wfn_mom,  lumo_idx, "MOM  / init LUMO (MO #9)")
    info_imom_lumo = orbital_overlap_diagnostic(
        wfn_rhf, wfn_imom, lumo_idx, "IMOM / init LUMO (MO #9)")
    info_mom_homo  = orbital_overlap_diagnostic(
        wfn_rhf, wfn_mom,  homo_idx, "MOM  / init HOMO (MO #8)")
    info_imom_homo = orbital_overlap_diagnostic(
        wfn_rhf, wfn_imom, homo_idx, "IMOM / init HOMO (MO #8)")

    print("\n  Interpretation:")
    print("    - If init LUMO maps to an OCCupied MO with |overlap| ~ 1,")
    print("      the excitation survived and MOM/IMOM worked.")
    print("    - If init LUMO maps to a VIRtual MO with |overlap| ~ 1,")
    print("      the run collapsed back to the ground state.")
    print("    - Differences between MOM and IMOM overlaps (non-trivial")
    print("      decimal places) demonstrate the two code paths are")
    print("      distinct even when their final energies agree.")

    # --- Plot SCF iterations ----------------------------------------
    rows_mom  = parse_scf_iterations("uhf_mom.out")
    rows_imom = parse_scf_iterations("uhf_imom.out")

    if not rows_mom or not rows_imom:
        print("\nWARNING: no iteration data parsed.  Check the .out files.")
        return

    print(f"\nParsed {len(rows_mom)} MOM iterations, "
          f"{len(rows_imom)} IMOM iterations.")
    plot(rows_mom, rows_imom, "mom_vs_imom.png", mom_start=5)


if __name__ == "__main__":
    main()

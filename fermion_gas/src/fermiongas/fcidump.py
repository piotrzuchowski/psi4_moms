"""Export the fermiongas Hamiltonian as a standard FCIDUMP file.

FCIDUMP is the common interchange format for a second-quantized Hamiltonian
(one-body h_ij, two-body (ij|kl) in chemist notation, core energy). Writing it
lets any external code -- pyscf, a psi4numpy CCSD run, DMRG codes, etc. --
consume exactly this contact-gas Hamiltonian, e.g. for cross-checking the
in-package CCSD/FCI.

The contact interaction is spin-independent and local, so the spatial-orbital
FCIDUMP is written directly: a standard antisymmetric treatment then makes the
same-spin contribution cancel on its own (same-spin contact vanishes).
"""

from __future__ import annotations

import numpy as np


def write_fcidump(
    energies: np.ndarray,
    eri: np.ndarray,
    n_elec: int,
    filename: str,
    ms2: int = 0,
    core_energy: float = 0.0,
) -> None:
    """Write an FCIDUMP for one-body diag ``energies`` and two-body ``eri``.

    Parameters
    ----------
    energies : (n_orb,) non-interacting orbital energies (diagonal h).
    eri      : (n_orb,)*4 contact tensor (ij|kl) = int phi_i phi_j phi_k phi_l,
               fully symmetric, coupling g already folded in.
    n_elec   : total number of electrons (e.g. n_up + n_dn).
    filename : output path.
    ms2      : 2 * S_z (0 for a balanced singlet reference).
    """
    energies = np.asarray(energies, float)
    n = energies.size

    with open(filename, "w") as fh:
        fh.write(f" &FCI NORB={n},NELEC={n_elec},MS2={ms2},\n")
        fh.write("  ORBSYM=" + "1," * n + "\n")
        fh.write("  ISYM=1,\n &END\n")

        # two-electron integrals (ij|kl), 8-fold unique: i>=j, k>=l, (ij)>=(kl)
        for i in range(n):
            for j in range(i + 1):
                ij = i * (i + 1) // 2 + j
                for k in range(n):
                    for l in range(k + 1):
                        kl = k * (k + 1) // 2 + l
                        if kl > ij:
                            continue
                        v = eri[i, j, k, l]
                        if abs(v) > 1e-14:
                            fh.write(f"{v:23.16e}{i+1:4d}{j+1:4d}{k+1:4d}{l+1:4d}\n")

        # one-body (diagonal): h_ii  ->  value i i 0 0
        for i in range(n):
            fh.write(f"{energies[i]:23.16e}{i+1:4d}{i+1:4d}{0:4d}{0:4d}\n")

        # core energy
        fh.write(f"{core_energy:23.16e}{0:4d}{0:4d}{0:4d}{0:4d}\n")

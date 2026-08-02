"""One fermionic species in an arbitrary 1D trap, solved by Colbert-Miller
sinc-DVR (discrete variable representation).

Standalone: depends only on numpy. Atomic-style units with hbar = 1; the mass
is a free parameter so any fermion (Li, Er, ...) can be modeled. Orbitals and
integrals come out as plain numpy arrays so the same tensors can later be
handed to psi4 (or any FCI/CC/SAPT backend).

This is a clean-room rewrite of the original PyBEST ``ContactInteraction1D``
model Hamiltonian, with two deliberate changes:
  * no PyBEST dependency (plain numpy);
  * no parity screening of the grid, so *asymmetric* potentials are handled
    correctly (the original ``(i+j+k+l) % 2`` screen silently zeroed valid
    integrals whenever the trap was not symmetric).

Reference for the kinetic-energy matrix:
    D. T. Colbert and W. H. Miller, J. Chem. Phys. 96, 1982 (1992).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def harmonic_oscillator(x: np.ndarray) -> np.ndarray:
    """Default trap: V(x) = 1/2 x^2  (omega = 1 in the chosen units)."""
    return 0.5 * x**2


class TrapDVR:
    """Single-species 1D trap solved on a uniform Colbert-Miller sinc-DVR grid.

    Parameters
    ----------
    potential:
        Callable V(x) acting elementwise on the grid. Defaults to 1/2 x^2.
    mass:
        Particle mass in the chosen units (hbar = 1, electron mass = 1 by
        convention, but any value is fine -- it just rescales the trap).
    x_min, x_max, dx:
        Uniform grid definition. The default [-10, 10] with dx = 0.1 resolves
        the low-lying harmonic-oscillator states to high accuracy.
    n_orb:
        Number of lowest trap orbitals to retain. The high-lying DVR states are
        grid artifacts, so we keep only the converged bottom of the spectrum.
    """

    def __init__(
        self,
        potential: Callable[[np.ndarray], np.ndarray] | None = None,
        mass: float = 1.0,
        x_min: float = -10.0,
        x_max: float = 10.0,
        dx: float = 0.1,
        n_orb: int = 20,
    ) -> None:
        self.potential = potential if potential is not None else harmonic_oscillator
        self.mass = float(mass)
        # Build the grid so the spacing is exactly dx and the endpoints are hit.
        n_pts = int(round((x_max - x_min) / dx)) + 1
        self.grid = x_min + dx * np.arange(n_pts)
        self.dx = float(dx)
        self.n_orb = int(n_orb)
        self._solve()

    def _kinetic(self) -> np.ndarray:
        """Colbert-Miller kinetic-energy matrix on the uniform grid.

        T_ij = 1/(2 m dx^2) * { pi^2/3            if i == j
                              { 2 (-1)^(i-j)/(i-j)^2 if i != j
        """
        n = self.grid.size
        idx = np.arange(n)
        diff = idx[:, None] - idx[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mat = 2.0 * ((-1.0) ** diff) / (diff**2)
        np.fill_diagonal(mat, np.pi**2 / 3.0)
        mat *= 1.0 / (2.0 * self.mass * self.dx**2)
        return mat

    def _solve(self) -> None:
        """Diagonalize H = T + V(x) on the grid; keep the lowest n_orb states."""
        hamiltonian = self._kinetic() + np.diag(self.potential(self.grid))
        energies, vectors = np.linalg.eigh(hamiltonian)  # ascending
        # DVR eigenvectors are grid amplitudes normalized to sum(v^2) = 1.
        # The wavefunction sampled on the grid is phi(x_a) = v_a / sqrt(dx),
        # so that the trapezoidal/rectangle integral of phi^2 is ~ 1.
        self.energies = energies[: self.n_orb].copy()
        self.orbitals = (vectors[:, : self.n_orb] / np.sqrt(self.dx)).copy()

    # -- convenience --------------------------------------------------------

    def density(self, occ: int) -> np.ndarray:
        """Sum |phi_i(x)|^2 over the lowest ``occ`` orbitals (a filled Fermi
        sea of ``occ`` spin-polarized fermions)."""
        return np.einsum("gi,gi->g", self.orbitals[:, :occ], self.orbitals[:, :occ])

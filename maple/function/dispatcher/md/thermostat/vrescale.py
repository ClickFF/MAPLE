"""
V-rescale (stochastic velocity rescaling) thermostat for NVT molecular dynamics.

V-rescale corrects the Berendsen thermostat by adding a stochastic term to the
kinetic energy update, ensuring the canonical (NVT) distribution is sampled
exactly. The kinetic energy after rescaling follows a chi-squared distribution
with N_dof degrees of freedom.

Algorithm (Bussi et al., 2007):
    At each step, the kinetic energy KE is rescaled to a new value KE_new drawn
    from the conditional distribution:

        KE_new = KE_ref + (KE - KE_ref) * exp(-dt/τ_T)
               + sqrt(KE_ref * KE / N_dof * (1 - exp(-dt/τ_T)))
                 * (W1² + W2 + ... + W_{N_dof-1}² - 1)   [noise term]

    In practice the noise is computed via:
        dKE = KE_ref * [(1-f)*χ²_{N_dof} / N_dof + f - 1]   (Eq. 14 of the paper)
    where f = exp(-dt/τ_T), and χ²_{N_dof} is sampled using the sum-of-squares
    of N_dof standard normals.

    Velocities are uniformly scaled by α = sqrt(KE_new / KE).

Notes:
    - Produces the correct canonical ensemble unlike plain Berendsen rescaling.
    - No per-atom friction; global kinetic energy is rescaled uniformly.
    - τ_T → 0 reduces to instantaneous rescaling (isokinetic, incorrect ensemble).
    - τ_T → ∞ reduces to NVE (no coupling).

Reference:
    Bussi, Donadio & Parrinello, J. Chem. Phys. 126, 014101 (2007).
"""

import numpy as np
from ase import Atoms
from typing import Optional

from ..utils import AMU_TO_AU, FS_TO_AU, KELVIN_TO_HARTREE


class VRescaleThermostat:
    """
    Stochastic velocity rescaling thermostat (V-rescale).

    Rescales all velocities by a global factor α each step, where the new
    kinetic energy is drawn from the correct canonical distribution.
    """

    def __init__(
        self,
        atoms: Atoms,
        temperature: float,
        tau_t: float,
        timestep: float,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Parameters
        ----------
        atoms : ase.Atoms
            Molecular system
        temperature : float
            Target temperature in Kelvin
        tau_t : float
            Temperature coupling time constant in fs.
            Larger τ_T = weaker coupling.  Recommended: 100–500 fs for MLP.
        timestep : float
            MD timestep in fs
        rng : np.random.Generator, optional
            Random number generator for reproducibility
        """
        self.atoms = atoms
        self.temperature = temperature
        self.tau_t = tau_t * FS_TO_AU       # fs → a.u.
        self.timestep = timestep * FS_TO_AU # fs → a.u.
        self.masses = atoms.get_masses() * AMU_TO_AU
        self.rng = rng if rng is not None else np.random.default_rng()

        n_atoms = len(atoms)
        # Periodic systems have no overall translation; isolated molecules lose 3 COM DOF.
        self._n_dof = 3 * n_atoms if any(atoms.pbc) else 3 * n_atoms - 3
        self._kT_target = temperature * KELVIN_TO_HARTREE
        self._ke_target = 0.5 * self._n_dof * self._kT_target

        # exp(-dt/τ_T) precomputed
        self._decay = np.exp(-self.timestep / self.tau_t)

        # COM treatment: V-rescale applies a global uniform scaling factor α to all
        # velocities.  Unlike Langevin (which adds per-atom random impulses), uniform
        # scaling does not change the COM velocity direction, only its magnitude by
        # the same factor as all other velocities.  The N_dof accounting above already
        # excludes the 3 COM translational modes for isolated systems (3N-3 vs 3N),
        # so no explicit COM removal is needed here.
        # Ref: Bussi et al. (2007) J. Chem. Phys. 126, 014101 — Eq. 14 derivation
        # assumes the velocity distribution is sampled in the correct subspace.
        self._is_periodic = any(atoms.pbc)

    def _sample_chi2(self, n: int) -> float:
        """
        Sample from χ²(n) distribution as sum of n squared normals.

        For large n uses the normal approximation χ²(n) ≈ N(n, 2n).
        """
        if n <= 0:
            return 0.0
        if n == 1:
            w = self.rng.standard_normal()
            return w * w
        # Use sum of squares directly (exact, vectorised)
        w = self.rng.standard_normal(n)
        return float(np.dot(w, w))

    def apply(self, velocities: np.ndarray) -> np.ndarray:
        """
        Apply one V-rescale step: globally rescale velocities.

        Draws a new kinetic energy from the canonical distribution:

            KE_new = KE_ref + f*(KE - KE_ref)
                   + sqrt(KE_ref * KE * (1-f²) / N_dof) * W₁
                   + KE_ref * (1-f²) / (2*N_dof) * (χ²(N_dof-1) - (N_dof-1))

        where f = exp(-dt/τ_T).  Both stochastic terms have zero mean,
        so there is no systematic drift.

        Parameters
        ----------
        velocities : np.ndarray
            Current velocities in atomic units, shape (N_atoms, 3)

        Returns
        -------
        np.ndarray
            Rescaled velocities in atomic units
        """
        ke = 0.5 * np.sum(self.masses[:, np.newaxis] * velocities ** 2)

        if ke < 1e-30:
            return velocities

        f = self._decay
        ke_ref = self._ke_target
        n_dof = self._n_dof
        one_minus_f2 = 1.0 - f * f

        # Linear stochastic term: E = 0, Var = ke_ref * ke * (1-f²) / n_dof
        w1 = self.rng.standard_normal()
        cross = np.sqrt(ke_ref * ke * one_minus_f2 / n_dof) * w1

        # Quadratic stochastic term: chi2(n_dof-1) centered at (n_dof-1)
        chi2 = self._sample_chi2(n_dof - 1)
        quad = ke_ref * one_minus_f2 / (2.0 * n_dof) * (chi2 - (n_dof - 1))

        ke_new = ke_ref + f * (ke - ke_ref) + cross + quad
        ke_new = max(ke_new, 0.0)

        alpha = np.sqrt(ke_new / ke)
        return alpha * velocities

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from ase import Atoms

# --- physical constants ---
ANG2BOHR = 1.8897259886  # 1 Å = 1.8897 bohr
EH2EV = 27.211386245988  # 1 Hartree = 27.211 eV

def load_gbsa_params(solvent="water"):
    """
    Read solvent parameters from ./data/{solvent}.dat.
    Format:
      Line 1: 8 constants (eps, molarMass, refracIndex, gamma, beta, born_scale, born_offset, reserved)
      then '# array1' and '# array2' sections with comma-separated floats.
    """
    path = Path(__file__).parent / "data" / f"{solvent}.dat"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    header = [float(x) for x in lines[0].split(",")]
    eps, _, _, gamma, beta, born_scale, born_offset, _ = header

    allnums = []
    for l in lines[1:]:
        allnums.extend([float(x) for x in l.split(",") if x])

    half = len(allnums) // 2
    array1 = allnums[:half]   # born_shift-like
    array2 = allnums[half:]   # vdw_ref-like

    return dict(
        eps=eps,
        gamma=gamma,
        beta=beta,
        born_scale=born_scale,
        born_offset=born_offset,
        array1=array1,
        array2=array2,
    )

class GBSA(nn.Module):
    """
    xTB-style GBSA.
    Input coords in Å, output energy in Hartree, forces in Hartree/Å.
    """

    def __init__(self, solvent="water", device="cpu"):
        super().__init__()
        params = load_gbsa_params(solvent)
        self.device = torch.device(device)

        # global constants (assumed already in Eh / Bohr units like xTB)
        self.eps = float(params["eps"])
        self.gamma = float(params["gamma"])  # Eh/Bohr^2
        self.beta = float(params["beta"])    # Eh
        self.born_scale = float(params["born_scale"])
        self.born_offset = float(params["born_offset"])
        # GB dielectric factor: (1 - 1/eps) > 0
        self.keps = float(1.0 - 1.0 / self.eps)

        # element-specific parameters (Bohr)
        self.register_buffer("born_shift",
                             torch.tensor(params["array1"], dtype=torch.float32, device=self.device))
        self.register_buffer("vdw_ref",
                             torch.tensor(params["array2"], dtype=torch.float32, device=self.device))

        # SASA settings (xTB uses probe ≈ 1.4 Å, smoothing window w = 0.3 Å)
        probe_rad_A = 1.4
        w_A = 0.3
        self.register_buffer("probe_bohr", torch.tensor(probe_rad_A * ANG2BOHR, dtype=torch.float32, device=self.device))
        self.register_buffer("tau_bohr",   torch.tensor(w_A * ANG2BOHR,       dtype=torch.float32, device=self.device))

    # ---- radii ----
    def compute_born_radius(self, atom_index: torch.Tensor) -> torch.Tensor:
        """Born radii in Bohr (xTB-param style)."""
        vdwr = self.vdw_ref[atom_index]     # Bohr
        shift = self.born_shift[atom_index] # Bohr
        R = self.born_scale * (vdwr + shift) + self.born_offset
        return torch.clamp(R, min=0.5)      # Bohr

    def compute_sasa_area(self, coords_B: torch.Tensor, vdw_B: torch.Tensor) -> torch.Tensor:
        """
        Soft-visibility SASA approximation in Bohr^2.
        coords_B: (N,3) Bohr
        vdw_B:    (N,)  Bohr  (here we use VDW + probe)
        A_i ≈ 4π R_i^2 * Π_j σ((r_ij - (R_i+R_j))/τ), with σ logistic.
        Diagonal excluded to avoid self-burying.
        """
        N = coords_B.size(0)
        # pairwise distances in Bohr
        Rij = torch.cdist(coords_B, coords_B, p=2)  # (N,N)
        Ri = vdw_B.view(-1, 1)
        Rj = vdw_B.view(1, -1)
        # smooth visibility logits
        tau = self.tau_bohr
        logits = (Rij - (Ri + Rj)) / tau
        # exclude self-terms: set s_ii = 1 -> log s_ii = 0
        log_s = F.logsigmoid(logits)
        log_s.fill_diagonal_(0.0)
        # visibility product by log-sum-exp trick
        log_S = torch.sum(log_s, dim=1)     # (N,)
        S = torch.exp(log_S)                # (N,)
        # area
        A = 4.0 * torch.pi * (vdw_B ** 2) * S  # Bohr^2
        return A

    # ---- API ----
    def get_energy(self, atoms: Atoms):
        """
        Returns: energy (Eh), coords_A (Å, requires_grad=True)
        """
        # charges
        q_np = atoms.atomic_charges
        if q_np is None or np.allclose(q_np, 0):
            raise ValueError("Atoms object must have nonzero partial charges.")
        q = torch.as_tensor(q_np, dtype=torch.float32, device=self.device)  # e

        # coordinates: Å as leaf; convert to Bohr for physics
        coords_A = torch.as_tensor(atoms.get_positions(),
                                dtype=torch.float32,
                                device=self.device)  # Å
        coords_A.requires_grad_(True)
        coords_B = coords_A * ANG2BOHR                                        # Bohr

        # atom indices
        Z = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        atom_index = Z - 1

        # radii
        Ri = self.compute_born_radius(atom_index)                  # Bohr
        # GB Still f_ij
        Rij = torch.cdist(coords_B, coords_B, p=2) + 1e-9          # Bohr
        RiRj = Ri.view(-1, 1) * Ri.view(1, -1)                     # Bohr^2
        f_ij = torch.sqrt(Rij**2 + RiRj * torch.exp(-Rij**2 / (4.0 * RiRj) + 1e-12))  # Bohr

        # electrostatic GB energy, includes i=j terms naturally
        q_i = q.view(-1, 1)
        q_j = q.view(1, -1)
        E_polar = -0.5 * self.keps * torch.sum((q_i * q_j) / f_ij)  # Eh

        # SASA area with soft burial (vdw + probe)
        vdwsa = self.vdw_ref[atom_index] + self.probe_bohr          # Bohr
        A_i = self.compute_sasa_area(coords_B, vdwsa)               # Bohr^2
        E_nonpolar = self.gamma * torch.sum(A_i) + self.beta        # Eh

        E_total = E_polar #+ E_nonpolar                              # Eh
        return E_total, coords_A

    def get_energy_and_force(self, atoms: Atoms):
        """Return energy (Hartree) and forces (Hartree/Å)."""
        energy, coords_A = self.get_energy(atoms)
        force = -torch.autograd.grad(energy, coords_A, create_graph=False)[0]  # Eh/Å
        return energy, force

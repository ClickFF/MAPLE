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
      We only use: eps, born_scale, born_offset, array1 (shift), array2 (vdw_ref).
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
    array1 = allnums[:half]   # element-wise shift (Bohr-like, from your file)
    array2 = allnums[half:]   # element-wise vdw_ref (Bohr)

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
    GB polar term with geometry-dependent Born radii (OBC-II style).
    Input coords in Å, output energy in Hartree, forces in Hartree/Å.
    Nonpolar term removed per your request.
    """

    def __init__(self, solvent="water", device="cpu"):
        super().__init__()
        params = load_gbsa_params(solvent)
        self.device = torch.device(device)

        # dielectric
        self.eps = float(params["eps"])
        # Still/GB dielectric prefactor. xTB uses keps = (1/eps - 1)/(1+alpbet).
        # 无 ALPB 时用最常见的 (1 - 1/eps)，与你之前保持一致。
        self.keps = float(1.0 - 1.0 / self.eps)

        # element parameters (in Bohr domain, as in your file)
        self.born_scale  = float(params["born_scale"])
        self.born_offset = float(params["born_offset"])
        self.register_buffer("born_shift",
                             torch.tensor(params["array1"], dtype=torch.float32, device=self.device))
        self.register_buffer("vdw_ref",
                             torch.tensor(params["array2"], dtype=torch.float32, device=self.device))

        # OBC-II constants (standard, unitless)
        # These control how burial deepness increases effective Born radii.
        self.alpha = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.beta  = torch.tensor(0.8, dtype=torch.float32, device=self.device)
        self.gamma = torch.tensor(4.85, dtype=torch.float32, device=self.device)

        # small eps to keep numerics stable
        self.eps_dist = torch.tensor(1e-8, dtype=torch.float32, device=self.device)

    # ---- intrinsic radii from element table (Bohr) ----
    def intrinsic_radius(self, atom_index: torch.Tensor) -> torch.Tensor:
        """
        ri0: intrinsic Born-like radius from your element tables (Bohr).
        ri0 = born_scale * (vdw_ref[Z] + born_shift[Z]) + born_offset
        """
        vdwr  = self.vdw_ref[atom_index]      # Bohr
        shift = self.born_shift[atom_index]   # Bohr
        ri0 = self.born_scale * (vdwr + shift) + self.born_offset
        # physical lower bound
        ri0 = torch.clamp(ri0, min=0.5)
        return ri0

    # ---- geometry-dependent Born radii via OBC-II ----
    @torch.no_grad()
    def _mask_self(self, M: torch.Tensor) -> torch.Tensor:
        """Set diagonal to zero in-place-safe way for no-grad contexts."""
        return M.fill_diagonal_(0.)

    def compute_born_radius(self, coords_B: torch.Tensor, atom_index: torch.Tensor) -> torch.Tensor:
        """
        OBC-II effective radii (Bohr), differentiable w.r.t. coords_B via ri0 held constant in this step.
        If you want full dRi/dR gradients, remove no_grad contexts and avoid explicit detach.
        """
        ri0 = self.intrinsic_radius(atom_index)          # (N,)
        N   = coords_B.size(0)

        # pairwise distances in Bohr
        Rij = torch.cdist(coords_B, coords_B, p=2)       # (N,N)
        Rij = torch.clamp(Rij, min=self.eps_dist)        # avoid 0

        # volumes ~ r^3
        Vi = (ri0 ** 3)                                  # (N,)
        Vj = Vi.unsqueeze(0)                             # (1,N)

        # form Psi_i = sum_{j!=i}  V_j * exp(-Rij^2 / (4 ri0_i ri0_j)) / Rij^2
        ri0i = ri0.view(-1, 1)                           # (N,1)
        ri0j = ri0.view(1, -1)                           # (1,N)
        expo = torch.exp(- (Rij**2) / (4.0 * ri0i * ri0j + self.eps_dist))
        term = (Vj * expo) / (Rij**2)
        # zero self terms
        term = term - torch.diag_embed(torch.diag(term))
        Psi = torch.sum(term, dim=1)                     # (N,)

        # OBC-II transform: Ri = ri0 - ri0 * tanh( alpha*Psi - beta*Psi^2 + gamma*Psi^3 )
        poly = self.alpha * Psi - self.beta * (Psi**2) + self.gamma * (Psi**3)
        Ri = ri0 - ri0 * torch.tanh(poly)

        # lower bound to avoid degenerate f_ij
        Ri = torch.clamp(Ri, min=0.5)
        return Ri

    # ---- API ----
    def get_energy(self, atoms: Atoms):
        """
        Returns: energy (Eh), coords_A (Å, requires_grad=True)
        Only GB polar term. Nonpolar removed.
        """
        # charges
        q_np = atoms.atomic_charges
        if q_np is None or np.allclose(q_np, 0):
            raise ValueError("Atoms object must have nonzero partial charges.")
        q = torch.as_tensor(q_np, dtype=torch.float32, device=self.device)  # e

        # coordinates Å as leaf
        coords_A = torch.as_tensor(atoms.get_positions(),
                                   dtype=torch.float32,
                                   device=self.device)  # (N,3) Å
        coords_A.requires_grad_(True)
        coords_B = coords_A * ANG2BOHR                      # Bohr

        # atom indices
        Z = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        atom_index = Z - 1

        # geometry-dependent Born radii (Bohr)
        Ri = self.compute_born_radius(coords_B, atom_index)          # (N,)

        # Still f_ij
        Rij = torch.cdist(coords_B, coords_B, p=2) + self.eps_dist   # Bohr
        RiRj = Ri.view(-1, 1) * Ri.view(1, -1)                       # Bohr^2
        f_ij = torch.sqrt(Rij**2 + RiRj * torch.exp(-Rij**2 / (4.0 * RiRj + self.eps_dist)))  # Bohr

        # GB electrostatic energy
        q_i = q.view(-1, 1)
        q_j = q.view(1, -1)
        E_polar = -0.5 * self.keps * torch.sum((q_i * q_j) / f_ij)   # Eh

        return E_polar, coords_A

    def get_energy_and_force(self, atoms: Atoms):
        """Return energy (Hartree) and forces (Hartree/Å)."""
        energy, coords_A = self.get_energy(atoms)
        force = -torch.autograd.grad(energy, coords_A, create_graph=False)[0]  # Eh/Å
        return energy, force

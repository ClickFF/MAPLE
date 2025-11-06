import os

import torch
import numpy as np

class QEqTorch:
    """
    GPU-compatible Charge Equilibration (QEq) solver implemented with PyTorch.
    Reads per-element parameters (electronegativity, hardness, Gaussian radius)
    from ./data/qeq.dat and computes atomic charges for an ASE Atoms object.

    Reference:
      A.K. Rappé and W.A. Goddard III, J. Phys. Chem. 95 (1991): 3358–3363.
    """

    def __init__(self, data_file="./data/qeq.dat", device="cpu", eps0=1.0):
        """
        Args:
            data_file (str): Path to QEq parameter file.
            device (str): 'cpu' or 'cuda' for GPU acceleration.
            eps0 (float): Permittivity constant (default = 1 for atomic units).
        """
        self.device = torch.device(device)
        self.eps0 = eps0
        data_path = os.path.join(os.path.dirname(__file__), data_file)
        self.params = self._load_params(data_path)

    def _load_params(self, path):
        """Read QEq parameters from file: Element, Electronegativity(V), Hardness(V/e), Radius(Å)."""
        table = {}
        with open(path, "r") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                elt, chi, J, radius = line.split()[:4]
                table[elt] = {"chi": float(chi), "J": float(J), "sigma": float(radius)}
        return table

    def _get_param_tensor(self, symbols):
        """Convert element parameters into PyTorch tensors."""
        chi, J, sigma = [], [], []
        for s in symbols:
            if s not in self.params:
                raise ValueError(f"Element {s} not found in qeq.dat")
            p = self.params[s]
            chi.append(p["chi"])
            J.append(p["J"])
            sigma.append(p["sigma"])
        return (
            torch.tensor(chi, dtype=torch.float32, device=self.device),
            torch.tensor(J, dtype=torch.float32, device=self.device),
            torch.tensor(sigma, dtype=torch.float32, device=self.device),
        )

    def forward(self, atoms, total_charge=None):
        """
        Compute QEq charges for an ASE Atoms object.

        Args:
            atoms (ase.Atoms): ASE Atoms object.
            total_charge (float, optional): Total charge constraint.
                If None, uses atoms.get_initial_charge() or defaults to 0.0.

        Returns:
            torch.Tensor: Atomic charges (N,)
        """
        coords = torch.tensor(atoms.get_positions(), dtype=torch.float32, device=self.device)
        symbols = atoms.get_chemical_symbols()
        total_charge = 0.0 if total_charge is None else total_charge

        N = len(symbols)
        chi, J, sigma = self._get_param_tensor(symbols)

        # Build hardness matrix and RHS vector
        H = torch.zeros((N + 1, N + 1), device=self.device)
        V = torch.zeros(N + 1, device=self.device)

        # Diagonal: atomic hardness
        H[:N, :N] = torch.diag(J)

        # Off-diagonal: Gaussian-screened Coulomb interaction
        rij = torch.cdist(coords, coords, p=2) + 1e-6
        a = sigma.view(-1, 1)
        b = sigma.view(1, -1)
        p = torch.sqrt(a * b / (a**2 + b**2))
        coulomb = torch.erf(p * rij) / rij

        H[:N, :N] += (1.0 / (4 * np.pi * self.eps0)) * (coulomb - torch.diag(torch.diag(coulomb)))

        # Charge conservation constraint
        H[N, :N] = 1.0
        H[:N, N] = 1.0
        V[:N] = chi
        V[N] = total_charge

        # Solve linear system
        q = torch.linalg.solve(H, -V)
        return q[:-1].detach().cpu()

    __call__ = forward

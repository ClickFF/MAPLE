import os
import torch
import numpy as np
from typing import Sequence, Union
from ase.calculators.calculator import all_changes
from ..calculator_base import CalcABC
from typing import Literal

EV2HARTREE = 1.0 / 27.211386245988

# ------------------------ Basic helpers ------------------------

_SYMBOL2Z = {
    "H":1, "He":2, "Li":3, "Be":4, "B":5, "C":6, "N":7, "O":8, "F":9, "Ne":10,
    "Na":11, "Mg":12, "Al":13, "Si":14, "P":15, "S":16, "Cl":17, "Ar":18,
    "K":19, "Ca":20, "Sc":21, "Ti":22, "V":23, "Cr":24, "Mn":25, "Fe":26,
    "Co":27, "Ni":28, "Cu":29, "Zn":30, "Br":35, "I":53,
}

# Model name → filename mapping
_MACEPOL_MODEL_FILES = {
    'macepols': 'macepols.pt',
    'macepolm': 'macepolm.pt',
    'macepoll': 'macepoll.pt',
    'macepolefs': 'macepol-ef-s.pt',  # field-response fine-tuned (S size)
}


def _one_hot_node_attrs(Z: torch.Tensor, atomic_number_table: list, dtype=torch.float32) -> torch.Tensor:
    """Convert atomic numbers into one-hot vectors aligned with atomic_number_table."""
    table = torch.tensor(atomic_number_table, dtype=torch.long, device=Z.device)
    eq = (Z[:, None] == table[None, :])
    if not torch.all(eq.any(dim=1)):
        miss = Z[~eq.any(dim=1)].unique().tolist()
        raise ValueError(f"Atomic number(s) {miss} not in AtomicNumberTable {atomic_number_table}")
    return eq.to(dtype)


def _radius_graph_no_pbc(positions: torch.Tensor, r_max: float):
    """Construct O(N^2) radius graph without periodic boundaries."""
    N = positions.size(0)
    rij = positions[:, None, :] - positions[None, :, :]
    d2 = (rij * rij).sum(dim=-1)
    mask = torch.ones((N, N), dtype=torch.bool, device=positions.device)
    mask.fill_diagonal_(False)
    mask &= (d2 <= (r_max + 1e-12) ** 2)
    iu, ju = torch.nonzero(torch.triu(mask), as_tuple=True)
    src = torch.cat([iu, ju], dim=0)
    dst = torch.cat([ju, iu], dim=0)
    edge_index = torch.stack([src, dst], dim=0).to(torch.long)
    shifts = torch.zeros((edge_index.size(1), 3), dtype=positions.dtype, device=positions.device)
    return edge_index, shifts


# ------------------------ Calculator ------------------------

class MACEPolCalculator(CalcABC):
    """ASE calculator for MACE-POLAR models (pure MLIP, f32).

    The traced model accepts flat tensor inputs and returns:
        (total_energy, node_energy, density_coefficients)

    Supports total_charge and total_spin via atoms.info['charge'] and atoms.info['mult'].
    """

    implemented_properties = ['energy', 'forces', 'free_energy']
    supported_hessian_modes = ("analytic", "numerical")

    def __init__(self,
        device: torch.device,
        model: str = 'macepols',
        model_path: str = None,
        implicit: Literal["gbsa", "none"] = "gbsa",
        solvent: str = 'none',
        external_field=None,
        ):
        """
        Args:
            device: Torch device.
            model: Model name ('macepols', 'macepolm', 'macepoll').
            model_path: Optional explicit path to .pt file (overrides model name lookup).
            implicit: Implicit solvent model type.
            solvent: Solvent type.
        """
        super().__init__()

        # External uniform field default (V/A); per-atoms override via atoms.info["external_field"].
        # Accepts None | "Ex,Ey,Ez" str | (Ex,Ey,Ez) tuple/list | (3,) or (N,3) np.ndarray.
        self.external_field_default = self._parse_external_field(external_field)

        if model_path is None:
            model_dir = os.path.dirname(os.path.realpath(__file__))
            model_dir = os.path.dirname(model_dir)
            filename = _MACEPOL_MODEL_FILES.get(model, f'{model}.pt')
            model_path = os.path.join(model_dir, 'model', filename)

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.dtype = torch.float32  # MACE-POLAR traced models are f32
        self.r_max = float(self.model.r_max)
        self.atomic_numbers = [int(z) for z in self.model.atomic_numbers]
        self.hessian = "analytic"

        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    @staticmethod
    def _parse_external_field(ef):
        """Normalize user-supplied external_field to (3,) or (N,3) np.float32 array, or None."""
        if ef is None:
            return None
        if isinstance(ef, str):
            parts = [float(x) for x in ef.replace("(", "").replace(")", "").split(",")]
            if len(parts) != 3:
                raise ValueError(f"external_field string must have 3 comma-sep values; got: {ef}")
            return np.asarray(parts, dtype=np.float32)
        arr = np.asarray(ef, dtype=np.float32)
        if arr.shape == (3,) or (arr.ndim == 2 and arr.shape[1] == 3):
            return arr
        raise ValueError(f"external_field must be (3,) or (N,3); got shape {arr.shape}")

    def _build_inputs(self, atoms, requires_grad=False):
        """Build the 12 flat tensor inputs for MACE-POLAR forward pass."""
        device = self.device
        dtype = self.dtype

        positions = torch.tensor(
            atoms.get_positions(), dtype=dtype, device=device,
            requires_grad=requires_grad
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)
        node_attrs = _one_hot_node_attrs(Z, self.atomic_numbers, dtype=dtype)
        edge_index, shifts = _radius_graph_no_pbc(positions, self.r_max)

        N = positions.size(0)
        unit_shifts = torch.zeros_like(shifts)
        batch = torch.zeros(N, dtype=torch.int64, device=device)
        ptr = torch.tensor([0, N], dtype=torch.int64, device=device)
        cell = torch.zeros(3, 3, dtype=dtype, device=device)

        # Charge and spin from atoms.info (default: 0, singlet)
        charge = float(atoms.info.get('charge', 0))
        mult = int(atoms.info.get('mult', 1))
        spin = float(mult - 1)
        total_charge = torch.tensor([charge], dtype=dtype, device=device)
        total_spin = torch.tensor([spin], dtype=dtype, device=device)

        # External field resolution (V/A, native model unit):
        # 1) atoms.info["external_field"] (per-atoms override)
        # 2) self.external_field_default (constructor default)
        # 3) zeros (vacuum SP)
        ef = atoms.info.get("external_field", None)
        if ef is None:
            ef = self.external_field_default
        if ef is None:
            external_field = torch.zeros(N, 3, dtype=dtype, device=device)
        else:
            ef = np.asarray(ef, dtype=np.float32)
            if ef.shape == (3,):
                ef = np.broadcast_to(ef, (N, 3)).copy()
            elif ef.shape != (N, 3):
                raise ValueError(f"external_field must be (3,) or ({N},3); got shape {ef.shape}")
            external_field = torch.from_numpy(ef).to(device=device, dtype=dtype)
        local_or_ghost = torch.ones(N, dtype=dtype, device=device)

        return (positions, node_attrs, edge_index, shifts, unit_shifts,
                batch, ptr, cell, total_charge, total_spin,
                external_field, local_or_ghost)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        """Main ASE calculation entry point."""
        super().calculate(atoms, properties, system_changes)

        # Energy (no grad needed)
        inputs = self._build_inputs(atoms, requires_grad=False)
        with torch.no_grad():
            total_energy, node_energy, density_coef = self.model(*inputs)

        energy = total_energy.sum().double() * EV2HARTREE

        if self.solvent_correction:
            solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()

        # Forces via autograd
        if 'forces' in properties:
            inputs_grad = self._build_inputs(atoms, requires_grad=True)
            total_energy_grad, _, _ = self.model(*inputs_grad)

            forces = -torch.autograd.grad(
                total_energy_grad.sum(),
                inputs_grad[0],  # positions
                create_graph=False,
                retain_graph=False
            )[0]
            forces = forces.double() * EV2HARTREE

            if self.solvent_correction:
                _, solvent_force = self.implicit_solv_energy_and_force(atoms)
                forces = forces + solvent_force

            self.results['forces'] = forces.detach().cpu().numpy()

        if 'hessian' in properties:
            if self.solvent_correction:
                raise NotImplementedError("Hessian calculation with implicit solvent is not implemented yet.")
            self.results['hessian'] = self.get_hessian(atoms)

    def get_energy(self, atoms) -> torch.Tensor:
        """Compute total energy as a torch scalar (eV)."""
        inputs = self._build_inputs(atoms, requires_grad=False)
        with torch.no_grad():
            total_energy, _, _ = self.model(*inputs)
        return total_energy.sum()

    @staticmethod
    def compute_hessian(coords: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        """Compute the Cartesian Hessian matrix (3N x 3N)."""
        num_atoms = coords.shape[0]
        hessian = torch.zeros((3 * num_atoms, 3 * num_atoms), dtype=coords.dtype, device=coords.device)
        grad = torch.autograd.grad(energy, coords, create_graph=True)[0].view(-1)
        for i in range(3 * num_atoms):
            grad2 = torch.autograd.grad(grad[i], coords, retain_graph=True)[0].view(-1)
            hessian[i, :] = grad2
        return hessian

    def _get_hessian_analytic(self, atoms) -> np.ndarray:
        if atoms is None:
            atoms = self.atoms

        inputs = self._build_inputs(atoms, requires_grad=True)
        total_energy, _, _ = self.model(*inputs)
        energy = total_energy.sum() * EV2HARTREE

        hessian = self.compute_hessian(inputs[0], energy)
        return hessian.detach().cpu().numpy()

    def _get_hessian_numerical(self, atoms, delta: float = 0.002) -> np.ndarray:
        from ase.constraints import FixAtoms

        N = len(atoms)
        pos0 = atoms.get_positions().copy()
        fixed = {
            i for c in getattr(atoms, "constraints", [])
            if isinstance(c, FixAtoms)
            for i in c.get_indices()
        }
        movable = [i for i in range(N) if i not in fixed]
        H = np.zeros((3 * N, 3 * N), dtype=np.float64)

        if len(movable) == 0:
            return H

        def force_at(positions: np.ndarray) -> np.ndarray:
            atoms_tmp = atoms.copy()
            atoms_tmp.set_positions(positions)
            if getattr(atoms, "constraints", None):
                atoms_tmp.set_constraint(atoms.constraints)
            self.calculate(atoms_tmp, properties=["forces"], system_changes=all_changes)
            return np.asarray(self.results["forces"], dtype=np.float64)

        for a in movable:
            for k in range(3):
                row = 3 * a + k
                pos_p = pos0.copy()
                pos_p[a, k] += delta
                Fp = force_at(pos_p)

                pos_m = pos0.copy()
                pos_m[a, k] -= delta
                Fm = force_at(pos_m)

                H[row, :] = (-(Fp - Fm) / (2.0 * delta)).reshape(-1)

        return H

    def get_hessian(self, atoms=None, delta: float = 0.002) -> np.ndarray:
        """Compute the Cartesian Hessian matrix for an ASE Atoms object."""
        if self.hessian == "analytic":
            return self._get_hessian_analytic(atoms)
        if self.hessian == "numerical":
            return self._get_hessian_numerical(atoms, delta)
        raise ValueError(
            f"Unknown hessian method: {self.hessian}. Must be 'analytic' or 'numerical'"
        )

import os
import torch
import numpy as np
from typing import Dict, Literal
from ase.calculators.calculator import Calculator, all_changes
from ..calculator_base import CalcABC


EV2HARTREE = 1.0 / 27.211386245988

# --------------------------------------------
# Build dense neighbor list (N+1, M) sentinel padded
# --------------------------------------------
def nblist_dense_padded(coord: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Brute-force dense neighbor list for single molecule.
    Returns: (N+1, M) int32 tensor, sentinel row = N
    """
    device = coord.device
    N = coord.shape[0]
    if N == 0:
        return torch.full((1, 1), 0, dtype=torch.int32, device=device)

    diff = coord[:, None, :] - coord[None, :, :]
    dist2 = torch.sum(diff ** 2, dim=-1)
    dist2[torch.eye(N, dtype=torch.bool, device=device)] = float('inf')
    mask = dist2 <= cutoff ** 2
    M = max(int(mask.sum(dim=1).max().item()), 1)

    nbmat = torch.full((N + 1, M), N, dtype=torch.int32, device=device)
    for i in range(N):
        nb_i = torch.nonzero(mask[i], as_tuple=False).flatten()
        if nb_i.numel() > 0:
            nbmat[i, :min(nb_i.numel(), M)] = nb_i[:min(nb_i.numel(), M)]
    return nbmat

# --------------------------------------------
# Pad helpers
# --------------------------------------------
def pad_dim0(a: torch.Tensor, value=0.0) -> torch.Tensor:
    """
    Pad one row along dim0.
    For (N, C) -> (N+1, C), (N,) -> (N+1,)
    """
    pad_shape = list(a.shape)
    pad_shape[0] = 1
    pad_row = torch.full(pad_shape, value, dtype=a.dtype, device=a.device)
    return torch.cat([a, pad_row], dim=0)

def maybe_pad_dim0(a: torch.Tensor, N: int, value=0.0) -> torch.Tensor:
    """
    If a.shape[0] == N, return as is.
    If a.shape[0] == N-1, pad one row to length N.
    """
    diff = N - a.shape[0]
    assert diff in (0, 1), f"Invalid pad: {a.shape[0]} vs target {N}"
    if diff == 1:
        a = pad_dim0(a, value=value)
    return a

# ==========================================================
# AIMNet2 Calculator (single-molecule minimal version)
# ==========================================================
class AIMNet2Calculator(CalcABC):
    implemented_properties = ["energy", "forces", "hessian", "free_energy"]

    def __init__(self, device: torch.device, 
                model: str = "aimnet2", 
                coulomb_method: str = "simple",
                implicit: Literal["gbsa", "none"] = "gbsa",
                solvent: str = 'none',
                ):
        super().__init__()
        self.device = device

        # Load model
        model_dir = os.path.dirname(os.path.realpath(__file__))
        model_dir = os.path.dirname(model_dir)
        model_path = os.path.join(model_dir, "model", f"{model}.pt")
        self.model = torch.jit.load(model_path, map_location=device).eval()

        self.cutoff = float(getattr(self.model, "cutoff"))
        # CHANGED: do not rely on hasattr(model, 'cutoff_lr') to decide LR; model may still need nbmat_lr
        self.cutoff_lr = float(getattr(self.model, "cutoff_lr", float("inf")))
        # keep a flag (optional); but we will always provide nbmat_lr anyway
        self.lr = True  # CHANGED: force-true to avoid conditional omission

        # Coulomb settings (keep original behavior)
        self._set_lrcoulomb_method(coulomb_method)

        # Initialize implicit solvent
        self.implicit_solv_init(implicit=implicit, solvent=solvent)

    def _set_lrcoulomb_method(self, method: str, cutoff: float = 15.0, dsf_alpha: float = 0.2):
            """
            Configure the long-range Coulomb interaction method if the model contains a 'lrcoulomb' submodule.
            method: 'simple', 'dsf', or 'ewald'
            cutoff: cutoff distance for long-range interactions
            dsf_alpha: DSF damping parameter (if used)
            """
            assert method in ("simple", "dsf", "ewald"), f"Invalid method: {method}"

            # recursively look for 'lrcoulomb' submodules
            def _iter_lrcoulomb_mods(model):
                for name, mod in model.named_modules():
                    if name == "lrcoulomb":
                        yield mod

            for mod in _iter_lrcoulomb_mods(self.model):
                mod.method = method
                if method == "dsf" and hasattr(mod, "dsf_alpha"):
                    mod.dsf_alpha = dsf_alpha

            # update cutoff_lr based on the chosen method
            self.cutoff_lr = float("inf") if method == "simple" else float(cutoff)
            self._coulomb_method = method

    # ------------------------ calculate ------------------------
    def calculate(self, atoms=None, properties=["energy", "forces", "free_energy", "hessian"], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=("forces" in properties or "hessian" in properties)
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)

        N = coord.shape[0]

        nbmat = nblist_dense_padded(coord, self.cutoff)
        data: Dict[str, torch.Tensor] = {
            "coord": pad_dim0(coord, value=0.0),         # (N+1, 3)
            "numbers": pad_dim0(Z, value=0),             # (N+1,)
            "charge": torch.tensor([0.0], dtype=torch.float32, device=self.device),
            "mol_idx": pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            "nbmat": nbmat,
        }

        # CHANGED: ALWAYS provide nbmat_lr + cutoff_lr to avoid KeyError inside TorchScript
        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data["nbmat_lr"] = nblist_dense_padded(coord, lr_cutoff)
        data["cutoff_lr"] = torch.tensor(lr_cutoff, device=self.device)

        energy = self.get_energy(data)
        energy = energy * EV2HARTREE  # to Hartree

        if self.solvent_correction:
            solvent_energy = self.implicit_solv_energy(atoms)
            energy += solvent_energy

        self.results["energy"] = float(energy.item())
        self.results["free_energy"] = float(energy.item())

        if "forces" in properties:
            grad_full = torch.autograd.grad(
                energy, data["coord"], create_graph=("hessian" in properties)
            )[0]                      # (N+1, 3)
            forces = -grad_full[:N]   # (N, 3)
            if self.solvent_correction:
                solvent_energy, solvent_force = self.implicit_solv_energy_and_force(atoms)
                forces += solvent_force

            self.results["forces"] = forces.detach().cpu().numpy()
            
        if "hessian" in properties:
            if self.solvent_correction:
                raise NotImplementedError("Hessian calculation with implicit solvent is not implemented yet.")
            self.results["hessian"] = self.get_hessian(atoms)

    # ------------------------ get_energy ------------------------
    def get_energy(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.jit.optimized_execution(False):
            out = self.model(data)
        return out["energy"].sum()

    # ------------------------ get_hessian ------------------------
    def get_hessian(self, atoms) -> np.ndarray:
        coord = torch.tensor(
            atoms.get_positions(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32, device=self.device)
        mol_idx = torch.zeros(coord.shape[0], dtype=torch.int32, device=self.device)

        N = coord.shape[0]
        nbmat = nblist_dense_padded(coord, self.cutoff)
        data = {
            "coord": pad_dim0(coord, value=0.0),               # (N+1, 3)
            "numbers": pad_dim0(Z, value=0),                   # (N+1,)
            "charge": torch.tensor([0.0], dtype=torch.float32, device=self.device),
            "mol_idx": pad_dim0(mol_idx, value=mol_idx[-1].item() if N > 0 else 0),
            "nbmat": nbmat,
        }

        # CHANGED: ALWAYS provide nbmat_lr + cutoff_lr
        lr_cutoff = self.cutoff_lr if np.isfinite(self.cutoff_lr) else self.cutoff
        data["nbmat_lr"] = nblist_dense_padded(coord, lr_cutoff)
        data["cutoff_lr"] = torch.tensor(lr_cutoff, device=self.device)

        energy = self.get_energy(data)
        energy = energy * EV2HARTREE

        forces_full = torch.autograd.grad(energy, data["coord"], create_graph=True)[0]  # (N+1, 3)
        forces = -forces_full[:N]  # (N, 3)

        # Use original-style assembly to stay consistent with AIMNet2 padding
        hessian = - torch.stack([
            torch.autograd.grad(f, data["coord"], retain_graph=True)[0]
            for f in forces.flatten().unbind()
        ]).view(-1, 3, N + 1, 3)[:, :, :N, :]  # slice out the padded row on atom-axis

        return hessian.detach().cpu().numpy().reshape(3 * N, 3 * N)

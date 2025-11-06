import os
import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms


def parse_pdb_residue_groups(pdbfile):
    """Parse PDB pure solvent template into groups using TER."""
    coords = []
    elems = []
    groups = []
    current_group = []

    with open(pdbfile) as f:
        for line in f:
            if line.startswith("ATOM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() or line[12:14].strip()
                coords.append([x, y, z])
                elems.append(elem)
                current_group.append(len(coords) - 1)
            elif line.startswith("TER"):
                if len(current_group) > 0:
                    groups.append(current_group)
                    current_group = []

    # ensure last group added
    if len(current_group) > 0:
        groups.append(current_group)

    coords = np.array(coords)
    elems = np.array(elems)
    return coords, elems, groups


class ExplicitSolv():
    def __new__(cls, atoms: Atoms, params: dict, device: torch.device, output: str):
        obj = super().__new__(cls)
        obj.device = device
        obj.output = output
        obj.atoms = atoms
        obj.radius = params.get('radius', 10.0)
        obj.solv_name = params.get('explicit', 'water')
        obj.clash_cutoff = params.get('clash_cutoff', 1.5)
        obj.fix_dis = params.get('fix_dis', 8.0)

        obj.data_path = os.path.join(os.path.dirname(__file__), "data", f"{obj.solv_name}.pdb")
        
        if not os.path.isfile(obj.data_path):
            obj.log_error([f"Solvent {obj.solv_name} is not found"])
            raise FileNotFoundError(f"Solvent {obj.solv_name} is not found")

        info_message = [
            "\n\n" + "-" * 70 + "\n",
            f"{'Explicit Solvation Setup'.center(70)}\n\n",
            f"• Solvent type: {obj.solv_name}\n",
            f"• Solvation sphere radius: {obj.radius:.2f} Å\n",
            f"• Remove solvent molecules within {obj.clash_cutoff:.2f} Å of any solute atom\n",
            f"• Fix solvent molecules whose atoms lie beyond {obj.fix_dis:.2f} Å from solute center\n\n",
        ]

        obj.log_info(info_message)
 
        
        obj._process()

        info_message = [
        "\n" + "-" * 70 + "\n"
        ]

        obj.log_info(info_message)

        return obj.atoms


    def log_error(self, error_message: str) -> None:
        """
        Logs error messages to the output file.

        Args:
            error_message: The error message to log.
        """
        with open(self.output, 'a') as file:
            file.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """
        Logs info messages to the output file.

        Args:
            info_message: The info message to log.
        """
        with open(self.output, 'a') as file:
            for info in info_message:   
                file.write(f"{info}")

    def _get_box_length(self):
        with open(self.data_path) as f:
            for line in f:
                if line.startswith("CRYST1"):
                    a, b, c = map(float, line.split()[1:4])
                    return a, b, c
        raise RuntimeError("CRYST1 missing in solvent template")


    def _tile_box(self):
        Lx, Ly, Lz = self._get_box_length()
        coords, elem, groups = parse_pdb_residue_groups(self.data_path)

        # Assign residue tags by TER grouping
        tags = np.zeros(len(coords), dtype=int)
        for rid, grp in enumerate(groups):
            tags[grp] = rid

        nx = ny = nz = 1
        while min(nx*Lx, ny*Ly, nz*Lz)/2 < self.radius:
            nx += 1
            ny += 1
            nz += 1

        coords_all = []
        elem_all = []
        tags_all = []

        base_residue_count = len(groups)

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    shift = np.array([i*Lx, j*Ly, k*Lz])
                    coords_all.append(coords + shift)
                    elem_all.extend(elem)
                    tags_all.append(tags + base_residue_count*(i*ny*nz + j*nz + k))

        coords_full = np.vstack(coords_all)
        tags_full = np.concatenate(tags_all)

        box = Atoms(symbols=list(elem_all), positions=coords_full)
        box.set_tags(tags_full)
        return box


    def _process(self):
        box = self._tile_box()

        S = torch.tensor(box.positions, device=self.device, dtype=torch.float32)
        M = torch.tensor(self.atoms.positions, device=self.device, dtype=torch.float32)
        tag_s = torch.tensor(box.get_tags(), device=self.device, dtype=torch.long)

        # ---- Step1: center molecule ----
        shift = S.mean(0) - M.mean(0)
        M = M + shift
        self.atoms.positions += shift.cpu().numpy()
        box_center = S.mean(0)

        # ---- Step2: radius mask ----
        dist2 = ((S - box_center)**2).sum(-1)
        atom_inside = dist2 <= self.radius**2

        max_tag = tag_s.max().item() + 1
        res_inside = torch.zeros(max_tag, dtype=torch.bool, device=self.device)
        res_inside.index_put_((tag_s,), atom_inside, accumulate=True)
        mask_radius = res_inside[tag_s]

        S = S[mask_radius]
        tag_s = tag_s[mask_radius]
        elem = np.array(box.get_chemical_symbols())[mask_radius.cpu().numpy()]

        # ---- Step3: clash mask (residue AND strict) ----
        dmat = torch.cdist(S, M)
        min_d = dmat.min(1).values
        atom_ok = min_d >= self.clash_cutoff

        max_tag2 = tag_s.max().item() + 1
        res_total = torch.bincount(tag_s, minlength=max_tag2)
        res_okcnt = torch.bincount(tag_s, weights=atom_ok.float(), minlength=max_tag2)

        residue_keep = (res_okcnt == res_total)
        mask_clash = residue_keep[tag_s]

        # final solvent mask
        mask_final = mask_clash

        # final solvent positions
        S_final = S[mask_final]
        elem_final = elem[mask_final.cpu().numpy()]
        tag_final = tag_s[mask_final]

        # ---- Step4: Add solvent
        start = len(self.atoms)
        self.atoms += Atoms(symbols=list(elem_final), positions=S_final.cpu().numpy())
        self.log_info([f"Added {len(S_final)} solvent atoms.\n"])

        # ---- Step5: Fix residues beyond fix_thres ----
        dist2_final = ((S_final - box_center)**2).sum(-1)
        fix_thres = self.fix_dis
        atom_far = dist2_final > fix_thres**2

        max_tag3 = tag_final.max().item() + 1
        res_total2 = torch.bincount(tag_final, minlength=max_tag3)
        res_far_cnt = torch.bincount(tag_final, weights=atom_far.float(), minlength=max_tag3)

        residue_fix = (res_far_cnt > 0)
        fix_mask = residue_fix[tag_final]
        fix_indices = start + torch.where(fix_mask)[0].cpu().numpy()

        if len(fix_indices) > 0:
            self.atoms.set_constraint(FixAtoms(indices=list(fix_indices)))

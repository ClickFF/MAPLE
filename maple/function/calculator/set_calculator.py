import urllib.request
import shutil
import torch
from pathlib import Path
from typing import Optional

import ase
from ase import Atoms

from .ani._ani_calculator import ANICalculator
from .mace._mace_calculator import MACECalculator

# ---------------------------------------------------------------------------
# Supported model identifiers
# ---------------------------------------------------------------------------

IMPLEMENTATION_MODELs = [
    'ani2x', 'ani1x', 'ani1ccx', 'ani1xnr',
    'maceoff23s', 'maceoff23m', 'maceoff23l', 'egret',
    'aimnet2', 'aimnet2nse',
    'uma',
    'maceomol',
]

# Models whose weights are stored locally in calculator/model/<filename>.pt
# and can be auto-downloaded from HuggingFace when absent.
_MODEL_FILE = {
    'ani2x':      'ani2x.pt',
    'ani1x':      'ani1x.pt',
    'ani1ccx':    'ani1ccx.pt',
    'ani1xnr':    'ani1xnr.pt',
    'aimnet2':    'aimnet2.pt',
    'aimnet2nse': 'aimnet2nse.pt',
    'maceoff23m': 'maceoff23m.pt',
    'maceomol':   'maceomol.pt',
    'egret':      'egret1s.pt',
    'uma-s-1p1':  'uma-s-1p1.pt',
    'uma-m-1p1':  'uma-m-1p1.pt',
}

_HF_REPO = "Wayne7815/MAPLE_model"
_MODEL_DIR = Path(__file__).parent / "model"


class SetClaculator():

    def __init__(
        self,
        device: torch.device,
        model: str,
        output: str,
        atoms: Optional[Atoms] = None,
        d4: bool = False,
        implicit: str = 'None',
        solvent: str = 'None',
        model_params: Optional[dict] = None,
    ) -> None:
        self.output      = output
        self.model       = model
        self.d4          = d4
        self.device      = device
        self.atoms       = atoms
        self.implicit    = implicit
        self.solvent     = solvent
        self.model_params = model_params

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        if self.model not in IMPLEMENTATION_MODELs:
            self.log_error(f"\n [ERROR] Unsupported model: {self.model}\n")
            raise ValueError(f"Unsupported model: '{self.model}'.")

        if self.d4 and self.model not in ['ani2x', 'ani1x', 'ani1ccx', 'ani1xnr']:
            self.log_info([f"\n [WARNING] D4 is not supported for model '{self.model}'. D4 will be ignored.\n"])

        calculator = self._build_calculator()
        self._warn_charge_mult(calculator)
        return calculator

    # ------------------------------------------------------------------
    # Model file management
    # ------------------------------------------------------------------

    def _ensure_model_file(self, key: str) -> Path:
        """Return path to a local model file, downloading it if absent."""
        filename = _MODEL_FILE.get(key)
        if filename is None:
            return None

        path = _MODEL_DIR / filename
        if path.exists():
            return path

        # Auto-download from HuggingFace
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{_HF_REPO}/resolve/main/{filename}"
        self.log_info([f" [INFO] Model file '{filename}' not found locally.\n",
                       f" [INFO] Downloading from: {url}\n"])
        try:
            tmp = path.with_suffix('.tmp')
            with urllib.request.urlopen(url) as resp:
                size = resp.headers.get('Content-Length')
                if size:
                    self.log_info([f" [INFO] File size: {int(size) / 1024 / 1024:.1f} MB\n"])
                with open(tmp, 'wb') as f:
                    shutil.copyfileobj(resp, f)
            tmp.rename(path)
            self.log_info([f" [INFO] Download complete: {path}\n"])
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            self.log_error(f" [ERROR] Download failed (HTTP {e.code}): {url}\n")
            raise RuntimeError(f"Failed to download model '{key}': HTTP {e.code}")
        except urllib.error.URLError as e:
            tmp.unlink(missing_ok=True)
            self.log_error(f" [ERROR] Network error: {e.reason}\n")
            raise RuntimeError(f"Failed to download model '{key}': {e.reason}")

        return path

    # ------------------------------------------------------------------
    # Calculator construction
    # ------------------------------------------------------------------

    def _build_calculator(self) -> ase.calculators.calculator.Calculator:
        model = self.model

        if model in ['ani2x', 'ani1x', 'ani1ccx', 'ani1xnr']:
            self._ensure_model_file(model)
            return ANICalculator(
                model=model, d4=self.d4, device=self.device,
                implicit=self.implicit, solvent=self.solvent,
            )

        if model in ['maceoff23s', 'maceoff23m', 'maceoff23l', 'egret']:
            self._ensure_model_file(model)
            from .mace._mace_calculator import MACECalculator
            return MACECalculator(
                model=model, device=self.device,
                implicit=self.implicit, solvent=self.solvent,
            )

        if model in ['aimnet2', 'aimnet2nse']:
            self._ensure_model_file(model)
            from .aimnet._aimnet2_calculator import AIMNet2Calculator
            return AIMNet2Calculator(
                model=model, device=self.device,
                implicit=self.implicit, solvent=self.solvent,
            )

        if model == 'uma':
            uma_task = None
            uma_size = None
            if self.model_params is not None:
                uma_task = self.model_params.get('task')
                uma_size = self.model_params.get('size')
            # Resolve the model size key for local file lookup
            size_key = uma_size if uma_size else 'uma-s-1p1'
            model_path = self._ensure_model_file(size_key)
            from .uma._uma_calculator import UMACalculator
            return UMACalculator(
                model=model, device=self.device,
                implicit=self.implicit, solvent=self.solvent,
                task=uma_task, size=uma_size,
                checkpoint_path=str(model_path),
            )

        if model == 'maceomol':
            self._ensure_model_file(model)
            from .mace._mace_general_calculator import MACEModelCalculator
            return MACEModelCalculator(
                model=model, device=self.device,
                implicit=self.implicit, solvent=self.solvent,
            )

        raise ValueError(f"Model '{model}' is not implemented yet.")

    # ------------------------------------------------------------------
    # Charge / multiplicity compatibility warning
    # ------------------------------------------------------------------

    _NO_CHARGE_MULT = {
        'ani2x', 'ani1x', 'ani1ccx', 'ani1xnr',
        'maceoff23s', 'maceoff23m', 'maceoff23l', 'egret', 'maceomol',
    }

    def _warn_charge_mult(self, calculator) -> None:
        if self.atoms is None:
            return
        has_charge = self.atoms.info.get('charge', 0) != 0
        has_mult   = self.atoms.info.get('mult',   1) != 1
        if (has_charge or has_mult) and self.model in self._NO_CHARGE_MULT:
            self.log_info([
                f"\n [WARNING] Model '{self.model}' does not support charge/multiplicity.\n"
                f"           charge={self.atoms.info.get('charge', 0)}, "
                f"mult={self.atoms.info.get('mult', 1)} will be IGNORED.\n"
                f"           Models with charge/mult support: aimnet2, aimnet2nse, uma\n"
            ])

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_error(self, error_message: str) -> None:
        """Log an error message to the output file."""
        with open(self.output, 'a') as f:
            f.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        """Log info messages to the output file."""
        with open(self.output, 'a') as f:
            for line in info_message:
                f.write(line)

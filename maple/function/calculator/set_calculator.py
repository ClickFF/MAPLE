import torch
from typing import Optional

import ase
from ase import Atoms

from pathlib import Path

from .ani._ani_calculator import ANICalculator
from .mace._mace_calculator import MACECalculator

IMPLEMENTATION_MODELs = [
            'ani2x',
            'ani1x',
            'ani1ccx',
            'ani1xnr',
            'maceoff23s',
            'maceoff23m',
            'maceoff23l',
            'egret',
            'aimnet2',
            'uma',
            'maceomol',
            'aimnet2nse',
        ]

# Model name to filename mapping for HuggingFace download
MODEL_NAME_TO_FILE = {
    'ani2x': 'ani2x.pt',
    'ani1x': 'ani1x.pt',
    'ani1ccx': 'ani1ccx.pt',
    'ani1xnr': 'ani1xnr.pt',
    'aimnet2': 'aimnet2.pt',
    'aimnet2nse': 'aimnet2nse.pt',
    'maceoff23m': 'maceoff23m.pt',
    'maceomol': 'maceomol.pt',
    'egret': 'egret1s.pt',
    'uma': 'uma-s-1p1.pt',
}

HF_REPO_ID = "Wayne7815/MAPLE_model"

class SetClaculator():

    def __init__(self, device: torch.device,
            model:str,
            output: str,
            atoms: Optional[Atoms] = None,
            d4:bool=False,
            implicit:str = 'None',
            solvent: str = 'None') -> None:
        self.output = output
        self.model = model
        self.d4 = d4
        self.device = device
        self.model = model
        self.atoms = atoms
        self.implicit = implicit
        self.solvent = solvent

    def _get_model_dir(self) -> Path:
        """Get model directory path (model folder in the same directory as this script)"""
        current_file_dir = Path(__file__).parent
        model_dir = current_file_dir / "model"
        return model_dir

    def _download_model(self, model_name: str) -> Path:
        """
        Download model from HuggingFace if not exists locally.

        Args:
            model_name: Model name (e.g., 'ani2x', 'aimnet2')

        Returns:
            Path to the model file
        """
        if model_name not in MODEL_NAME_TO_FILE:
            return None

        model_dir = self._get_model_dir()
        model_filename = MODEL_NAME_TO_FILE[model_name]
        model_path = model_dir / model_filename

        # Create model directory if not exists
        if not model_dir.exists():
            self.log_info([f" [INFO] Creating model directory: {model_dir}\n"])
            model_dir.mkdir(parents=True, exist_ok=True)

        # Download if model file not exists
        if not model_path.exists():
            download_url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{model_filename}"
            self.log_info([f" [INFO] Model {model_filename} not found locally.\n"])
            self.log_info([f" [INFO] Downloading from: {download_url}\n"])

            try:
                with urllib.request.urlopen(download_url) as response:
                    total_size = response.headers.get('Content-Length')
                    if total_size:
                        total_size = int(total_size)
                        self.log_info([f" [INFO] File size: {total_size / 1024 / 1024:.2f} MB\n"])

                    temp_path = model_path.with_suffix('.tmp')
                    with open(temp_path, 'wb') as f:
                        shutil.copyfileobj(response, f)
                    temp_path.rename(model_path)

                self.log_info([f" [INFO] Download complete: {model_path}\n"])

            except urllib.error.HTTPError as e:
                error_message = f" [ERROR] Download failed (HTTP {e.code}): {download_url}\n"
                self.log_error(error_message)
                raise RuntimeError(error_message)
            except urllib.error.URLError as e:
                error_message = f" [ERROR] Network error: {e.reason}\n"
                self.log_error(error_message)
                raise RuntimeError(error_message)

        return model_path

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        if self.model not in IMPLEMENTATION_MODELs:
            error_message = f"\n [ERROR] Unsupported model: {self.model}\n"
            self.log_error(error_message)
            raise ValueError(error_message)

        # Initialize calculator based on model
        if self.model in ['ani2x', 'ani1x', 'ani1ccx', 'ani1xnr']:
            calculator = ANICalculator(model=self.model, d4=self.d4, device=self.device, implicit = self.implicit, solvent = self.solvent)
        else:
            if self.d4 == True : self.log_info([f"\n [WARNING:] D4 is not supported for model {self.model}. D4 will be ignored.\n"])
            if self.model in ['maceoff23s', 'maceoff23m', 'maceoff23l','egret']:
                from .mace._mace_calculator import MACECalculator
                calculator = MACECalculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
            elif self.model in ['aimnet2', 'aimnet2nse']:
                from .aimnet._aimnet2_calculator import AIMNet2Calculator
                calculator = AIMNet2Calculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
            elif self.model in ['uma']:
                from .uma._uma_calculator import UMACalculator
                uma_model_path = self._download_model('uma')
                calculator = UMACalculator(model_path=uma_model_path, device=self.device, implicit=self.implicit, solvent=self.solvent)
            elif self.model in ['maceomol']:
                from .mace._mace_general_calculator import MACEModelCalculator
                calculator = MACEModelCalculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
            else:
                raise ValueError(f"Model '{self.model}' is not implemented yet.")

        # Check charge/mult compatibility and issue warning if needed
        if self.atoms is not None:
            has_charge = 'charge' in self.atoms.info and self.atoms.info['charge'] != 0
            has_mult = 'mult' in self.atoms.info and self.atoms.info['mult'] != 1

            if has_charge or has_mult:
                # Models that do NOT support charge/mult
                unsupported_models = ['ani2x', 'ani1x', 'ani1ccx', 'ani1xnr',
                                     'maceoff23s', 'maceoff23m', 'maceoff23l',
                                     'egret', 'maceomol']

                if self.model in unsupported_models:
                    charge_val = self.atoms.info.get('charge', 0)
                    mult_val = self.atoms.info.get('mult', 1)
                    self.log_info([
                        f"\n [WARNING] Model '{self.model}' does not support charge ({charge_val}) "
                        f"and multiplicity ({mult_val}) parameters.\n"
                        f"           These values will be IGNORED by the calculator.\n"
                        f"           Supported models: aimnet2, aimnet2nse, uma\n"
                    ])

        return calculator
        

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
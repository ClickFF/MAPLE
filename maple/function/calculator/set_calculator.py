import torch
import urllib.request
import shutil

import ase

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
            'dpa3',
            'chgnet',
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
}

HF_REPO_ID = "Wayne7815/MAPLE_model"

class SetClaculator():

    def __init__(self, device: torch.device,
            model:str, 
            output: str, 
            d4:bool=False,
            implicit:str = 'None',
            solvent: str = 'None') -> None:
        self.output = output
        self.model = model
        self.d4 = d4
        self.device = device
        self.model = model
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

        # Check and download model if needed
        if self.model in MODEL_NAME_TO_FILE:
            self._download_model(self.model)

        if self.model in ['ani2x', 'ani1x', 'ani1ccx', 'ani1xnr']:
            calculator = ANICalculator(model=self.model, d4=self.d4, device=self.device, implicit = self.implicit, solvent = self.solvent)
            return calculator
        else:
            if self.d4 == True : self.log_info([f"\n [WARNING:] D4 is not supported for model {self.model}. D4 will be ignored.\n"])
            if self.model in ['maceoff23s', 'maceoff23m', 'maceoff23l','egret']:
                from .mace._mace_calculator import MACECalculator
                calculator = MACECalculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
                return calculator   
            elif self.model in ['aimnet2', 'aimnet2nse']:
                from .aimnet._aimnet2_calculator import AIMNet2Calculator
                calculator = AIMNet2Calculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
                return calculator
            elif self.model in ['uma']:
                from .uma._uma_calculator import UMACalculator
                calculator = UMACalculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
                return calculator
            elif self.model in ['maceomol']:
                from .mace._mace_general_calculator import MACEModelCalculator
                calculator = MACEModelCalculator(model=self.model, device=self.device, implicit = self.implicit, solvent = self.solvent)
                return calculator
            elif self.model == 'dpa3':
                from .deepmd._dpa3_calculator import DPA3Calculator
                calculator = DPA3Calculator(model=self.model, device=self.device, implicit=self.implicit, solvent=self.solvent)
                return calculator
            elif self.model == 'chgnet':
                from .chgnet._chgnet_calculator import CHGNetCalc
                calculator = CHGNetCalc(model=self.model, device=self.device, implicit=self.implicit, solvent=self.solvent)
                return calculator
            else:
                raise ValueError(f"Model '{self.model}' is not implemented yet.")
        

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
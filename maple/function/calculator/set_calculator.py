import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import ase
import torch
from ase import Atoms

from .ani._ani_calculator import ANICalculator
from .mace._mace_calculator import MACECalculator


IMPLEMENTATION_MODELS = [
    "ani2x",
    "ani1x",
    "ani1ccx",
    "ani1xnr",
    "maceoff23s",
    "maceoff23m",
    "maceoff23l",
    "egret",
    "aimnet2",
    "aimnet2nse",
    "uma",
    "maceomol",
    "macepols",
    "macepolm",
    "macepoll",
    "macepolefs",
]

MODEL_NAME_TO_FILE = {
    "ani2x": "ani2x.pt",
    "ani1x": "ani1x.pt",
    "ani1ccx": "ani1ccx.pt",
    "ani1xnr": "ani1xnr.pt",
    "aimnet2": "aimnet2.pt",
    "aimnet2nse": "aimnet2nse.pt",
    "maceoff23m": "maceoff23m.pt",
    "maceomol": "maceomol.pt",
    "egret": "egret1s.pt",
    "uma-s-1p1": "uma-s-1p1.pt",
    "uma-s-1p2": "uma-s-1p2.pt",
    "uma-m-1p1": "uma-m-1p1.pt",
    "macepols": "macepols.pt",
    "macepolm": "macepolm.pt",
    "macepoll": "macepoll.pt",
    "macepolefs": "macepol-ef-s.pt",
}

HF_REPO_ID = "Wayne7815/MAPLE_model"

MODEL_HESSIAN_SUPPORT = {
    "ani2x": ("analytic", "numerical"),
    "ani1x": ("analytic", "numerical"),
    "ani1ccx": ("analytic", "numerical"),
    "ani1xnr": ("analytic", "numerical"),
    "maceoff23s": ("analytic", "numerical"),
    "maceoff23m": ("analytic", "numerical"),
    "maceoff23l": ("analytic", "numerical"),
    "egret": ("analytic", "numerical"),
    "aimnet2": ("analytic", "numerical"),
    "aimnet2nse": ("analytic", "numerical"),
    "uma": ("numerical",),
    "maceomol": ("analytic", "numerical"),
    "macepols": ("analytic", "numerical"),
    "macepolm": ("analytic", "numerical"),
    "macepoll": ("analytic", "numerical"),
    "macepolefs": ("analytic", "numerical"),
}

UNSUPPORTED_CHARGE_MULT_MODELS = {
    "ani2x",
    "ani1x",
    "ani1ccx",
    "ani1xnr",
    "maceoff23s",
    "maceoff23m",
    "maceoff23l",
    "egret",
    "maceomol",
}


class SetClaculator:
    def __init__(
        self,
        device: torch.device,
        model: str,
        output: str,
        atoms: Optional[Atoms] = None,
        d4: bool = False,
        implicit: str = "None",
        solvent: str = "None",
        model_options: Optional[dict] = None,
    ) -> None:
        self.output = output
        self.model = model
        self.d4 = d4
        self.device = device
        self.atoms = atoms
        self.implicit = implicit
        self.solvent = solvent
        self.model_options = model_options or {}

    def _validate_requested_hessian_mode(self) -> None:
        mode = self.model_options.get("hessian")
        if mode is None:
            return

        mode = str(mode).lower()
        declared = MODEL_HESSIAN_SUPPORT.get(self.model)
        if declared is not None and mode not in declared:
            supported_text = ", ".join(sorted(declared))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f"Supported modes: {supported_text}"
            )

    def _apply_hessian_mode(self, calculator) -> None:
        mode = self.model_options.get("hessian")
        if mode is None:
            return

        mode = str(mode).lower()
        supported = getattr(calculator, "supported_hessian_modes", None)
        if supported is not None and mode not in supported:
            supported_text = ", ".join(sorted(supported))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f"Supported modes: {supported_text}"
            )

        if not hasattr(calculator, "hessian"):
            raise ValueError(f"Model '{self.model}' does not expose configurable Hessian modes.")

        calculator.hessian = mode

    def _ensure_model_file(self, model_name: str) -> Optional[Path]:
        filename = MODEL_NAME_TO_FILE.get(model_name)
        if filename is None:
            return None

        model_dir = Path(__file__).parent / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        if model_path.exists():
            return model_path

        url = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{filename}"
        self.log_info(
            [
                f" [INFO] Model file '{filename}' not found locally.\n",
                f" [INFO] Downloading from: {url}\n",
            ]
        )

        temp_path = model_path.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(url) as response:
                size = response.headers.get("Content-Length")
                if size:
                    self.log_info([f" [INFO] File size: {int(size) / 1024 / 1024:.1f} MB\n"])
                with open(temp_path, "wb") as handle:
                    shutil.copyfileobj(response, handle)
            temp_path.rename(model_path)
            self.log_info([f" [INFO] Download complete: {model_path}\n"])
        except urllib.error.HTTPError as exc:
            temp_path.unlink(missing_ok=True)
            self.log_error(f" [ERROR] Download failed (HTTP {exc.code}): {url}\n")
            raise RuntimeError(f"Failed to download model '{model_name}': HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            temp_path.unlink(missing_ok=True)
            self.log_error(f" [ERROR] Network error: {exc.reason}\n")
            raise RuntimeError(f"Failed to download model '{model_name}': {exc.reason}") from exc

        return model_path

    def _warn_charge_mult(self) -> None:
        if self.atoms is None:
            return

        has_charge = self.atoms.info.get("charge", 0) != 0
        has_mult = self.atoms.info.get("mult", 1) != 1
        if (has_charge or has_mult) and self.model in UNSUPPORTED_CHARGE_MULT_MODELS:
            self.log_info(
                [
                    f"\n [WARNING] Model '{self.model}' does not support charge/multiplicity.\n",
                    f"           charge={self.atoms.info.get('charge', 0)}, ",
                    f"mult={self.atoms.info.get('mult', 1)} will be IGNORED.\n",
                    "           Models with charge/mult support: aimnet2, aimnet2nse, uma, macepols/m/l/efs\n",
                ]
            )

    def _build_calculator(self) -> ase.calculators.calculator.Calculator:
        model = self.model

        if model in {"ani2x", "ani1x", "ani1ccx", "ani1xnr"}:
            self._ensure_model_file(model)
            calculator = ANICalculator(
                model=model,
                d4=self.d4,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in {"maceoff23s", "maceoff23m", "maceoff23l", "egret"}:
            self._ensure_model_file(model)
            calculator = MACECalculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in {"aimnet2", "aimnet2nse"}:
            self._ensure_model_file(model)
            from .aimnet._aimnet2_calculator import AIMNet2Calculator

            calculator = AIMNet2Calculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model == "uma":
            from .uma._uma_calculator import (
                UMACalculator,
                UMA_DEFAULT_SIZE,
                UMA_FALLBACK_HF_MODELS,
            )

            uma_task = self.model_options.get("task")
            uma_size = self.model_options.get("size")
            checkpoint_path = None
            # Default to uma-s-1p1 when no size specified, so local model file is found
            effective_size = uma_size if uma_size else "uma-s-1p1"
            if effective_size in {"uma-s-1p1", "uma-s-1p2", "uma-m-1p1"}:
                checkpoint = self._ensure_model_file(effective_size)
                checkpoint_path = str(checkpoint) if checkpoint is not None else None

            calculator = UMACalculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
                task=uma_task,
                size=uma_size,
                checkpoint_path=checkpoint_path,
            )
        elif model == "maceomol":
            self._ensure_model_file(model)
            from .mace._mace_general_calculator import MACEModelCalculator

            calculator = MACEModelCalculator(
                model=model,
                device=self.device,
                implicit=self.implicit,
                solvent=self.solvent,
            )
        elif model in {"macepols", "macepolm", "macepoll", "macepolefs"}:
            from .mace._macepol_calculator import MACEPolCalculator

            model_path = self.model_options.get("model_path")
            if model_path is None:
                downloaded = self._ensure_model_file(model)
                model_path = str(downloaded) if downloaded is not None else None

            external_field = self.model_options.get("external_field")

            calculator = MACEPolCalculator(
                model=model,
                device=self.device,
                model_path=model_path,
                implicit=self.implicit,
                solvent=self.solvent,
                external_field=external_field,
            )
        else:
            raise ValueError(f"Model '{model}' is not implemented yet.")

        self._apply_hessian_mode(calculator)
        return calculator

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        if self.model not in IMPLEMENTATION_MODELS:
            self.log_error(f"\n [ERROR] Unsupported model: {self.model}\n")
            raise ValueError(f"Unsupported model: '{self.model}'.")

        self._validate_requested_hessian_mode()

        if self.d4 and self.model not in {"ani2x", "ani1x", "ani1ccx", "ani1xnr"}:
            self.log_info([f"\n [WARNING] D4 is not supported for model '{self.model}'. D4 will be ignored.\n"])

        calculator = self._build_calculator()
        self._warn_charge_mult()
        return calculator

    def log_error(self, error_message: str) -> None:
        with open(self.output, "a") as handle:
            handle.write(f"ERROR: {error_message}\n")

    def log_info(self, info_message: list) -> None:
        with open(self.output, "a") as handle:
            for line in info_message:
                handle.write(line)

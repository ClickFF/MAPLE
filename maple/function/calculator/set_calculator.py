from __future__ import annotations

import importlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import ase
from ase import Atoms

from .calculator_base import (
    atoms_has_pbc,
    get_registered_calculator,
    import_calculator_plugin,
    load_calculator_plugins_from_env,
    normalize_none_option,
    validate_implicit_solvent_choice,
)


HF_REPO_ID = 'Wayne7815/MAPLE_models'
HF_MODEL_REVISION = 'd5d8eb902246fb3f0e3b24e0d4bbcdc12f12e816'
_COMMON_MODEL_OPTION_KEYS = {'module', 'hessian'}

# Historical public names map to the current artifact and metadata identities.
MODEL_ALIASES = {
    'aimnet2nse': 'aimnet2-nse',
    'maceoff23s': 'mace-off-23s',
    'maceoff23m': 'mace-off-23m',
    'maceoff23l': 'mace-off-23l',
    'egret': 'egret-1s',
    'maceomol': 'mace-omol-0',
    'macepols': 'mace-polar-1s',
    'macepolm': 'mace-polar-1m',
    'macepoll': 'mace-polar-1l',
    'fennix-bio1s': 'fennix-bio-1s',
    'fennix-bio1m': 'fennix-bio-1m',
    'so3lr-t-s': 'so3lr-tauri-s',
    'so3lr-t-m': 'so3lr-tauri-m',
    'so3lr-t-l': 'so3lr-tauri-l',
}


# Static seed map: builtin name → module that registers the calculator class.
# External users add backends via input-header `module=` or the
# MAPLE_CALCULATOR_PLUGINS env var; this dict is for shipped backends only.
_BUILTIN_NAME_TO_MODULE = {
    'ani1x': 'maple.function.calculator.ani._ani_calculator',
    'ani1ccx': 'maple.function.calculator.ani._ani_calculator',
    'ani2x': 'maple.function.calculator.ani._ani_calculator',
    'ani1xnr': 'maple.function.calculator.ani._ani_calculator',
    'ani2xr': 'maple.function.calculator.ani._ani_calculator',
    'ani2dr': 'maple.function.calculator.ani._ani_calculator',
    'snn-ani2xr': 'maple.function.calculator.ani._ani_calculator',
    'ani-mbis': 'maple.function.calculator.ani._ani_calculator',
    'aimnet2': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'aimnet2-nse': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'aimnet2-rxn': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'aimnet2-pd': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'aimnet2-b973c': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'aimnet2-2025': 'maple.function.calculator.aimnet._aimnet2_calculator',
    'mace-off-23s': 'maple.function.calculator.mace._mace_calculator',
    'mace-off-23m': 'maple.function.calculator.mace._mace_calculator',
    'mace-off-23l': 'maple.function.calculator.mace._mace_calculator',
    'mace-off-23bm': 'maple.function.calculator.mace._mace_calculator',
    'mace-off-24m': 'maple.function.calculator.mace._mace_calculator',
    'mace-les-off': 'maple.function.calculator.mace._mace_calculator',
    'egret-1': 'maple.function.calculator.mace._mace_calculator',
    'egret-1e': 'maple.function.calculator.mace._mace_calculator',
    'egret-1m': 'maple.function.calculator.mace._mace_calculator',
    'egret-1s': 'maple.function.calculator.mace._mace_calculator',
    'egret-1t': 'maple.function.calculator.mace._mace_calculator',
    'mace-omol-0': 'maple.function.calculator.mace._mace_general_calculator',
    'mace-polar-1s': 'maple.function.calculator.mace._macepol_calculator',
    'mace-polar-1m': 'maple.function.calculator.mace._macepol_calculator',
    'mace-polar-1l': 'maple.function.calculator.mace._macepol_calculator',
    'fennix-bio-1s': 'maple.function.calculator.fennol._fennol_calculator',
    'fennix-bio-1m': 'maple.function.calculator.fennol._fennol_calculator',
    'grace-off-1s': 'maple.function.calculator.grace._grace_calculator',
    'grace-off-1m': 'maple.function.calculator.grace._grace_calculator',
    'grace-off-1l': 'maple.function.calculator.grace._grace_calculator',
    'grace-off-2s': 'maple.function.calculator.grace._grace_calculator',
    'grace-off-2m': 'maple.function.calculator.grace._grace_calculator',
    'grace-off-2l': 'maple.function.calculator.grace._grace_calculator',
    'so3lr': 'maple.function.calculator.so3lr._so3lr_calculator',
    'so3lr-tauri-s': 'maple.function.calculator.so3lr._so3lr_calculator',
    'so3lr-tauri-m': 'maple.function.calculator.so3lr._so3lr_calculator',
    'so3lr-tauri-l': 'maple.function.calculator.so3lr._so3lr_calculator',
    'allegro-fm-256': 'maple.function.calculator.nequip._nequip_calculator',
    'allegro-fm-512': 'maple.function.calculator.nequip._nequip_calculator',
    'nequip-t3-cosmo': 'maple.function.calculator.nequip._nequip_calculator',
    'nequip-t123-cosmo': 'maple.function.calculator.nequip._nequip_calculator',
    'nequip-t3-gas': 'maple.function.calculator.nequip._nequip_calculator',
    'uma': 'maple.function.calculator.uma._uma_calculator',
}

_plugins_loaded_from_env = False


def _compact_model_name(name: str) -> str:
    return str(name).strip().lower().translate(str.maketrans('', '', '-_ ()'))


_BUILTIN_ALIAS_TO_NAME = {
    _compact_model_name(name): name for name in _BUILTIN_NAME_TO_MODULE
}
_BUILTIN_ALIAS_TO_NAME.update({
    _compact_model_name(alias): name for alias, name in MODEL_ALIASES.items()
})


def _builtin_canonical_name(name: str) -> Optional[str]:
    return _BUILTIN_ALIAS_TO_NAME.get(_compact_model_name(name))


def _model_download_url(filename: str) -> str:
    import os

    revision = os.environ.get('MAPLE_MODEL_REVISION', HF_MODEL_REVISION).strip() or HF_MODEL_REVISION
    return f'https://huggingface.co/{HF_REPO_ID}/resolve/{revision}/{filename}'


def _normalize_model_options(model_options: Optional[dict]) -> dict:
    """Normalize option keys and enum-like values without touching path values."""
    options = {}
    for raw_key, raw_value in (model_options or {}).items():
        key = str(raw_key).strip().lower()
        if raw_value is None:
            value = None
        elif key in {'hessian', 'lr', 'dtype', 'inference', 'task', 'size'}:
            value = str(raw_value).strip().lower()
        elif key in {'module', 'model_path', 'checkpoint_path'}:
            value = str(raw_value).strip()
        elif isinstance(raw_value, str):
            value = raw_value.strip()
        else:
            value = raw_value
        options[key] = value
    return options


class SetCalculator:
    def __init__(
        self,
        device,
        model: str,
        output: str,
        atoms: Optional[Atoms] = None,
        d4: bool = False,
        implicit: str = 'None',
        solvent: str = 'None',
        model_options: Optional[dict] = None,
        solvation_options: Optional[dict] = None,
    ) -> None:
        self.output = output
        self.model = str(model).strip().lower()
        self.d4 = d4
        self.device = device
        self.atoms = atoms
        self.implicit = normalize_none_option(implicit)
        self.solvent = normalize_none_option(solvent)
        self.model_options = _normalize_model_options(model_options)
        self.solvation_options = solvation_options or {}
        self._model_error_logged = False

    def set_calculator(self) -> ase.calculators.calculator.Calculator:
        try:
            calculator = self._build_calculator()
            return calculator
        except Exception as exc:
            if not self._model_error_logged:
                self._log_model_error(
                    f"Failed to initialize model '{self.model}'.\n"
                    f'       MAPLE model directory: {self._model_dir_description()}\n'
                    f'       Error: {type(exc).__name__}: {exc}'
                )
            raise

    def _build_calculator(self) -> ase.calculators.calculator.Calculator:
        requested_name = self.model
        self._validate_solvent_config()

        cls = self._discover_calculator_class(requested_name)
        name = self.model
        self._validate_model_options(cls)
        self._validate_against_class(cls)
        options = dict(self.model_options)
        options.setdefault('d4', self.d4)
        resolved_model_path = self._resolve_model_path(cls, name)

        # UMA's `size`/`checkpoint_path` resolution is special: when no checkpoint
        # is given and the requested size has a known HF fallback, prefer a
        # locally-cached MAPLE copy if present. Handle the coercion here so that
        # UMACalculator only receives clean kwargs.
        if cls.__name__ == 'UMACalculator':
            from .uma._uma_calculator import UMACalculator, UMA_DEFAULT_SIZE, UMA_FALLBACK_HF_MODELS

            inference = options.get('inference')
            effective_device = UMACalculator._normalize_device(self.device)
            options['inference'] = self._coerce_uma_inference_for_device(inference, effective_device)

            checkpoint_path = options.get('checkpoint_path')
            if checkpoint_path:
                checkpoint_path = Path(str(checkpoint_path)).expanduser()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Explicit checkpoint_path for '{self.model}' does not exist or is not a file: "
                        f"{checkpoint_path}"
                    )
                checkpoint_path = str(checkpoint_path)
            elif resolved_model_path is not None:
                checkpoint_path = str(resolved_model_path)
            effective_size = str(options.get('size')).lower() if options.get('size') else UMA_DEFAULT_SIZE
            if checkpoint_path is None and effective_size in UMA_FALLBACK_HF_MODELS:
                local_checkpoint = self._local_model_file(f'{effective_size}.pt')
                if local_checkpoint is not None:
                    checkpoint_path = str(local_checkpoint)
            options['checkpoint_path'] = checkpoint_path

        resolved_path_str = str(resolved_model_path) if resolved_model_path is not None else None
        kwargs = cls.build_kwargs_from_options(name, options, resolved_model_path=resolved_path_str)

        calculator = cls(
            device=self.device,
            model=name,
            implicit=self.implicit,
            solvent=self.solvent,
            **kwargs,
        )

        self._apply_hessian_mode(calculator)
        return calculator

    def _discover_calculator_class(self, name: str):
        """Resolve `name` to a registered calculator class.

        Plug-in discovery layers:
        1. Explicit `module=` in model_options (run import_calculator_plugin).
        2. _BUILTIN_NAME_TO_MODULE seed for shipped backends.
        3. MAPLE_CALCULATOR_PLUGINS env var (loaded once per process).
        """
        global _plugins_loaded_from_env
        if not _plugins_loaded_from_env:
            load_calculator_plugins_from_env()
            _plugins_loaded_from_env = True

        normalized = name.lower()
        module_override = self.model_options.get('module')
        if module_override:
            import_calculator_plugin(str(module_override))
        else:
            builtin_name = _builtin_canonical_name(normalized)
            builtin_module = _BUILTIN_NAME_TO_MODULE.get(builtin_name)
            if builtin_module is not None:
                importlib.import_module(builtin_module)

        try:
            cls = get_registered_calculator(normalized)
            self.model = normalized
            return cls
        except KeyError:
            pass

        builtin_name = _builtin_canonical_name(normalized)
        if builtin_name is not None and builtin_name != normalized:
            importlib.import_module(_BUILTIN_NAME_TO_MODULE[builtin_name])
            cls = get_registered_calculator(builtin_name)
            self.model = builtin_name
            return cls

        raise ValueError(f"Unsupported model: '{name}'.")

    def _validate_solvent_config(self) -> None:
        self.implicit, self.solvent = validate_implicit_solvent_choice(
            self.implicit, self.solvent
        )
        if self.implicit == 'none':
            return

        if self.implicit != 'gbsa':
            raise ValueError(
                "Unsupported implicit solvation method: "
                f"'{self.implicit}'. Supported experimental method: gbsa."
            )

        if self.solvation_options.get('experimental') is not True:
            raise ValueError(
                "Implicit GB-polar/QEq solvation is experimental and disabled "
                "by default. Add experimental=true in #solv(...) to request "
                "energy-only use."
            )

        if self.model_options.get('hessian') is not None:
            raise ValueError(
                "Experimental implicit GB-polar solvation does not support "
                "Hessian/HVP workflows."
            )

        if self.atoms is not None and atoms_has_pbc(self.atoms):
            raise ValueError(
                "Experimental implicit GB-polar solvation is non-periodic only; "
                "remove #pbc or use a periodic solvent backend."
            )

    def _validate_model_options(self, cls) -> None:
        """Fail loudly on misspelled/unsupported model_options for shipped backends."""
        if 'model_path' in self.model_options:
            raise ValueError('model_path has been removed; select a registered model name. '
                             'UMA retains its checkpoint_path option.')
        option_keys = getattr(cls, 'OPTION_KEYS', None)
        if option_keys is None:
            return
        allowed = set(_COMMON_MODEL_OPTION_KEYS)
        allowed.update(option_keys)
        unknown = sorted(key for key in self.model_options if key not in allowed)
        if unknown:
            allowed_text = ', '.join(sorted(allowed)) or '(none)'
            raise ValueError(
                f"Unsupported model option(s) for '{self.model}': {', '.join(unknown)}. "
                f"Supported options: {allowed_text}"
            )

    def _validate_against_class(self, cls) -> None:
        """Pre-instantiation gates: pbc, hessian mode, charge/mult, d4."""
        if self.atoms is not None and atoms_has_pbc(self.atoms) and not getattr(cls, 'SUPPORTS_PBC', False):
            raise NotImplementedError(
                f"Model '{self.model}' is a no-PBC molecular wrapper. "
                "Use UMA or a backend-native PBC calculator for periodic systems."
            )

        mode = self.model_options.get('hessian')
        if mode is not None:
            if mode not in cls.SUPPORTED_HESSIAN_MODES:
                supported_text = ', '.join(sorted(cls.SUPPORTED_HESSIAN_MODES))
                raise ValueError(
                    f"Model '{self.model}' does not support hessian='{mode}'. "
                    f'Supported modes: {supported_text}'
                )

        if self.atoms is not None:
            has_charge = self.atoms.info.get('charge', 0) != 0
            has_mult = self.atoms.info.get('mult', 1) != 1
            if (has_charge or has_mult) and not cls.SUPPORTS_CHARGE_MULT:
                raise ValueError(f"Model '{self.model}' has no charge/multiplicity conditioning; "
                                 "use charge=0 and mult=1.")

        if self.d4:
            import inspect

            ctor_params = inspect.signature(cls.__init__).parameters
            if 'd4' not in ctor_params:
                self.log_info(
                    [f"\n [WARNING] D4 is not supported for model '{self.model}'. D4 will be ignored.\n"]
                )

    def _apply_hessian_mode(self, calculator) -> None:
        mode = self.model_options.get('hessian')
        if mode is None:
            return

        supported = type(calculator).SUPPORTED_HESSIAN_MODES
        if mode not in supported:
            supported_text = ', '.join(sorted(supported))
            raise ValueError(
                f"Model '{self.model}' does not support hessian='{mode}'. "
                f'Supported modes: {supported_text}'
            )

        if not hasattr(calculator, 'hessian'):
            raise ValueError(f"Model '{self.model}' does not expose configurable Hessian modes.")

        calculator.hessian = mode

    def _coerce_uma_inference_for_device(self, inference, device_name: str):
        if inference == 'turbo' and device_name == 'cpu':
            self.log_info(
                [
                    "\n [WARNING] UMA inference='turbo' requires CUDA; "
                    "falling back to 'default' on CPU.\n"
                ]
            )
            return 'default'
        return inference

    def _resolve_model_path(self, cls, name: str) -> Optional[Path]:
        """Resolve checkpoint path per class attrs.

        - If `cls.CHECKPOINT_FILENAME` maps the name → ensure (download or local).
        - Else if `cls.REQUIRES_LOCAL_MODEL_FILE` is True → require local.
        - Else → no path (backend looks up its own default).
        """
        local_filenames = getattr(cls, 'LOCAL_MODEL_FILENAMES', None)
        if local_filenames and name in local_filenames:
            return self._require_local_model_file(name, local_filenames[name])
        if cls.CHECKPOINT_FILENAME and name in cls.CHECKPOINT_FILENAME:
            filename = cls.CHECKPOINT_FILENAME[name]
            return self._ensure_model_file(filename, name)
        if cls.REQUIRES_LOCAL_MODEL_FILE:
            return self._require_local_model_file(name)
        return None

    def _ensure_model_file(self, filename: str, model_name: str) -> Path:
        """Ensure a HF-hosted checkpoint is on disk; download if missing.

        Production safeguards: a pinned HuggingFace revision (override with
        MAPLE_MODEL_REVISION only when intentionally refreshing model assets),
        an offline switch (MAPLE_OFFLINE), a socket timeout
        (MAPLE_DOWNLOAD_TIMEOUT, default 60s), a process-unique temp file so
        concurrent runs never clobber each other's partial download, and a
        byte-count check against Content-Length before the atomic replace so a
        truncated stream cannot land as the final model file.
        Checksum pinning is intentionally omitted: it needs an upstream
        hash manifest, which this repo does not publish — do not fake one.
        """
        import os

        model_dir = self._model_dir()
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / filename
        if model_path.exists():
            return model_path

        if os.environ.get('MAPLE_OFFLINE', '').strip().lower() in ('1', 'true', 'yes', 'on'):
            message = (
                f"Model file '{filename}' for '{model_name}' is missing and MAPLE_OFFLINE "
                'is set, so auto-download is disabled.\n'
                f'       MAPLE model directory searched: {self._model_dir_description()}'
            )
            self._log_model_error(message)
            raise FileNotFoundError(message)

        url = _model_download_url(filename)
        self.log_info(
            [
                f" [INFO] Model file '{filename}' not found locally.\n",
                f' [INFO] Downloading from: {url}\n',
            ]
        )

        timeout = float(os.environ.get('MAPLE_DOWNLOAD_TIMEOUT', '60'))
        temp_path = model_path.with_suffix(f'.{os.getpid()}.tmp')
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                size = response.headers.get('Content-Length')
                expected = int(size) if size else None
                if expected is not None:
                    self.log_info([f' [INFO] File size: {expected / 1024 / 1024:.1f} MB\n'])
                with open(temp_path, 'wb') as handle:
                    shutil.copyfileobj(response, handle)
            downloaded = temp_path.stat().st_size
            if expected is not None and downloaded != expected:
                temp_path.unlink(missing_ok=True)
                message = (
                    f"Incomplete download for model '{model_name}': received {downloaded} bytes, "
                    f'expected {expected} (Content-Length).'
                )
                self._log_model_error(message)
                raise RuntimeError(message)
            temp_path.replace(model_path)
            self.log_info([f' [INFO] Download complete: {model_path}\n'])
        except urllib.error.HTTPError as exc:
            temp_path.unlink(missing_ok=True)
            self._log_model_error(
                f"Download failed for model '{model_name}' (HTTP {exc.code}): {url}\n"
                f'       MAPLE model directory: {self._model_dir_description()}'
            )
            raise RuntimeError(f"Failed to download model '{model_name}': HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            temp_path.unlink(missing_ok=True)
            reason = getattr(exc, 'reason', exc)
            self._log_model_error(
                f"Network error while downloading model '{model_name}': {reason}\n"
                f'       MAPLE model directory: {self._model_dir_description()}'
            )
            raise RuntimeError(f"Failed to download model '{model_name}': {reason}") from exc

        return model_path

    def _require_local_model_file(self, model_name: str, filename: Optional[str] = None) -> Path:
        filename = filename or f'{model_name}.pt'
        model_path = self._model_dir() / filename
        if model_path.exists():
            return model_path

        message = (
            f"Model file '{filename}' for '{model_name}' was not found locally.\n"
            f'       MAPLE model directory searched: {self._model_dir_description()}\n'
            f'       Expected file path: {model_path}\n'
            f'This model is not available from {HF_REPO_ID}; install the backend-specific model '
            'file under its registered filename.'
        )
        self._log_model_error(message)
        raise FileNotFoundError(message)

    def _local_model_file(self, filename: str) -> Optional[Path]:
        model_path = self._model_dir() / filename
        return model_path if model_path.exists() else None

    def _model_dir(self) -> Path:
        return Path(__file__).parent / 'model'

    def _model_dir_description(self) -> str:
        model_dir = self._model_dir()
        resolved_dir = model_dir.resolve()
        if resolved_dir != model_dir:
            return f'{model_dir} (resolved: {resolved_dir})'
        return str(model_dir)

    def _log_model_error(self, message: str) -> None:
        self._model_error_logged = True
        self.log_error(message)

    def log_error(self, error_message: str) -> None:
        with open(self.output, 'a') as handle:
            handle.write(f'ERROR: {error_message.rstrip()}\n')

    def log_info(self, info_message: list) -> None:
        with open(self.output, 'a') as handle:
            for line in info_message:
                handle.write(line)
# Historical misspelling kept as an alias for one release.
SetClaculator = SetCalculator

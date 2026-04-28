import re
from typing import Any, Dict, List, Optional


class CommandControl:
    """
    Parse and validate input settings.
    One task only: sp/opt/ts/scan/freq/irc/md.
    All other settings are global parameters.
    """

    SUPPORTED_MODELS = {
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
    }

    SUPPORTED_TASKS = {"sp", "opt", "ts", "scan", "freq", "irc", "md"}

    SUPPORTED_UMA_TASKS = {"omol", "omat", "oc20", "odac", "omc", "oc22", "oc25"}
    SUPPORTED_UMA_SIZES = {"uma-s-1p1", "uma-s-1p2", "uma-m-1p1"}
    SUPPORTED_HESSIAN_MODES = {"analytic", "numerical"}

    DEFAULTS = {
        "model": None,
        "device": None,
        "d4": False,
        "sp": {},
        "opt": {},
        "ts": {},
        "scan": {},
        "freq": {
            "method": "mw",
            "temperature": 298.15,
            "pressure_kpa": 101.325,
            "ilowfreq": 2,
            "verbosity": 1,
            "treat_imag_as_real": False,
            "device": "cpu",
        },
        "md": {
            "ensemble": "nve",
            "timestep": 0.25,
            "steps": 400000,
            "temperature": 300.0,
            "traj_every": 100,
            "log_every": 100,
            "init_velocities": True,
            "restart": False,
            "load_state": False,
            "rst_file": "",
            "rst_every": 1000,
            "remove_com": True,
            "remove_com_every": 100,
            "remove_rotation": False,
            "remove_angular": False,
            "remove_angular_every": 0,
            "random_seed": None,
            "thermostat": "langevin",
            "friction": 0.001,
            "tau_t": 100.0,
            "barostat": "c-rescale",
            "pressure": 1.0,
            "tau_p": 2000.0,
            "compressibility": 4.5e-5,
            "mdp": None,
            "traj_format": "xyz",
            "debug": False,
        },
        "solv": {"solvent": "water", "explicit": None},
    }

    IMPLEMENTATION_MAP = {
        "opt": {"lbfgs", "rfo", "sd", "cg", "sdcg", ""},
        "scan": {"lbfgs", "cg"},
        "ts": {"prfo", "string", "neb", "dimer", "autoneb"},
        "freq": {"mw", "nonmw", "both"},
        "sp": set(),
        "irc": {"gs", "hpc", "eulerpc", "lqa"},
        "md": {"nve", "nvt", "npt"},
    }

    def __init__(self, params: Dict[str, Any], task: str, output_path: Optional[str] = None):
        self.params = params
        self.task = task
        self.output_path = output_path

    @classmethod
    def from_settings(
        cls,
        settings_lines: List[str],
        output_path: Optional[str] = None,
    ) -> "CommandControl":
        params: Dict[str, Any] = {}
        task: Optional[str] = None
        seen_keys = set()
        log_lines = ["Parsing # commands...\n"]

        for raw in settings_lines:
            line = raw.strip()
            if not line.startswith("#"):
                continue

            match = re.match(
                r"#\s*([A-Za-z0-9_]+)\s*(?:=\s*([^()\s]+))?\s*(?:\((.*)\))?",
                line,
            )
            if not match:
                continue

            key = match.group(1).strip().lower()
            assign_val = match.group(2)
            paren_val = match.group(3)

            if key in cls.SUPPORTED_TASKS:
                if task and task != key:
                    cls._log_error(output_path, f"Multiple tasks defined: '{task}' and '{key}'.")
                    raise ValueError(f"Multiple tasks defined: '{task}' and '{key}'.")

                task = key
                params.update(cls.DEFAULTS.get(key, {}))
                log_lines.append(f"Task set to '{task}'\n")

                inline_md_keys = set()
                if paren_val:
                    cls._parse_nested(params, paren_val)
                    if task == "md":
                        inline_md_keys = {
                            kv.split("=", 1)[0].strip().lower()
                            for kv in paren_val.split(",")
                            if "=" in kv
                        }

                if task == "md" and params.get("mdp"):
                    cls._load_mdp(params, inline_md_keys, output_path)

                continue

            if key in seen_keys:
                cls._log_error(output_path, f"Duplicate parameter: '{key}'.")
                raise ValueError(f"Duplicate parameter: '{key}'.")
            seen_keys.add(key)

            if paren_val is not None and assign_val is not None:
                sub = {}
                cls._parse_nested(sub, paren_val)
                params[key] = cls._auto_cast(assign_val.strip())
                params[f"{key}_options"] = sub
                log_lines.append(f"Global parameter: {key} = {params[key]} with options {sub}\n")
                continue

            if paren_val:
                if key == "pbc":
                    params["pbc"] = cls._parse_pbc(paren_val, output_path)
                    log_lines.append(f"Global parameter: pbc = {params['pbc']}\n")
                    continue

                sub = {}
                cls._parse_nested(sub, paren_val)
                params[key] = sub
                log_lines.append(f"Global nested parameter: {key} = {sub}\n")
                continue

            if assign_val:
                value = cls._auto_cast(assign_val.strip())
                params[key] = value
                log_lines.append(f"Global parameter: {key} = {value}\n")
                continue

            params[key] = True
            log_lines.append(f"Global flag: {key} = True\n")

        if not task:
            task = "sp"
            params.update(cls.DEFAULTS.get("sp", {}))
            log_lines.append("No task specified. Defaulting to 'sp'.\n")

        cls._normalize_params(params)
        cls._validate(params, task, output_path)
        cls._log_info(output_path, log_lines)

        return cls(params, task, output_path)

    @staticmethod
    def _parse_nested(target: Dict[str, Any], inner: str) -> None:
        for kv in inner.split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                target[k.strip().lower()] = CommandControl._auto_cast(v.strip())
            else:
                target[kv.strip().lower()] = True

    @classmethod
    def _parse_pbc(cls, inner: str, output_path: Optional[str]) -> List[float]:
        try:
            values = [float(x.strip()) for x in inner.lstrip("=").strip().split(",")]
        except ValueError as exc:
            cls._log_error(output_path, f"Invalid PBC values: {inner} - {exc}")
            raise ValueError(f"Invalid PBC values: {inner}") from exc

        if len(values) == 2:
            a, b = values
            cellpar = [a, b, 1000.0, 90.0, 90.0, 90.0]
        elif len(values) == 3:
            a, b, c = values
            cellpar = [a, b, c, 90.0, 90.0, 90.0]
        elif len(values) == 6:
            cellpar = values
        else:
            cls._log_error(output_path, f"PBC requires 2, 3, or 6 values, got {len(values)}.")
            raise ValueError(
                f"PBC requires 2, 3, or 6 values (a,b[,c][,alpha,beta,gamma]), got {len(values)}."
            )

        return cellpar

    @staticmethod
    def _auto_cast(value: str) -> Any:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return int(value)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            pass
        return value

    @classmethod
    def _load_mdp(
        cls,
        params: Dict[str, Any],
        inline_keys: set[str],
        output_path: Optional[str] = None,
    ) -> None:
        from ..dispatcher.md.mdp_reader import parse_mdp

        mdp_path = params["mdp"]
        try:
            mdp_params = parse_mdp(mdp_path)
        except FileNotFoundError:
            cls._log_error(output_path, f"MDP file not found: {mdp_path!r}")
            raise
        except ValueError as exc:
            cls._log_error(output_path, str(exc))
            raise

        defaults = cls.DEFAULTS.get("md", {})
        for key, mdp_val in mdp_params.items():
            if key in defaults and key not in inline_keys:
                params[key] = mdp_val

        if "remove_rotation" in mdp_params and "remove_angular" not in mdp_params:
            params["remove_angular"] = params["remove_rotation"]

    @classmethod
    def _normalize_params(cls, params: Dict[str, Any]) -> None:
        if "remove_angular" not in params and "remove_rotation" in params:
            params["remove_angular"] = params["remove_rotation"]
        if params.get("remove_angular"):
            params["remove_com"] = True

        if "model" in params and params["model"] is not None:
            params["model"] = (
                str(params["model"])
                .lower()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
                .replace("(", "")
                .replace(")", "")
            )

        model_options = params.get("model_options")
        if isinstance(model_options, dict):
            for key in ("task", "size", "hessian"):
                if key in model_options and isinstance(model_options[key], str):
                    model_options[key] = model_options[key].lower()

        if "ensemble" in params and isinstance(params["ensemble"], str):
            params["ensemble"] = params["ensemble"].lower()

    @classmethod
    def _validate(cls, params: Dict[str, Any], task: str, output_path: Optional[str]) -> None:
        model = params.get("model")
        if model is not None and model not in cls.SUPPORTED_MODELS:
            cls._log_error(output_path, f"Unsupported model: {model}")
            raise ValueError(f"Unsupported model: '{model}'.")

        if "gpuid" in params and params["gpuid"] is not None and not isinstance(params["gpuid"], int):
            cls._log_error(output_path, "GPU ID must be an integer.")
            raise ValueError("GPU ID must be an integer.")

        if "d4" in params and not isinstance(params["d4"], bool):
            cls._log_error(output_path, "D4 must be 'true' or 'false'.")
            raise ValueError("D4 must be 'true' or 'false'.")

        if "method" in params:
            if task == "md":
                cls._log_error(output_path, "'method' is not valid for MD tasks; use 'ensemble=' instead.")
                raise ValueError("'method' is not valid for MD tasks. Use 'ensemble=' to choose nve/nvt/npt.")

            allowed = cls.IMPLEMENTATION_MAP.get(task, set())
            if allowed and params["method"] not in allowed:
                cls._log_error(output_path, f"Method '{params['method']}' not implemented for task '{task}'.")
                raise ValueError(f"Method '{params['method']}' not implemented for task '{task}'.")

        if task == "md":
            ensemble = params.get("ensemble", "nve")
            allowed = cls.IMPLEMENTATION_MAP["md"]
            if ensemble not in allowed:
                cls._log_error(output_path, f"MD ensemble '{ensemble}' not supported.")
                raise ValueError(f"MD ensemble '{ensemble}' not supported. Choose from: {sorted(allowed)}")
        elif "ensemble" in params:
            cls._log_error(output_path, f"'ensemble' is only valid for MD tasks, not '{task}'.")
            raise ValueError(f"'ensemble' is only valid for MD tasks, not '{task}'.")

        if "pbc" in params:
            pbc_val = params["pbc"]
            if not isinstance(pbc_val, list) or len(pbc_val) != 6:
                cls._log_error(output_path, "PBC must be a list of 6 values [a, b, c, alpha, beta, gamma].")
                raise ValueError("PBC must be a list of 6 values.")
            if any(pbc_val[i] <= 0 for i in range(3)):
                cls._log_error(output_path, "PBC lattice parameters (a, b, c) must be positive.")
                raise ValueError("PBC lattice parameters must be positive.")
            if any(pbc_val[i] <= 0 or pbc_val[i] >= 180 for i in range(3, 6)):
                cls._log_error(output_path, "PBC angles (alpha, beta, gamma) must be in range (0, 180).")
                raise ValueError("PBC angles must be in range (0, 180).")

        model_options = params.get("model_options", {})
        if model == "uma":
            task_opt = model_options.get("task")
            if task_opt is not None and task_opt not in cls.SUPPORTED_UMA_TASKS:
                msg = f"Unsupported UMA task: '{task_opt}'. Supported: {sorted(cls.SUPPORTED_UMA_TASKS)}"
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            size_opt = model_options.get("size")
            if size_opt is not None and size_opt not in cls.SUPPORTED_UMA_SIZES:
                msg = f"Unsupported UMA size: '{size_opt}'. Supported: {sorted(cls.SUPPORTED_UMA_SIZES)}"
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            if "pbc" in params and task_opt == "omol":
                cls._log_error(output_path, "PBC is incompatible with UMA task='omol'.")
                raise ValueError("PBC is incompatible with UMA task='omol'.")

        hessian_mode = model_options.get("hessian")
        if hessian_mode is not None and hessian_mode not in cls.SUPPORTED_HESSIAN_MODES:
            msg = (
                f"Unsupported Hessian mode: '{hessian_mode}'. "
                f"Supported: {sorted(cls.SUPPORTED_HESSIAN_MODES)}"
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

    @staticmethod
    def _log_info(output_path: Optional[str], lines: List[str]) -> None:
        if output_path:
            with open(output_path, "a") as handle:
                for line in lines:
                    handle.write(line)

    @staticmethod
    def _log_error(output_path: Optional[str], message: str) -> None:
        if output_path:
            with open(output_path, "a") as handle:
                handle.write(f"ERROR: {message}\n")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.params.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.params)
        out["task"] = self.task
        return out

    def summary(self) -> str:
        lines = ["Parsed configuration:\n", "-" * 40 + "\n"]
        lines.append(f"Task: {self.task}\n")
        for key, value in self.params.items():
            lines.append(f"{key:<15}: {value}\n")
        return "".join(lines)

    def __repr__(self) -> str:
        return f"CommandControl(task={self.task}, params={self.params})"

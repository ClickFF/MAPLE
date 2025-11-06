import re
import os
from typing import Dict, Any, List, Optional


class CommandControl:
    """
    Parse and validate input settings.
    One task only: sp/opt/ts/scan/freq/irc.
    All other settings are global parameters.
    """

    SUPPORTED_MODELS = {
        "ani2x", "ani1x", "ani1ccx", "ani1xnr",
        "maceoff23s", "maceoff23m", "maceoff23l",
        "egret", "aimnet2", "uma"
    }

    SUPPORTED_TASKS = {"sp", "opt", "ts", "scan", "freq", "irc"}

    # Defaults assigned only when task is selected
    DEFAULTS = {
        "model": None,
        "device": None,
        "d4": False,
        "sp": {},
        "opt": {"method": "lbfgs", "maxiter": 200, "convergence": 1e-5},
        "ts": {"method": "prfo", "maxiter": 200, "neb_images": 7, "level": "medium", "refine": None},
        "scan": {"method": "lbfgs"},
        "freq": {"method": "mw", "temperature": 298.15},
        "solv": {"solvent": "water", "explicit": None},
    }

    IMPLEMENTATION_MAP = {
        "opt": {"lbfgs", "rfo", "cg", ""},
        "scan": {"lbfgs", "cg"},
        "ts": {"prfo", "string", "neb", "dimer"},
        "freq": {"mw", "nonmw", "both"},
        "sp": set(),
        "irc": {"gs"},
    }

    def __init__(self, params: Dict[str, Any], task: str, output_path: Optional[str] = None):
        self.params = params
        self.task = task
        self.output_path = output_path

    @classmethod
    def from_settings(cls, settings_lines: List[str], output_path: Optional[str] = None) -> "CommandControl":
        params: Dict[str, Any] = {}
        task: Optional[str] = None
        seen_keys = set()
        log_lines = ["Parsing # commands...\n"]

        for raw in settings_lines:
            line = raw.strip()
            if not line.startswith("#"):
                continue

            # ✅ Unified parsing: #key = value OR #key(value)
            match = re.match(r"#\s*([A-Za-z0-9_]+)\s*(?:=\s*([^()]+))?(?:\((.*)\))?", line)
            if not match:
                continue

            key = match.group(1).strip().lower()
            assign_val = match.group(2)
            paren_val = match.group(3)

            # ✅ Identify task
            if key in cls.SUPPORTED_TASKS:
                if task and task != key:
                    cls._log_error(output_path, f"Multiple tasks defined: '{task}' and '{key}'.")
                    raise ValueError(f"Multiple tasks defined: '{task}' and '{key}'.")
                task = key

                # Default params for task
                params.update(cls.DEFAULTS.get(key, {}))
                log_lines.append(f"Task set to '{task}'\n")

                # Task options inside parentheses
                if paren_val:
                    cls._parse_nested(params, paren_val)

                continue

            # ✅ Global parameters
            if key in seen_keys:
                cls._log_error(output_path, f"Duplicate parameter: '{key}'.")
                raise ValueError(f"Duplicate parameter: '{key}'.")
            seen_keys.add(key)

            # Parenthesized nested form
            if paren_val:
                sub = {}
                cls._parse_nested(sub, paren_val)
                params[key] = sub
                log_lines.append(f"Global nested parameter: {key} = {sub}\n")
                continue

            # Simple key=value assignment
            if assign_val:
                value = cls._auto_cast(assign_val.strip())
                params[key] = value
                log_lines.append(f"Global parameter: {key} = {value}\n")
                continue

            # Flag style (#d4)
            params[key] = True
            log_lines.append(f"Global flag: {key} = True\n")

        # ✅ If no task specified → default sp
        if not task:
            task = "sp"
            params.update(cls.DEFAULTS.get("sp", {}))
            log_lines.append("No task specified. Defaulting to 'sp'.\n")

        # Normalize model name
        if 'model' in params and params['model'] is not None:
            params['model'] = params['model'].lower().replace('_', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')

        # ✅ Validation
        cls._validate(params, task, output_path)
        cls._log_info(output_path, log_lines)

        return cls(params, task, output_path)

    @staticmethod
    def _parse_nested(target: Dict[str, Any], inner: str) -> None:
        for kv in inner.split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                target[k.strip()] = CommandControl._auto_cast(v.strip())
            else:
                target[kv.strip()] = True

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
    def _validate(cls, params: Dict[str, Any], task: str, output_path: Optional[str]) -> None:
        # model
        if "model" in params and params["model"] is not None and params["model"] not in cls.SUPPORTED_MODELS:
            cls._log_error(output_path, f"Unsupported model: {params['model']}")
            raise ValueError(f"Unsupported model: '{params['model']}'.")

        # device ID type
        if "gpuid" in params and params["gpuid"] is not None and not isinstance(params["gpuid"], int):
            cls._log_error(output_path, "GPU ID must be an integer.")
            raise ValueError("GPU ID must be an integer.")

        # d4 must be bool if present
        if "d4" in params and not isinstance(params["d4"], bool):
            cls._log_error(output_path, "D4 must be 'true' or 'false'.")
            raise ValueError("D4 must be 'true' or 'false'.")

        # check method compatibility
        if "method" in params:
            allowed = cls.IMPLEMENTATION_MAP.get(task, set())
            if allowed and params["method"] not in allowed:
                cls._log_error(output_path, f"Method '{params['method']}' not implemented for task '{task}'.")
                raise ValueError(f"Method '{params['method']}' not implemented for task '{task}'.")

    @staticmethod
    def _log_info(output_path: Optional[str], lines: List[str]) -> None:
        if output_path:
            with open(output_path, "a") as f:
                for line in lines:
                    f.write(line)

    @staticmethod
    def _log_error(output_path: Optional[str], message: str) -> None:
        if output_path:
            with open(output_path, "a") as f:
                f.write(f"ERROR: {message}\n")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.params.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.params)
        out["task"] = self.task
        return out

    def summary(self) -> str:
        lines = ["Parsed configuration:\n", "-" * 40 + "\n"]
        lines.append(f"Task: {self.task}\n")
        for k, v in self.params.items():
            lines.append(f"{k:<15}: {v}\n")
        return "".join(lines)

    def __repr__(self) -> str:
        return f"CommandControl(task={self.task}, params={self.params})"

# command_control.py

import re
import os
from typing import Dict, Any, List, Optional


class CommandControl:
    """
    CommandControl parses, validates, and checks implementation compatibility for all
    input settings. No 'jobtype' exists — tasks (sp, opt, ts, scan, freq) are specified
    directly. If no task is defined, 'sp' is assumed.
    """

    SUPPORTED_MODELS = {"ANI-2x", "ANI-1x", "ANI-1ccx", "ANI-1xnr"}
    SUPPORTED_TASKS = {"sp", "opt", "ts", "scan", "freq"}

    DEFAULTS = {
        "model": None,
        "device": None,
        "d4": False,
        "sp": {},
        "opt": {"method": "lbfgs", "maxiter": 200, "convergence": 1e-5},
        "ts": {"method": "prfo", "maxiter": 200, "neb_images": 7, "level": "medium", "refine": None},
        "scan": {"method": "lbfgs"},
        "freq": {"method": "mw", "temperature": 298.15},
    }

    IMPLEMENTATION_MAP = {
        "opt": {"lbfgs", "bfgs", "cg", "fire"},
        "scan": {"lbfgs", "cg"},
        "ts": {"dimer", "string", "neb"},
        "freq": {"mw", "nonmw"},
        "sp": set()
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

            # ✅ task-style line: #ts(...) / #opt / #scan(...)
            task_match = re.match(r"#\s*([A-Za-z0-9_]+)(?:\((.*?)\))?$", line)
            if task_match:
                key = task_match.group(1).strip().lower()
                if key in cls.SUPPORTED_TASKS:
                    if task and task != key:
                        cls._log_error(output_path, f"Multiple tasks defined: '{task}' and '{key}'. Only one is allowed.")
                        raise ValueError(f"Multiple tasks defined: '{task}' and '{key}'.")

                    task = key
                    params.update(cls.DEFAULTS.get(key, {}))
                    inner = task_match.group(2)
                    if inner:
                        for kv in inner.split(","):
                            kv = kv.strip()
                            if "=" in kv:
                                subkey, value = kv.split("=", 1)
                                subkey = subkey.strip()
                                value = cls._auto_cast(value.strip())
                                params[subkey] = value
                            else:
                                params[kv] = True
                    log_lines.append(f"Task set to '{task}'\n")
                    continue

            # ✅ global style line: #model=..., #gpuid=..., #d4=...
            generic_match = re.match(r"#\s*([A-Za-z0-9_]+)\s*=\s*(.+)", line)
            if generic_match:
                gkey = generic_match.group(1).strip().lower()
                gvalue = cls._auto_cast(generic_match.group(2).strip())

                if gkey in seen_keys:
                    cls._log_error(output_path, f"Duplicate definition for '{gkey}'.")
                    raise ValueError(f"Duplicate definition for '{gkey}'.")
                seen_keys.add(gkey)
                params[gkey] = gvalue
                log_lines.append(f"Global parameter: {gkey} = {gvalue}\n")

        # ✅ default to 'sp' if no task found
        if not task:
            task = "sp"
            params.update(cls.DEFAULTS.get("sp", {}))
            log_lines.append("No task specified. Defaulting to 'sp'.\n")

        # ✅ run validation and implementation checks
        cls._validate(params, task, output_path)
        cls._log_info(output_path, log_lines)

        return cls(params, task, output_path)

    @staticmethod
    def _auto_cast(value: str) -> Any:
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @classmethod
    def _validate(cls, params: Dict[str, Any], task: str, output_path: Optional[str]) -> None:
        # model
        if "model" in params and params["model"] is not None and params["model"] not in cls.SUPPORTED_MODELS:
            cls._log_error(output_path, f"Unsupported model: {params['model']}")
            raise ValueError(f"Unsupported model: '{params['model']}'.")

        # gpuid
        if "gpuid" in params and params["gpuid"] is not None and not isinstance(params["gpuid"], int):
            cls._log_error(output_path, "GPU ID must be an integer.")
            raise ValueError("GPU ID must be an integer.")

        # d4
        if "d4" in params and not isinstance(params["d4"], bool):
            cls._log_error(output_path, "D4 must be 'true' or 'false'.")
            raise ValueError("D4 must be 'true' or 'false'.")

        # implementation-level check: method validity
        if "method" in params:
            allowed_methods = cls.IMPLEMENTATION_MAP.get(task, set())
            if allowed_methods and params["method"] not in allowed_methods:
                cls._log_error(
                    output_path,
                    f"Method '{params['method']}' not implemented for task '{task}'. "
                    f"Allowed methods: {', '.join(sorted(allowed_methods))}"
                )
                raise ValueError(
                    f"Method '{params['method']}' is not implemented for task '{task}'. "
                    f"Allowed methods: {', '.join(sorted(allowed_methods))}"
                )

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

import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional


class CommandControl:
    """
    Parse and validate input settings.
    One task only: sp/opt/ts/scan/freq/irc/md.
    All other settings are global parameters.
    """

    SUPPORTED_TASKS = {"sp", "opt", "ts", "scan", "freq", "irc", "md", "phonon"}

    SUPPORTED_UMA_TASKS = {"omol", "omat", "oc20", "odac", "omc", "oc22", "oc25"}
    SUPPORTED_UMA_SIZES = {"uma-s-1p1", "uma-s-1p2", "uma-m-1p1"}
    SUPPORTED_UMA_INFERENCE = {"default", "turbo"}
    UMA_DEFAULT_SIZE = "uma-s-1p1"  # keep in sync with _uma_calculator.UMA_DEFAULT_SIZE
    SUPPORTED_HESSIAN_MODES = {"analytic", "numerical"}

    DEFAULTS = {
        "model": None,
        "device": None,
        "d4": False,
        "sp": {},
        "opt": {},
        "phonon": {
            "supercell": "3,3,3",
            "displacement": 0.01,
            "npoints": 20,
            "kpath": None,
            "fmax": 0.05,
            "relax": True,
            "units": "THz",
            "device": "cpu",
        },
        "ts": {},
        "irc": {"method": "gs"},
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
    }

    IMPLEMENTATION_MAP = {
        "opt": {"lbfgs", "rfo", "sd", "cg", "sdcg", ""},
        "scan": {"lbfgs", "rfo", "sd", "cg", "sdcg"},
        "ts": {"prfo", "string", "neb", "dimer", "autoneb"},
        "freq": {"mw", "nonmw", "both"},
        "sp": set(),
        "irc": {"gs", "hpc", "eulerpc", "lqa"},
        "md": {"nve", "nvt", "npt"},
    }
    GLOBAL_PARAMS = {
        "model",
        "model_options",
        "device",
        "gpuid",
        "d4",
        "pbc",
        "solv",
        "level",
    }
    LBFGS_PARAMS = {
        "memory",
        "curvature",
        "max_step",
        "max_iter",
        "verbose",
        "log_final_paths",
    }
    RFO_PARAMS = {
        "max_iter",
        "trust_radius_init",
        "trust_radius_min",
        "trust_radius_max",
        "eta_shrink",
        "eta_expand",
        "evals_eps",
        "mu_margin",
        "max_bisect_it",
        "verbose",
        "log_final_paths",
    }
    SDCG_PARAMS = {
        "max_step",
        "max_iter",
        "verbose",
        "sd_enabled",
        "cg_enabled",
        "sd_max_iter",
        "cg_switch_fmax",
        "cg_restart_threshold",
        "cg_beta_method",
        "diis_enabled",
        "diis_store_every",
        "diis_min_snapshots",
        "diis_memory",
        "log_final_paths",
    }
    OPT_METHOD_PARAMS = {
        "lbfgs": LBFGS_PARAMS,
        "rfo": RFO_PARAMS,
        "sd": SDCG_PARAMS,
        "cg": SDCG_PARAMS,
        "sdcg": SDCG_PARAMS,
    }
    SCAN_PARAMS = {"method", "mode"}
    SOLV_PARAMS = {
        "method",
        "implicit",
        "explicit",
        "solvent",
        "radius",
        "padding",
        "shape",
        "box_size",
        "density",
        "density_scale",
        "number",
        "clash_method",
        "tolerance",
        "vdw_scale",
        "vdw_fallback_radius",
        "seed",
        "randomize",
        "experimental",
        "write_shell",
        "shell_cutoff",
        "solvent_pdb",
        # Compatibility aliases / explicit rejections.
        "clash_cutoff",
        "write_cell",
    }
    SOLV_REMOVED_PARAMS = {
        "fix_dis": (
            "Explicit solvent 'fix_dis' has been removed: clusters are now "
            "non-periodic and no atom constraints are written. Use "
            "write_shell=true with shell_cutoff=<Å> to export a solute-centred "
            "shell instead."
        ),
    }

    VALIDATED_TASK_PARAMS = {"opt", "scan", "md"}

    TS_REFINE_MAP = {
        "neb": {"cineb", "nebts"},
        "string": {"cistring", "stringts"},
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
        cls._normalize_method_flags(params, task, output_path)
        cls._validate(params, task, output_path)
        cls._log_info(output_path, log_lines)

        return cls(params, task, output_path)

    @staticmethod
    def _normalize_key(key: str) -> str:
        return key.strip().replace("\ufeff", "").lower()

    @staticmethod
    def _parse_nested(target: Dict[str, Any], inner: str) -> None:
        for kv in inner.split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                target[CommandControl._normalize_key(k)] = CommandControl._auto_cast(v.strip())
            else:
                target[CommandControl._normalize_key(kv)] = True

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
            params["model"] = str(params["model"]).strip().lower()

        model_options = params.get("model_options")
        if isinstance(model_options, dict):
            for key in ("task", "size", "hessian", "inference"):
                if key in model_options and isinstance(model_options[key], str):
                    model_options[key] = model_options[key].lower()

        solv_options = params.get("solv")
        if isinstance(solv_options, dict):
            for key in ("shape", "explicit", "method", "implicit", "solvent", "clash_method"):
                if key in solv_options and isinstance(solv_options[key], str):
                    solv_options[key] = solv_options[key].lower()
            if solv_options.get("shape") == "box":
                solv_options["shape"] = "cube"

        if "ensemble" in params and isinstance(params["ensemble"], str):
            params["ensemble"] = params["ensemble"].lower()

    @classmethod
    def _normalize_method_flags(
        cls, params: Dict[str, Any], task: str, output_path: Optional[str]
    ) -> None:
        allowed = {
            method for method in cls.IMPLEMENTATION_MAP.get(task, set()) if method
        }
        flags = sorted(method for method in allowed if params.get(method) is True)
        if not flags:
            return
        if len(flags) > 1:
            msg = f"Multiple method flags for task '{task}': {flags}"
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        method = flags[0]
        current = params.get("method")
        if current is not None and current != method:
            msg = (
                f"Conflicting method settings for task '{task}': "
                f"'{current}' and '{method}'"
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)
        params["method"] = method
        del params[method]

    @classmethod
    def _raise_unknown_param(
        cls,
        output_path: Optional[str],
        context: str,
        key: str,
        allowed: set[str],
    ) -> None:
        msg = f"Unknown {context} parameter: '{key}'."
        match = get_close_matches(key, sorted(allowed), n=1, cutoff=0.72)
        if match:
            msg += f" Did you mean '{match[0]}'?"
        cls._log_error(output_path, msg)
        raise ValueError(msg)

    @classmethod
    def _allowed_task_params(cls, task: str, params: Dict[str, Any]) -> Optional[set[str]]:
        if task not in cls.VALIDATED_TASK_PARAMS:
            return None

        allowed = set(cls.GLOBAL_PARAMS)
        if task == "md":
            allowed.update(cls.DEFAULTS["md"])
            return allowed

        method = str(params.get("method") or "lbfgs").lower()
        method_params = cls.OPT_METHOD_PARAMS.get(method)
        if method_params is None:
            method_params = set().union(*cls.OPT_METHOD_PARAMS.values())

        allowed.add("method")
        allowed.update(method_params)
        if task == "scan":
            allowed.update(cls.SCAN_PARAMS)
        return allowed

    @classmethod
    def _validate_unknown_params(
        cls, params: Dict[str, Any], task: str, output_path: Optional[str]
    ) -> None:
        allowed = cls._allowed_task_params(task, params)
        if allowed is not None:
            context = task.upper()
            for key in params:
                if key not in allowed:
                    cls._raise_unknown_param(output_path, context, key, allowed)

        if "solv" in params:
            solv_params = params["solv"]
            if not isinstance(solv_params, dict):
                msg = "Solvation settings must use '#solv(key=value,...)' syntax."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            for key in solv_params:
                if key in cls.SOLV_REMOVED_PARAMS:
                    msg = cls.SOLV_REMOVED_PARAMS[key]
                    cls._log_error(output_path, msg)
                    raise ValueError(msg)
                if key not in cls.SOLV_PARAMS:
                    cls._raise_unknown_param(
                        output_path, "solvation", key, cls.SOLV_PARAMS
                    )

    @classmethod
    def _validate_solvation(
        cls, params: Dict[str, Any], task: str, output_path: Optional[str]
    ) -> None:
        solv_params = params.get("solv")
        if not isinstance(solv_params, dict):
            return

        solvent_alias = solv_params.pop("solvent", None)
        if solvent_alias is not None:
            explicit = solv_params.get("explicit")
            implicit = solv_params.get("implicit")
            if explicit is None and implicit is None:
                # Backward-compatible interpretation:
                #   #solv(method=gbsa, solvent=water) -> implicit solvent
                #   #solv(solvent=water)              -> explicit solvent
                target = "implicit" if solv_params.get("method") else "explicit"
                solv_params[target] = solvent_alias
            elif solvent_alias not in {explicit, implicit}:
                msg = (
                    "Conflicting solvation alias 'solvent': use explicit=<name> "
                    "and/or implicit=<name> directly when they differ."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        for key in ("randomize", "write_shell", "experimental"):
            if key in solv_params and not isinstance(solv_params[key], bool):
                msg = f"Solvation {key} must be 'true' or 'false'."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        method = solv_params.get("method")
        explicit = solv_params.get("explicit")
        implicit = solv_params.get("implicit")

        if method is not None:
            method = str(method).lower()
            solv_params["method"] = method
            if method != "gbsa":
                msg = "Implicit solvation method must be 'gbsa'."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if explicit is not None and implicit is not None:
            msg = "Use either explicit=<solvent> or implicit=<solvent>, not both."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if implicit is not None:
            if method != "gbsa":
                msg = "Implicit solvation requires method=gbsa."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if str(implicit).lower() in {"", "none"}:
                msg = "Implicit solvation requires a real solvent name, not 'none'."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if solv_params.get("experimental") is not True:
                msg = (
                    "Implicit GB-polar/QEq solvation is experimental and "
                    "energy-only; add experimental=true in #solv(...) to "
                    "request it explicitly."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if task != "sp":
                msg = (
                    "Implicit GB-polar/QEq solvation is currently energy-only "
                    "and may be used only with task 'sp'."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if params.get("verbose", 0) >= 1:
                msg = (
                    "Implicit GB-polar/QEq solvation is energy-only and supports "
                    "only #sp(verbose=0); verbose=1 requests gradients/forces."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "pbc" in params:
                msg = "Implicit GB-polar/QEq solvation is non-periodic; remove #pbc."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            explicit_only = {
                "radius",
                "padding",
                "shape",
                "box_size",
                "density",
                "density_scale",
                "number",
                "clash_method",
                "tolerance",
                "vdw_scale",
                "vdw_fallback_radius",
                "seed",
                "randomize",
                "write_shell",
                "shell_cutoff",
                "solvent_pdb",
                "clash_cutoff",
                "write_cell",
            }
            conflicts = sorted(key for key in explicit_only if key in solv_params)
            if conflicts:
                msg = (
                    "Explicit-solvent options cannot be combined with implicit "
                    f"GB-polar solvation: {', '.join(conflicts)}."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            return

        if method is not None:
            msg = "method=gbsa requires implicit=<solvent>; omit method for explicit solvent."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "experimental" in solv_params:
            msg = "experimental=true is only valid with method=gbsa, implicit=<solvent>."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if explicit is None:
            if solv_params:
                msg = "Solvation requires either explicit=<solvent> or implicit=<solvent>."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            return

        if "pbc" in params:
            msg = (
                "Explicit solvent clusters are non-periodic coordinate-only clusters; "
                "remove #pbc or use a periodic solvent backend."
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "write_cell" in solv_params:
            msg = (
                "Explicit solvent clusters are non-periodic; "
                "write_cell/PBC output is not supported."
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        shape = str(solv_params.get("shape", "sphere")).lower()
        if shape == "box":
            shape = "cube"
        solv_params["shape"] = shape
        if shape not in {"sphere", "cube"}:
            msg = "Explicit solvent shape must be 'sphere' or 'cube'."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "seed" in solv_params and type(solv_params["seed"]) is not int:
            msg = (
                "Explicit solvent seed must be an integer; "
                "use seed=-1 for non-reproducible sampling."
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "number" in solv_params and (
            type(solv_params["number"]) is not int or solv_params["number"] < 0
        ):
            msg = "Explicit solvent number must be an integer >= 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        numeric_keys = (
            "radius",
            "padding",
            "box_size",
            "density",
            "density_scale",
            "tolerance",
            "vdw_scale",
            "vdw_fallback_radius",
            "shell_cutoff",
            "clash_cutoff",
        )
        for key in numeric_keys:
            if key in solv_params and (
                isinstance(solv_params[key], bool)
                or not isinstance(solv_params[key], (int, float))
            ):
                msg = f"Explicit solvent {key} must be numeric."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if "solvent_pdb" in solv_params:
            if (
                not isinstance(solv_params["solvent_pdb"], str)
                or not solv_params["solvent_pdb"].strip()
            ):
                msg = "Explicit solvent solvent_pdb must be a non-empty path string."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if "clash_method" not in solv_params:
            if "tolerance" in solv_params or "clash_cutoff" in solv_params:
                solv_params["clash_method"] = "distance"
            else:
                solv_params["clash_method"] = "vdw"

        if solv_params["clash_method"] not in {"vdw", "distance"}:
            msg = "Explicit solvent clash_method must be 'vdw' or 'distance'."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "padding" in solv_params and solv_params["padding"] <= 0:
            msg = "Explicit solvent padding must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if shape == "sphere":
            if "padding" in solv_params and "radius" in solv_params:
                msg = (
                    "Explicit solvent padding derives the sphere radius from the "
                    "solute envelope; do not combine padding with radius."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            radius = solv_params.get("radius", 10.0)
            if radius <= 0:
                msg = "Explicit solvent radius must be > 0 for shape=sphere."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "padding" not in solv_params:
                solv_params.setdefault("radius", radius)
            if "box_size" in solv_params:
                msg = "Explicit solvent box_size is only valid for shape=cube."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
        else:
            if "radius" in solv_params:
                msg = "Explicit solvent radius is only valid for shape=sphere."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "padding" in solv_params and "box_size" in solv_params:
                msg = (
                    "Explicit solvent padding derives the cube box_size from the "
                    "solute envelope; do not combine padding with box_size."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "padding" not in solv_params and "box_size" not in solv_params:
                msg = "Explicit solvent shape=cube requires box_size."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "box_size" in solv_params and solv_params["box_size"] <= 0:
                msg = "Explicit solvent box_size must be > 0."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

        if "density" in solv_params and solv_params["density"] <= 0:
            msg = "Explicit solvent density must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if "density_scale" in solv_params and solv_params["density_scale"] <= 0:
            msg = "Explicit solvent density_scale must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if solv_params["clash_method"] == "vdw":
            if "tolerance" in solv_params or "clash_cutoff" in solv_params:
                msg = (
                    "Explicit solvent tolerance/clash_cutoff are only valid "
                    "with clash_method=distance."
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if "vdw_scale" in solv_params and solv_params["vdw_scale"] <= 0:
                msg = "Explicit solvent vdw_scale must be > 0."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if (
                "vdw_fallback_radius" in solv_params
                and solv_params["vdw_fallback_radius"] <= 0
            ):
                msg = "Explicit solvent vdw_fallback_radius must be > 0."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
        elif "tolerance" in solv_params and solv_params["tolerance"] <= 0:
            msg = "Explicit solvent tolerance must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if solv_params["clash_method"] == "distance" and (
            "vdw_scale" in solv_params or "vdw_fallback_radius" in solv_params
        ):
            msg = (
                "Explicit solvent vdw_scale/vdw_fallback_radius are only valid "
                "with clash_method=vdw."
            )
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if (
            solv_params["clash_method"] == "distance"
            and "clash_cutoff" in solv_params
            and solv_params["clash_cutoff"] <= 0
        ):
            msg = "Explicit solvent clash_cutoff must be > 0."
            cls._log_error(output_path, msg)
            raise ValueError(msg)

        if solv_params.get("write_shell", False):
            if "shell_cutoff" not in solv_params:
                msg = "Explicit solvent write_shell=true requires shell_cutoff."
                cls._log_error(output_path, msg)
                raise ValueError(msg)
            if solv_params["shell_cutoff"] <= 0:
                msg = "Explicit solvent shell_cutoff must be > 0."
                cls._log_error(output_path, msg)
                raise ValueError(msg)

    @classmethod
    def _validate(cls, params: Dict[str, Any], task: str, output_path: Optional[str]) -> None:
        model = params.get("model")
        # Calculator names and class-declared model_options are registry-owned:
        # SetCalculator imports builtins, honors module= / MAPLE_CALCULATOR_PLUGINS,
        # and validates class OPTION_KEYS before construction.
        cls._validate_unknown_params(params, task, output_path)
        cls._validate_solvation(params, task, output_path)

        if "gpuid" in params and params["gpuid"] is not None and not isinstance(params["gpuid"], int):
            cls._log_error(output_path, "GPU ID must be an integer.")
            raise ValueError("GPU ID must be an integer.")

        if "d4" in params and not isinstance(params["d4"], bool):
            cls._log_error(output_path, "D4 must be 'true' or 'false'.")
            raise ValueError("D4 must be 'true' or 'false'.")

        if task == "sp":
            if "verbosity" in params:
                cls._log_error(output_path, "SP uses 'verbose', not 'verbosity'.")
                raise ValueError("SP uses 'verbose', not 'verbosity'.")
            if "verbose" in params and (
                type(params["verbose"]) is not int or params["verbose"] not in {0, 1}
            ):
                cls._log_error(output_path, "SP verbose must be 0 or 1.")
                raise ValueError("SP verbose must be 0 or 1.")

        if "method" in params:
            if task == "md":
                cls._log_error(output_path, "'method' is not valid for MD tasks; use 'ensemble=' instead.")
                raise ValueError("'method' is not valid for MD tasks. Use 'ensemble=' to choose nve/nvt/npt.")

            allowed = cls.IMPLEMENTATION_MAP.get(task, set())
            if allowed and params["method"] not in allowed:
                cls._log_error(output_path, f"Method '{params['method']}' not implemented for task '{task}'.")
                raise ValueError(f"Method '{params['method']}' not implemented for task '{task}'.")

        if task == "ts" and "refine" in params:
            method = params.get("method")
            allowed_refines = cls.TS_REFINE_MAP.get(method)
            if allowed_refines is None:
                cls._log_error(output_path, f"'refine' is not valid for TS method '{method}'.")
                raise ValueError(f"'refine' is not valid for TS method '{method}'.")
            if params["refine"] not in allowed_refines:
                cls._log_error(output_path, f"Refine '{params['refine']}' not implemented for TS method '{method}'.")
                raise ValueError(f"Refine '{params['refine']}' not implemented for TS method '{method}'.")

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

            inference_opt = model_options.get("inference")
            if inference_opt is not None and inference_opt not in cls.SUPPORTED_UMA_INFERENCE:
                msg = (
                    f"Unsupported UMA inference mode: '{inference_opt}'. "
                    f"Supported: {sorted(cls.SUPPORTED_UMA_INFERENCE)}"
                )
                cls._log_error(output_path, msg)
                raise ValueError(msg)

            if size_opt is None:
                # Make the actual checkpoint visible in summary() / .out.
                model_options.setdefault("size", cls.UMA_DEFAULT_SIZE)
                params["model_options"] = model_options

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
        model_options = self.params.get("model_options") or {}
        for key, value in self.params.items():
            if key == "model_options":
                continue  # folded into the model line below
            if key == "model" and isinstance(value, str) and model_options:
                opt_str = ",".join(f"{k}={v}" for k, v in model_options.items())
                value = f"{value}({opt_str})"
            lines.append(f"{key:<15}: {value}\n")
        return "".join(lines)

    def __repr__(self) -> str:
        return f"CommandControl(task={self.task}, params={self.params})"

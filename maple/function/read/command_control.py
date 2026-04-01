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
        "egret", "aimnet2", "uma", "maceomol", "aimnet2nse"
    }

    SUPPORTED_TASKS = {"sp", "opt", "ts", "scan", "freq", "irc", "md"}

    # Defaults assigned only when task is selected
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
            "ensemble":        "nve",
            # timestep: NVE default 0.25 fs (optimal for universal MLFFs).
            #
            # dt-sweep on Ala-Glu dipeptide (30 atoms, gas phase, UMA-S-1p1, NVE)
            # showed σ(TE) is nearly flat across dt = 0.125–0.5 fs
            # (1.30, 1.06, 1.18 kcal/mol), confirming
            # that the energy-conservation floor is set by MLFF prediction noise,
            # not by VV integrator error (which would scale as dt²).
            # dt = 0.25 fs achieves the minimum σ(TE) of the three candidates.
            # Refs: Fu et al. (2023) JCTC 19, 1863; Kovács et al. (2023) JPCL 14, 8725;
            #       Zhang et al. (2023) J. Chem. Phys. 159, 054801 §III.C.
            # NVT/NPT ensembles override to 1 fs via their dataclass defaults
            # (thermostat damps integrator errors, allowing larger dt).
            "timestep":        0.25,
            # steps: NVE default 400000 = 100 ps at 0.25 fs (Páll 2020; AMBER 2023).
            #        NVT/NPT override to 100000 = 100 ps at 1 fs.
            "steps":           400000,
            "temperature":     300.0,
            # traj_every/log_every: ML potentials are ~1000-3000x slower than
            # classical FFs (UMA: ~0.22 ns/day vs GROMACS ~100-1000 ns/day).
            # GROMACS defaults (nstxout=500×2fs=1ps) target μs-scale runs.
            # ML-MD runs are typically 10-100 ps; 1ps/frame gives only 10-100
            # frames — too sparse for MSD/RDF analysis.
            # Target: 100-500 frames per 10 ps → 100 steps (0.025-0.1 ps/frame).
            # NVT/NPT ensemble classes override to 100 via their dataclass defaults.
            # Refs: Fu et al. (2023) JCTC 19, 1863; Kovács et al. (2023) JPCL 14, 8725.
            "traj_every":      100,
            "log_every":       100,
            "init_velocities": True,
            "restart":         False,
            "rst_file":        "",         # path to RST checkpoint (default: auto-detect)
            "rst_every":       1000,
            "remove_com":      True,
            "remove_com_every": 100,   # NVE only: steps between COM removal
            "remove_rotation": False,   # NVE only: remove initial angular momentum
            "random_seed":     None,
            # Thermostat (NVT / NPT)
            # Langevin default: correct canonical ensemble + ergodic by construction.
            # Default in AMBER (ntt=3), NAMD, OpenMM (LangevinMiddleIntegrator),
            # LAMMPS (fix langevin), MACE (ASE Langevin), DeePMD-kit.
            # Refs: Leimkuhler & Matthews (2013) AMRX 2013, 34–56 (BAOAB);
            #       Basconi & Shirts (2013) JCTC 9, 2887 (thermostat comparison).
            # V-rescale is available as alternative (Bussi et al. 2007 JCP 126, 014101).
            "thermostat":      "langevin",
            # friction = 0.001 1/fs = 1 ps⁻¹ (Langevin only)
            # Leimkuhler & Matthews (2013) AMRX; AMBER gamma_ln=1; NAMD langevinDamping=1.
            "friction":        0.001,
            # tau_t = 100 fs: GROMACS default (Manual 2024); Bussi 2007 test value;
            #   LAMMPS fix nvt Tdamp=0.1 ps (metal units).
            "tau_t":           100.0,
            # Barostat (NPT only)
            "barostat":        "c-rescale",
            "pressure":        1.0,       # bar
            "tau_p":           2000.0,    # fs
            "compressibility": 4.5e-5,    # 1/bar (water at 300 K, CRC Handbook)
            "mdp":             None,      # path to GROMACS-style .mdp file
            "traj_format":     "xyz",      # trajectory format: "xyz" (text, default) or "dcd" (binary)
        },
        "solv": {"solvent": "water", "explicit": None},
    }

    IMPLEMENTATION_MAP = {
        "opt":  {"lbfgs", "rfo", "cg", ""},
        "scan": {"lbfgs", "cg"},
        "ts":   {"prfo", "string", "neb", "dimer", "afir", "descafir", "autoneb"},
        "freq": {"mw", "nonmw", "both"},
        "sp":   set(),
        "irc":  {"gs"},
        "md":   {"nve", "nvt", "npt"},
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

            # ✅ Unified parsing: #key = value OR #key(value) OR #key=value(nested)
            match = re.match(r"#\s*([A-Za-z0-9_]+)\s*(?:=\s*([^()\s]+))?\s*(?:\((.*)\))?", line)
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
                inline_md_keys = set()
                if paren_val:
                    cls._parse_nested(params, paren_val)
                    if task == 'md':
                        inline_md_keys = {
                            kv.split('=', 1)[0].strip()
                            for kv in paren_val.split(',')
                            if '=' in kv
                        }

                # Load MDP file - REQUIRED for MD tasks (inline params override MDP)
                if task == 'md':
                    if not params.get('mdp'):
                        error_msg = """
ERROR: MD tasks require an MDP configuration file.

MAPLE uses GROMACS-style MDP files as the primary input format for MD parameters.

Example MDP file (save as 'md_config.mdp'):
    ; NVT production run
    integrator = md
    timestep = 1.0     ; fs
    nsteps = 100000    ; total steps
    ref-t = 300.0      ; K
    tcoupl = langevin

Usage in your input file:
    #md(mdp=md_config.mdp)

You can override MDP parameters inline:
    #md(mdp=md_config.mdp, ref-t=400.0)

For a complete parameter reference, see the MAPLE MD documentation.
"""
                        cls._log_error(output_path, error_msg)
                        raise ValueError(
                            "MD tasks require an MDP file. "
                            "Add 'mdp=path/to/config.mdp' to your #md(...) directive."
                        )
                    cls._load_mdp(params, inline_md_keys, output_path)

                continue

            # ✅ Global parameters
            if key in seen_keys:
                cls._log_error(output_path, f"Duplicate parameter: '{key}'.")
                raise ValueError(f"Duplicate parameter: '{key}'.")
            seen_keys.add(key)

            # Parenthesized nested form
            if paren_val:
                # Special case: PBC with flexible cell parameters
                if key == 'pbc':
                    # #pbc(a,b,c,alpha,beta,gamma) - 6 values: full cell parameters
                    # #pbc(a,b,c) - 3 values: defaults to alpha=beta=gamma=90
                    # #pbc(a,b) - 2 values: defaults to c=1000, alpha=beta=gamma=90
                    try:
                        # Remove '=' if present at the start
                        paren_val_clean = paren_val.lstrip('=').strip()
                        values = [float(x.strip()) for x in paren_val_clean.split(',')]

                        if len(values) == 2:
                            # 2 values: a, b, defaults c=1000, angles=90
                            a, b = values
                            cellpar = [a, b, 1000.0, 90.0, 90.0, 90.0]
                            log_lines.append(f"Global parameter: pbc = {values} (expanded to {cellpar})\n")
                        elif len(values) == 3:
                            # 3 values: a, b, c, defaults angles=90
                            a, b, c = values
                            cellpar = [a, b, c, 90.0, 90.0, 90.0]
                            log_lines.append(f"Global parameter: pbc = {values} (expanded to {cellpar})\n")
                        elif len(values) == 6:
                            # 6 values: full cell parameters
                            cellpar = values
                            log_lines.append(f"Global parameter: pbc = {cellpar}\n")
                        else:
                            cls._log_error(output_path, f"PBC requires 2, 3, or 6 values, got {len(values)}")
                            raise ValueError(f"PBC requires 2, 3, or 6 values (a,b[,c][,alpha,beta,gamma]), got {len(values)}")

                        # Validate values
                        if any(cellpar[i] <= 0 for i in range(3)):  # a, b, c must be positive
                            cls._log_error(output_path, f"PBC lattice parameters (a,b,c) must be positive")
                            raise ValueError(f"PBC lattice parameters must be positive")

                        params['pbc'] = cellpar
                    except ValueError as e:
                        cls._log_error(output_path, f"Invalid PBC values: {paren_val} - {e}")
                        raise ValueError(f"Invalid PBC values: {paren_val}")
                    continue

                # Special case: model with both base name and nested parameters
                # #model=uma(task=oc20) should create {'name': 'uma', 'task': 'oc20'}
                if key == 'model' and assign_val:
                    value = cls._auto_cast(assign_val.strip())
                    model_params = {'name': value}
                    cls._parse_nested(model_params, paren_val)
                    params[key] = model_params
                    log_lines.append(f"Global parameter: {key} = {model_params}\n")
                    continue

                # Generic parenthesized form for other parameters
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
            if isinstance(params['model'], dict):
                # Model with nested parameters: normalize the 'name' key
                if 'name' in params['model']:
                    params['model']['name'] = params['model']['name'].lower().replace('_', '').replace('-', '').replace(' ', '')
            else:
                # Simple model string
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

    @classmethod
    def _load_mdp(cls, params: dict, inline_keys: set[str], output_path: Optional[str] = None) -> None:
        """
        Load parameters from a GROMACS-style MDP file into params.

        Inline parameters from the #md(...) line take precedence over MDP file
        values. MDP values are only applied to known MD keys that were not
        specified inline.

        Args:
            params: Parameter dict to update in-place
            inline_keys: Keys explicitly provided inline in #md(...)
            output_path: Output file path for error logging
        """
        from ..dispatcher.md.mdp_reader import parse_mdp
        mdp_path = params['mdp']
        try:
            mdp_params = parse_mdp(mdp_path)
        except FileNotFoundError:
            cls._log_error(output_path, f"MDP file not found: {mdp_path!r}")
            raise
        except ValueError as e:
            cls._log_error(output_path, str(e))
            raise

        defaults = cls.DEFAULTS.get('md', {})
        for key, mdp_val in mdp_params.items():
            if key in defaults and key not in inline_keys:
                params[key] = mdp_val

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
        if "model" in params and params["model"] is not None:
            # Handle both simple model string and dict with nested parameters
            model_name = params["model"]
            if isinstance(params["model"], dict):
                model_name = params["model"].get('name', '')

            if model_name not in cls.SUPPORTED_MODELS:
                cls._log_error(output_path, f"Unsupported model: {model_name}")
                raise ValueError(f"Unsupported model: '{model_name}'.")

        # Validate UMA model parameters
        if "model" in params and params["model"] is not None:
            model_val = params["model"]
            if isinstance(model_val, dict):
                model_name = model_val.get('name', '')
                if model_name == 'uma':
                    # Validate task parameter
                    valid_tasks = {'omol', 'oc20', 'omat', 'omc', 'odac'}
                    task_param = model_val.get('task', 'omol')
                    if task_param not in valid_tasks:
                        cls._log_error(output_path, f"Invalid UMA task: {task_param}. Valid: {valid_tasks}")
                        raise ValueError(f"Invalid UMA task: '{task_param}'. Valid options: {valid_tasks}")

                    # Validate size parameter
                    valid_sizes = {'uma-s-1p1', 'umas1p1', 'uma-m-1p1', 'umam1p1'}
                    size_param = model_val.get('size', 'uma-s-1p1')
                    # Normalize size for comparison
                    size_normalized = size_param.lower().replace('-', '').replace('_', '')
                    if size_normalized not in {'umas1p1', 'umam1p1'}:
                        cls._log_error(output_path, f"Invalid UMA size: {size_param}. Valid: uma-s-1p1, uma-m-1p1")
                        raise ValueError(f"Invalid UMA size: '{size_param}'. Valid options: uma-s-1p1, uma-m-1p1")

        # Validate PBC parameters
        if "pbc" in params:
            pbc_val = params["pbc"]
            if not isinstance(pbc_val, list) or len(pbc_val) != 6:
                cls._log_error(output_path, "PBC must be a list of 6 values [a, b, c, alpha, beta, gamma]")
                raise ValueError("PBC must be a list of 6 values")

            # Validate lattice parameters (a, b, c) are positive
            if any(pbc_val[i] <= 0 for i in range(3)):
                cls._log_error(output_path, "PBC lattice parameters (a, b, c) must be positive")
                raise ValueError("PBC lattice parameters must be positive")

            # Validate angles are in valid range (0, 180)
            if any(pbc_val[i] <= 0 or pbc_val[i] >= 180 for i in range(3, 6)):
                cls._log_error(output_path, "PBC angles (alpha, beta, gamma) must be in range (0, 180) degrees")
                raise ValueError("PBC angles must be in range (0, 180) degrees")

            # Check PBC compatibility with UMA task
            if "model" in params and params["model"] is not None:
                model_val = params["model"]
                if isinstance(model_val, dict):
                    model_name = model_val.get('name', '')
                    task_param = model_val.get('task', 'omol')
                    if model_name == 'uma' and task_param == 'omol':
                        cls._log_error(output_path, "PBC should not be used with UMA task='omol' (molecular systems)")
                        raise ValueError("PBC is incompatible with UMA task='omol'")

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
            if task == "md":
                # MD uses 'ensemble', not 'method'
                cls._log_error(output_path,
                    f"'method' is not a valid MD parameter. Did you mean 'ensemble={params['method']}'?")
                raise ValueError(
                    f"'method' is not a valid MD parameter. Use 'ensemble=' to specify the MD ensemble "
                    f"(nve, nvt, npt). Did you mean 'ensemble={params['method']}'?")
            allowed = cls.IMPLEMENTATION_MAP.get(task, set())
            if allowed and params["method"] not in allowed:
                cls._log_error(output_path, f"Method '{params['method']}' not implemented for task '{task}'.")
                raise ValueError(f"Method '{params['method']}' not implemented for task '{task}'.")

        # check md ensemble compatibility
        if task == "md":
            if "ensemble" in params:
                allowed = cls.IMPLEMENTATION_MAP.get("md", set())
                if params["ensemble"] not in allowed:
                    cls._log_error(output_path, f"MD ensemble '{params['ensemble']}' not supported.")
                    raise ValueError(f"MD ensemble '{params['ensemble']}' not supported. Choose from: {allowed}")
        else:
            if "ensemble" in params:
                cls._log_error(output_path,
                    f"'ensemble' is not a valid parameter for task '{task}'. "
                    f"'ensemble' is only used with #md(ensemble=nve/nvt/npt).")
                raise ValueError(
                    f"'ensemble' is not a valid parameter for task '{task}'. "
                    f"'ensemble' is only used with #md(ensemble=nve/nvt/npt).")

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

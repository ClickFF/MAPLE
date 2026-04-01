"""
MDP template generator for MAPLE MD ensembles.

Provides MDP template files for NVE, NVT, and NPT ensembles.
"""

import os
from pathlib import Path


# MDP template definitions (MAPLE internal parameter names)
_MDP_TEMPLATES = {
    'nve': """\
; NVE (microcanonical) ensemble - Constant N, V, E
; MAPLE MD template - Velocity Verlet integrator
; Typical workflow: NVT equilibration -> NVE production

integrator  = md          ; Velocity Verlet
ensemble    = nve

timestep    = 0.1        ; fs
steps       = 400000     ; steps (= 40 ps)

temperature = 300.0      ; K (used only if init_velocities = yes)

traj_every  = 100        ; trajectory output every N steps
log_every   = 100        ; log energy every N steps
traj_format = xyz        ; xyz (text) or dcd (binary)

remove_com_every = 100   ; remove COM motion every N steps

init_velocities = no     ; load velocities from rst_file (NVT -> NVE)
restart     = yes
rst_file    = nvt_md.rst ; load state from NVT equilibration
rst_every   = 1000       ; checkpoint frequency
; random_seed = 12345    ; uncomment for reproducibility
""",

    'nvt': """\
; NVT (canonical) ensemble - Constant N, V, T
; MAPLE MD template - Langevin or V-rescale thermostat

integrator  = md
ensemble    = nvt

timestep    = 0.1        ; fs
steps       = 100000     ; steps (= 10 ps)

temperature = 300.0      ; K
thermostat  = langevin   ; langevin or v-rescale

friction    = 0.001      ; 1/fs (Langevin only)
; tau_t     = 100.0      ; fs (V-rescale only)

traj_every  = 100
log_every   = 100
traj_format = xyz

init_velocities = yes
restart     = no
; rst_file  =            ; load state from a .rst checkpoint
rst_every   = 1000
""",

    'npt': """\
; NPT (isothermal-isobaric) ensemble - Constant N, P, T
; MAPLE MD template - V-rescale thermostat + C-rescale barostat

integrator  = md
ensemble    = npt

timestep    = 0.1        ; fs
steps       = 20000      ; steps (= 2 ps)

temperature = 300.0      ; K
thermostat  = v-rescale  ; langevin or v-rescale
; tau_t     = 100.0      ; fs (V-rescale only)

pressure    = 1.0        ; bar
barostat    = c-rescale  ; berendsen or c-rescale
tau_p       = 2000.0     ; fs
compressibility = 4.5e-5 ; 1/bar

traj_every  = 100
log_every   = 100
traj_format = xyz

init_velocities = yes
restart     = no
; rst_file  =            ; load state from a .rst checkpoint
rst_every   = 1000
""",
}


def generate_mdp_template(ensemble: str, output: str = None, force: bool = False) -> None:
    """
    Generate MDP template file for the specified ensemble.

    Parameters
    ----------
    ensemble : str
        Ensemble type ('nve', 'nvt', or 'npt')
    output : str, optional
        Output filename. If not provided, defaults to '{ensemble}.mdp'
    force : bool, optional
        If True, overwrite existing file without prompting. Default is False.

    Raises
    ------
    ValueError
        If ensemble is not one of 'nve', 'nvt', or 'npt'
    FileExistsError
        If output file already exists and force=False

    Examples
    --------
    >>> generate_mdp_template('nve')  # Creates nve.mdp
    >>> generate_mdp_template('nvt', 'my_template.mdp')
    >>> generate_mdp_template('npt', force=True)  # Overwrites npt.mdp
    """
    ensemble = ensemble.lower()

    if ensemble not in _MDP_TEMPLATES:
        raise ValueError(
            f"Unknown ensemble: {ensemble}. "
            f"Valid options are: {', '.join(sorted(_MDP_TEMPLATES.keys()))}"
        )

    if output is None:
        output = f"{ensemble}.mdp"

    output_path = Path(output)

    # Check if file exists
    if output_path.exists() and not force:
        raise FileExistsError(
            f"File '{output}' already exists. Use -f/--force to overwrite."
        )

    # Write template
    output_path.write_text(_MDP_TEMPLATES[ensemble])

    print(f"Generated: {output}")
    print(f"  Ensemble: {ensemble.upper()}")
    print(f"  Edit the file to adjust parameters, then use:")
    print(f"  #md(mdp={output})")


def list_templates() -> None:
    """List available MDP templates."""
    print("Available MDP templates:")
    for name in sorted(_MDP_TEMPLATES.keys()):
        print(f"  maple md {name}")
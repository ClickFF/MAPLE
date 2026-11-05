from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import ase.calculators.calculator

if TYPE_CHECKING:
    import torch


_REGISTRY: dict[str, type] = {}


def register_calculator(cls):
    """Register a calculator class under each name in cls.MODEL_NAMES.

    Raises ValueError on duplicate registration to surface accidental name
    collisions instead of silently overwriting.
    """
    for raw_name in cls.MODEL_NAMES:
        name = raw_name.lower()
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Calculator name '{name}' is already registered to {existing.__name__}; "
                f"refusing to overwrite with {cls.__name__}."
            )
        _REGISTRY[name] = cls
    return cls


def get_registered_calculator(name: str) -> type:
    return _REGISTRY[name.lower()]


def import_calculator_plugin(module_path: str) -> None:
    import importlib
    importlib.import_module(module_path)


def load_calculator_plugins_from_env() -> None:
    import os
    raw = os.environ.get('MAPLE_CALCULATOR_PLUGINS', '')
    for entry in (s.strip() for s in raw.split(',') if s.strip()):
        import_calculator_plugin(entry)


_NONE_OPTIONS = {'', 'none', 'null', 'false', '0'}


IMPLICIT_SOLVENT_FORCE_ERROR = (
    "Experimental implicit GB-polar solvation is energy-only. Forces, stress, "
    "Hessians, and HVPs are disabled because QEq charges are "
    "geometry-dependent and are not coupled variationally to the solvent "
    "energy."
)
IMPLICIT_SOLVENT_DERIVATIVE_PROPERTIES = {
    "forces",
    "force",
    "stress",
    "stresses",
    "virial",
    "virials",
    "hessian",
}


def reject_implicit_solvent_derivatives(calculator, properties):
    """Fail fast when experimental implicit solvation is asked for derivatives."""
    if not getattr(calculator, "solvent_correction", None):
        return _property_list(properties)
    normalized = _property_list(properties)
    requested = {str(prop).lower() for prop in normalized}
    if requested.intersection(IMPLICIT_SOLVENT_DERIVATIVE_PROPERTIES):
        raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
    return normalized


def normalize_none_option(value):
    """Normalize user-facing none-like option values to the literal 'none'."""
    if value is None:
        return 'none'
    text = str(value).strip().lower()
    return 'none' if text in _NONE_OPTIONS else text


def validate_implicit_solvent_choice(implicit, solvent):
    """Normalize and validate implicit-solvent selector pair."""
    implicit_norm = normalize_none_option(implicit)
    solvent_norm = normalize_none_option(solvent)
    if implicit_norm == 'gbsa' and solvent_norm == 'none':
        raise ValueError(
            "implicit='gbsa' requires an explicit solvent name such as solvent='water'; "
            "use implicit='none' to disable implicit solvent."
        )
    return implicit_norm, solvent_norm


def parse_bool_option(value, *, name='option'):
    """Parse a model-option flag into a real bool.

    Accepts actual bools and the strings true/false/1/0/yes/no/on/off
    (case-insensitive). Raises ValueError on anything else so a typo like
    d4=flase fails loudly instead of silently enabling the flag — note that
    bool('false') is True, the exact trap this guards against.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in ('true', '1', 'yes', 'on'):
        return True
    if text in ('false', '0', 'no', 'off', ''):
        return False
    raise ValueError(
        f"Cannot parse {name}={value!r} as a boolean; use one of true/false/1/0/yes/no."
    )


EV2HARTREE = 1.0 / 27.211386245988


def _convert_properties(result, *, source_unit):
    """Convert native properties to Hartree/Å conventions; stress stays eV/Å³.

    Multiplication preserves array/Tensor dtype, device and derivative graphs.
    Neither the input dictionary nor its values are modified.
    """
    if source_unit not in ('eV', 'hartree'):
        raise ValueError(
            f"Unknown source_unit: {source_unit!r}; expected 'eV' or 'hartree'."
        )
    converted = dict(result)
    if source_unit == 'eV':
        def convert(value):
            if isinstance(value, (list, tuple)):
                return type(value)(convert(item) for item in value)
            return value * EV2HARTREE

        for name in ('energy', 'free_energy', 'forces', 'hessian', 'hvp', 'virial'):
            value = result.get(name)
            if value is not None:
                converted[name] = convert(value)
    elif result.get('stress') is not None:
        converted['stress'] = result['stress'] / EV2HARTREE
    return converted


def _convert_energy_force_units(energy, forces, *, source_unit):
    """Convert backend (energy, forces) to Hartree and Hartree/Å."""
    result = _convert_properties({'energy': energy, 'forces': forces}, source_unit=source_unit)
    return result['energy'], result['forces']


def init_implicit_solvent(calc, implicit, solvent, device):
    """Shared implicit-solvent initializer.

    Usable by CalcABC subclasses and duck-typed calculators (UMA) so the
    GBSA/QEq construction lives in one place.
    """
    implicit, solvent = validate_implicit_solvent_choice(implicit, solvent)
    if implicit == 'gbsa':
        from .extra_correction import GBSA, QEqTorch

        calc.solvent_correction = GBSA(solvent=solvent, device=device)
        calc.chargecalc = QEqTorch(device=device)
    else:
        calc.solvent_correction = None


def atoms_has_pbc(atoms) -> bool:
    """Return True when an Atoms-like object carries any periodic boundary."""
    return atoms is not None and bool(np.any(getattr(atoms, 'pbc', False)))


def reject_periodic_atoms(atoms, backend_name: str) -> None:
    """Fail loudly for molecular wrappers that do not implement PBC graphs."""
    if atoms_has_pbc(atoms):
        raise NotImplementedError(
            f"{backend_name} is a no-PBC molecular wrapper. "
            "Use UMA or a backend-native PBC calculator for periodic systems."
        )


def numerical_hessian_from_atoms(calc, atoms, delta=0.002):
    """Central force differences using MAPLE batch or plain ASE evaluation."""
    from ._batch import numerical_hessian
    return numerical_hessian(calc, atoms, delta)


def hessian_via_double_autograd(energy_fn, leaf):
    """Row-by-row analytic Hessian via double autograd.

    Shared by every CalcABC backend whose ``_analytic_hessian`` builds the
    Hessian one column at a time. ``leaf`` is the position tensor created with
    ``requires_grad=True``; ``energy_fn()`` must return a scalar energy tensor
    that depends on ``leaf`` **already in the target unit** (eV-native backends
    multiply by ``EV2HARTREE`` inside ``energy_fn``; ANI is Hartree-native).

    Returns a ``(3N, 3N)`` float ndarray. Polymorphic over ``leaf`` shape —
    ``(N, 3)`` for MACE-style backends and ``(1, N, 3)`` for ANI both flatten to
    ``3N`` — because it only ever reshapes gradients to 1-D.
    """
    import torch

    energy = energy_fn()
    n = leaf.numel()
    hessian = torch.zeros((n, n), dtype=leaf.dtype, device=leaf.device)
    grad = torch.autograd.grad(energy, leaf, create_graph=True)[0].reshape(-1)
    for i in range(n):
        grad2 = torch.autograd.grad(grad[i], leaf, retain_graph=True)[0].reshape(-1)
        hessian[i, :] = grad2
    return hessian.detach().cpu().numpy()


def _property_list(properties):
    return ['energy'] if properties is None else ([properties] if isinstance(properties, str) else list(properties))


def _state_value(value):
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, dict):
        return tuple((key, _state_value(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_state_value(item) for item in value)
    return value if isinstance(value, (str, int, float, bool, type(None))) else str(value)


class CalcABC(ase.calculators.calculator.Calculator):
    # Protocol attributes — each subclass overrides what's relevant.
    MODEL_NAMES: tuple = ()
    MODEL_ENERGY_UNIT: str = 'eV'
    SUPPORTED_HESSIAN_MODES: tuple = ('numerical',)
    SUPPORTS_CHARGE_MULT: bool = False
    SUPPORTS_PBC: bool = False
    CHECKPOINT_FILENAME: dict | None = None
    REQUIRES_LOCAL_MODEL_FILE: bool = False
    # None keeps legacy/plugins permissive. Shipped backends set an explicit
    # tuple so input typos fail before a model is loaded.
    OPTION_KEYS: tuple | None = None

    def __init__(self):
        super().__init__()
        self._input_state = None

    def _calculation_state(self, atoms):
        info = getattr(atoms, 'info', {})
        arrays = getattr(atoms, 'arrays', {})
        keys = ('charge', 'total_charge', 'mult', 'multiplicity', 'external_field', 'external_potential')
        config = ('device', 'dtype', 'lr', 'lr_cutoff', 'hessian', 'd4', 'implicit', 'solvent')
        return _state_value(([(key, info.get(key)) for key in keys],
                             [(key, arrays.get(key)) for key in ('external_field', 'external_potential')],
                             [(key, getattr(self, key, None)) for key in config]))

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if self._input_state != self._calculation_state(atoms):
            changes.append('calculation_context')
        return changes

    def calculate_batch(self, atoms_list, properties=None, batch_size=None):
        from ._batch import calculate_batch
        return calculate_batch(self, atoms_list, properties, batch_size)

    def _calculate_batch(self, atoms_list, properties):
        return NotImplemented

    def _batch_key(self, atoms):
        return ()

    def _store_properties(self, atoms, result):
        self._finalize_results(atoms, energy=result['energy'][0],
                               forces=result.get('forces', [None])[0],
                               hessian=result.get('hessian', [None])[0], unit='hartree')
        for name, values in result.items():
            if name not in {'energy', 'forces', 'hessian'}:
                self.results[name] = values[0]

    def _reject_unsupported_pbc(self, atoms) -> None:
        if getattr(self, 'solvent_correction', None) is not None and atoms_has_pbc(atoms):
            raise ValueError('Implicit GBSA/QEq solvent correction has no periodic implementation')
        if not self.SUPPORTS_PBC:
            reject_periodic_atoms(atoms, type(self).__name__)

    @staticmethod
    def _normalize_properties(properties):
        return _property_list(properties)

    @staticmethod
    def _total_charge_from_atoms(atoms) -> float:
        info = getattr(atoms, 'info', {})
        charge = info.get('total_charge', info.get('charge'))
        if charge is not None:
            return float(charge)

        if hasattr(atoms, "get_initial_charges"):
            initial_charges = np.asarray(atoms.get_initial_charges(), dtype=float)
            if initial_charges.size and np.all(np.isfinite(initial_charges)):
                return float(initial_charges.sum())

        return 0.0

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=ase.calculators.calculator.all_changes,
    ):
        target_atoms = atoms if atoms is not None else getattr(self, 'atoms', None)
        # A failed direct request must never leave a prior geometry/state's
        # results attached to ASE's newly copied atoms.
        self.results = {}
        self._reject_unsupported_pbc(target_atoms)
        properties = reject_implicit_solvent_derivatives(self, properties)
        unknown = set(properties) - set(self.implemented_properties)
        if unknown:
            raise ase.calculators.calculator.PropertyNotImplementedError(
                f'Unsupported calculator properties: {sorted(unknown)}')
        super().calculate(atoms, properties, system_changes)
        self._input_state = self._calculation_state(target_atoms)
        return target_atoms

    @classmethod
    def build_kwargs_from_options(cls, model, model_options, *, resolved_model_path=None):
        """Translate input-header options into ctor kwargs. Backends override."""
        return {}

    def _finalize_results(self, atoms, *, energy, forces=None, hessian=None, unit=None):
        """Single entry: unit conversion + implicit-solvent + write self.results.

        Backends pass the pure model outputs (in the unit declared by
        MODEL_ENERGY_UNIT). This method converts to Hartree, then optionally
        adds the implicit-solvent correction, then writes self.results.
        """
        source_unit = unit if unit is not None else self.MODEL_ENERGY_UNIT
        energy_ha, forces_ha = _convert_energy_force_units(
            energy, forces, source_unit=source_unit
        )

        if getattr(self, 'solvent_correction', None) is not None:
            # GB-polar solvation is energy-only. Reaching here with forces under
            # active solvent means the calculate()/get_hessian() guards were
            # bypassed; fail loudly rather than emit a solvent-inconsistent force.
            if forces_ha is not None:
                raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
            solvent_energy = self.implicit_solv_energy(atoms)
            se = solvent_energy.item() if hasattr(solvent_energy, 'item') else float(solvent_energy)
            energy_ha = energy_ha + se

        # Sole results-writing chokepoint for every CalcABC backend: clear first
        # so an energy-only call cannot inherit stale forces/hessian from a
        # previous forces/hessian call on the same calculator instance.
        self.results = {}
        self.results['energy'] = float(energy_ha)
        self.results['free_energy'] = float(energy_ha)
        if forces_ha is not None:
            self.results['forces'] = forces_ha
        if hessian is not None:
            self.results['hessian'] = hessian

    def get_hessian(self, atoms, delta: float = 0.002):
        """Dispatch on self.hessian. Subclasses may override for backend autograd."""
        mode = getattr(self, 'hessian', self.SUPPORTED_HESSIAN_MODES[0])
        if mode == 'analytic':
            self._reject_unsupported_pbc(atoms)
            if getattr(self, 'solvent_correction', None) is not None:
                raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
            return np.asarray(self._analytic_hessian(atoms))
        if mode == 'numerical':
            if getattr(self, 'solvent_correction', None) is not None:
                raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
            from ._batch import numerical_hessian
            return numerical_hessian(self, atoms, delta)
        raise ValueError(f"Unknown hessian mode: {mode!r}")

    def _analytic_hessian(self, atoms):
        """Backend autograd Hessian. Override in subclasses that can autodiff."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement analytic Hessian; "
            "set hessian='numerical' or override _analytic_hessian."
        )


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

    def get_hvp(self, atoms, n: np.ndarray):
        """Hessian-vector product Hn. Override in backends that support Dimer-mode TS.

        There is deliberately no shared default: the autograd forward shape
        differs per backend, so a single implementation would silently misread
        non-matching models. Backends that can autodiff their forward override
        this; everyone else fails loudly here instead of returning garbage.
        """
        if getattr(self, 'solvent_correction', None) is not None:
            raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_hvp; Dimer-mode TS "
            "requires a backend-specific Hessian-vector product."
        )

    def implicit_solv_init(self, implicit: str, solvent: str):
        init_implicit_solvent(self, implicit, solvent, self.device)
    
    def implicit_solv_energy(self, atoms: ase.Atoms) -> torch.Tensor:
        """
        Compute implicit solvent correction energy if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            torch.Tensor: Implicit solvent correction energy in Hartree.
        """
        atoms.atomic_charges = self.chargecalc(
            atoms, total_charge=self._total_charge_from_atoms(atoms)
        )
        solvent_energy,_ = self.solvent_correction.get_energy(atoms)
        return solvent_energy

    def implicit_solv_energy_and_force(self, atoms: ase.Atoms) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute implicit solvent correction energy and forces if applicable.

        Args:
            atoms (ase.Atoms): Atomic structure.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Implicit solvent correction energy in Hartree and forces in Hartree/Å.
        """
        raise NotImplementedError(IMPLICIT_SOLVENT_FORCE_ERROR)

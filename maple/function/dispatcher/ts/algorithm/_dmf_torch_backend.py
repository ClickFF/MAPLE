# -*- coding: utf-8 -*-
"""
Private MAPLE adapter for the upstream PyTorch Direct MaxFlux backend.

The vendored torch implementation accelerates the DMF path algebra and
FB-ENM geometry processing, but still exposes a cyipopt/ASE-style API. This
adapter keeps MAPLE-specific defaults and unit conversion outside dmf.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from maple.function.calculator._batch import energy_forces_images

TORCH_DTYPE = torch.float64
TORCH_DTYPE_LABEL = "float64"
_UPSTREAM_PACKAGE = "_maple_vendored_dmf_torch"


def _load_upstream_torch_package():
    module = sys.modules.get(_UPSTREAM_PACKAGE)
    if module is not None:
        return module

    torch_pkg = Path(__file__).with_name("dmf") / "dmf_git" / "src" / "dmf" / "torch"
    init_file = torch_pkg / "__init__.py"
    if not init_file.is_file():
        raise ImportError(f"Vendored DMF torch backend not found at {init_file}.")

    spec = importlib.util.spec_from_file_location(
        _UPSTREAM_PACKAGE,
        init_file,
        submodule_search_locations=[str(torch_pkg)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load vendored DMF torch backend from {init_file}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[_UPSTREAM_PACKAGE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_UPSTREAM_PACKAGE, None)
        raise
    return module


_UPSTREAM = _load_upstream_torch_package()
_BaseTorchDirectMaxFlux = _UPSTREAM.DirectMaxFlux
_upstream_interpolate_fbenm = _UPSTREAM.interpolate_fbenm


def _coerce_torch_device(value):
    if value is None:
        return None
    if isinstance(value, torch.device):
        return value
    try:
        return torch.device(value)
    except (TypeError, ValueError):
        return None


def _iter_child_calculators(calc):
    mixer = getattr(calc, "mixer", None)
    for owner in (mixer, calc):
        if owner is None:
            continue
        for attr in ("calcs", "calculators"):
            children = getattr(owner, attr, None)
            if children is not None:
                for child in children:
                    yield child


def resolve_torch_device_from_calc(calc):
    """Resolve the DMF torch device from MAPLE's already-configured calculator."""
    device = _coerce_torch_device(getattr(calc, "device", None))
    if device is not None:
        return device

    for child in _iter_child_calculators(calc):
        device = _coerce_torch_device(getattr(child, "device", None))
        if device is not None:
            return device

    return torch.device("cpu")


class TorchDirectMaxFlux(_BaseTorchDirectMaxFlux):
    """Upstream torch DMF with MAPLE Hartree-to-eV scaling support."""

    def __init__(self, *args, energy_force_scale=1.0, **kwargs):
        self.energy_force_scale = float(energy_force_scale)
        self.batch_size = None
        kwargs["dtype"] = TORCH_DTYPE
        super().__init__(*args, **kwargs)

    def _get_forces_by_img_idxs(self, idxs, energies, forces):
        E, F = energy_forces_images([self.images[i] for i in idxs],
                                    self.batch_size, force_consistent=False)
        for slot, i in enumerate(idxs):
            energies[i] = E[slot] * self.energy_force_scale
            forces[i] = F[slot] * self.energy_force_scale


def interpolate_fbenm_torch(ref_images, *args, dmf_options=None, device=None, **kwargs):
    """Run upstream torch FB-ENM while accepting MAPLE's shared DMF kwargs."""
    dmf_options = dict(dmf_options) if dmf_options is not None else {}
    dmf_options.pop("energy_force_scale", None)
    dmf_options["dtype"] = TORCH_DTYPE
    return _upstream_interpolate_fbenm(
        ref_images,
        *args,
        dmf_options=dmf_options,
        device=device,
        **kwargs,
    )

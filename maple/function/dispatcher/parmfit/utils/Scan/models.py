"""Usage: define silent scan input and output data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ScanConstraint:
    atoms: tuple[int, ...]
    step: float
    steps: int

    def as_legacy_row(self) -> list[float | int]:
        return [*self.atoms, float(self.step), int(self.steps)]


@dataclass(frozen=True)
class SilentScanOptions:
    mode: str = "relaxed"
    backend: str = "lbfgs"
    constraint_mode: str = "fixinternals"
    max_iter: int = 256
    max_step: float = 0.2
    memory: Optional[int] = None
    curvature: Optional[float] = None
    extra_opt: dict = field(default_factory=dict)
    batch_size: int | None = None

    def to_params(self) -> dict:
        opt = dict(self.extra_opt)
        opt.setdefault("max_iter", int(self.max_iter))
        opt.setdefault("max_step", float(self.max_step))
        if self.memory is not None:
            opt.setdefault("memory", int(self.memory))
        if self.curvature is not None:
            opt.setdefault("curvature", float(self.curvature))
        return {
            "batch_size": self.batch_size,
            "mode": str(self.mode).strip().lower(),
            "backend": str(self.backend).strip().lower(),
            "constraint_mode": str(self.constraint_mode).strip().lower(),
            "opt": opt,
        }


@dataclass(frozen=True)
class SilentScanResult:
    output_path: str
    xyz_path: str

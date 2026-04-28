# External Field Single-Point Examples

These examples demonstrate the `#external_field` directive for passing a uniform
external electric field (V/Å) to field-aware calculators directly from the
MAPLE CLI / `.inp` interface.

## Files

| File | Field (V/Å) | Purpose |
|------|-------------|---------|
| `water_zero.inp`    | (0, 0,  0.0) | Reference, no field |
| `water_plus_z.inp`  | (0, 0, +0.5) | Uniform field along +z |
| `water_minus_z.inp` | (0, 0, -0.5) | Uniform field along -z |

## Run

```bash
cd example/external_field
maple water_zero.inp
maple water_plus_z.inp
maple water_minus_z.inp
```

Compare the three `Energy:` lines in the resulting `.out` files. The +z and
-z field SPs should yield (approximately) equal-and-opposite energy shifts
relative to zero-field, with magnitude controlled by the molecular dipole
projection along z.

## The directive

```
#external_field=Ex,Ey,Ez
```

- Three comma-separated floats (Cartesian components).
- Units: **V/Å** (volts per angstrom). To convert from atomic units:
  `1 a.u. = 51.4220675 V/Å`.
- Sign convention: a positive component points along the +x / +y / +z
  laboratory axis. Energies follow the standard `E = -μ·F + ...` expansion,
  i.e. a field aligned with the molecular dipole lowers the energy.
- The field is **uniform** across the system. For non-uniform / per-atom
  fields, use the Python API
  (`atoms.info['external_field'] = (Ex, Ey, Ez)`) directly.

## Currently field-aware models

The directive flows through to the `MACEPolCalculator` forward pass for:

| Model key | Underlying weights | Notes |
|-----------|--------------------|-------|
| `macepol-s`     | MACE-POLAR-1 (S), OMol25 foundation | Small, fast, broad coverage |
| `macepol-l`     | MACE-POLAR-1 (L), OMol25 foundation | Large, more accurate |
| `macepol-ef-s`  | User fine-tune for explicit field response (registers internally as `macepolefs`) | Trained for charged organic substrates |

Calculators outside the `macepol*` family currently ignore the directive
silently — extending support is a one-line addition in the calculator
constructor.

## Validation

Implementation tested end-to-end against ORCA wB97M-V/def2-TZVPP on a
62-point reaction MEP for a charged organic Diels–Alder substrate
(56 atoms, +1 singlet) at three field magnitudes (-0.5, 0, +0.5 V/Å).
A total of 558 MAPLE single points (3 models × 3 fields × 62 geometries)
completed without error on Ibex A100 GPUs. Per-step energies from the
CLI path matched the direct-Python smoke test
(`atoms.info['external_field']`) to all printed digits.

## Known limitations

- **Comma escaping not handled in nested syntax** — if `#external_field=` is
  embedded inside a directive that already uses commas, the parser may
  mis-split. Fine for the current top-level placement but worth a design
  pass before broader use.
- **GPU compatibility — pick the right PyTorch wheel** —
  `macepol-{s,l,ef-s}` are jit-traced under PyTorch 2.5+. The cuXXX channel
  determines which compute capabilities are baked into the wheel:

  | wheel | arch_list (verified on GPU srun) | works on |
  |---|---|---|
  | `torch 2.5.1+cu121` | `sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90` | V100, 1080 Ti, A100, H100, RTX 20/30/40xx |
  | `torch 2.11+cu128` | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` | A100/H100/H200, RTX 30/40xx; **NOT** V100/1080 Ti |

  The `cu121` channel still ships sm_60/sm_70 kernels, so for multi-GPU access
  install MAPLE under a torch 2.5.1+cu121 environment. cu128 wheels dropped
  sm_70 upstream — using cu128 on a V100 raises `cudaErrorNoKernelImageForDevice`
  at `torch.jit.load`. Verified on Ibex 2026-04-28 (jobs 46781888 V100 +
  46781889 1080 Ti): all three models pass identical-to-six-figures forward
  passes against an A100 reference under torch 2.5.1+cu121.

  Diagnostic: `torch.cuda.get_arch_list()` must be queried from a real GPU
  srun, **not** a login node — on a login node it returns `[]` because CUDA
  isn't initialised, which can mislead diagnosis.

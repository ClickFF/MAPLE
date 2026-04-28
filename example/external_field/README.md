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
- **GPU architecture coupling for `macepol-ef-s`** — the user-distributed
  `macepol-ef-s.pt` was traced/saved against a PyTorch wheel without
  multi-arch fat binaries; loads on sm_80 (A100) but raises
  `cudaErrorNoKernelImageForDevice` on sm_70 (V100) / sm_61 (1080 Ti). This
  is an environment/wheel issue, not a calculator issue, and is not addressed
  in this PR.

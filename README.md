# MAPLE

**MA**chine Learning **P**otential for **L**andscape **E**xploration

MAPLE is a computational chemistry toolkit that leverages machine learning potentials for efficient molecular simulations, including structure optimization, transition state searching, and reaction pathway analysis.

---

## Features

| Category | Methods |
|----------|---------|
| **Optimization** | LBFGS, RFO |
| **Transition State** | NEB, CI-NEB, PRFO, Dimer, String (GSM), AFIR |
| **Reaction Path** | IRC (GS method) |
| **Analysis** | Frequency, PES Scan, Single Point |
| **ML Potentials** | AIMNet2, ANI, MACE, UMA |
| **Corrections** | DFT-D4 dispersion, GBSA solvation |

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA-capable GPU (recommended)

### Install

```bash
git clone https://github.com/ClickFF/maple.git
cd maple
pip install -e .
```

### Dependencies

```bash
# Core
pip install numpy scipy ase

# PyTorch (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# PyTorch (CPU only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ML potentials
pip install fairchem-core
```

---

## Usage

### Command Line

```bash
maple input.inp              # Output: input.out
maple input.inp result.out   # Custom output file
maple --test 1               # Run test case
```

### Test Cases

```bash
maple --test 1   # LBFGS optimization
maple --test 2   # NEB transition state
maple --test 3   # String method
maple --test 4   # Dimer method
maple --test 5   # RFO optimization
maple --test 6   # IRC
maple --test 7   # Frequency
maple --test 8   # PES Scan
```

---

## Input File Format

### Header Options

```
#model=<model>           # ML potential: uma, aimnet2, ani, mace
#<jobtype>(options)      # Job type with optional parameters
#device=<device>         # gpu0, gpu1, cpu
```

### Coordinates

Two ways to specify coordinates:

1. **Inline XYZ** - directly in input file:
```
#model=uma
#opt(method=lbfgs)
#device=gpu0

C   0.000   0.000   0.000
H   1.089   0.000   0.000
...
```

2. **External XYZ file** - reference path:
```
#model=uma
#opt(method=lbfgs)
#device=gpu0

XYZ /path/to/molecule.xyz
```

For multi-structure jobs (NEB, etc.), use multiple XYZ lines:
```
XYZ /path/to/reactant.xyz
XYZ /path/to/product.xyz
```

### Job Types

| Header | Description |
|--------|-------------|
| `#opt(method=lbfgs)` | Geometry optimization |
| `#sp` | Single point energy |
| `#ts(method=neb)` | Transition state search |
| `#freq` | Frequency analysis |
| `#irc` | Intrinsic reaction coordinate |
| `#scan` | PES scan |

### ML Models

| Model | Description |
|-------|-------------|
| `uma` | UMA (universal materials) |
| `aimnet2` | AIMNet2 (general organic) |
| `ANI-1xnr` | ANI (CHNO molecules) |
| `mace` | MACE (universal potential) |

---

## Examples

### Geometry Optimization

```
#model=uma
#opt(method=lbfgs)
#device=gpu0

C      -0.77812600     -1.06756100      0.32105900
C       1.30255300      0.05212000     -0.02829900
...
```

### NEB Transition State

```
#model=ANI-1xnr
#ts(method=neb,refine=nebts)
#device=gpu0

XYZ /path/to/reactant.xyz
XYZ /path/to/product.xyz
```

### Frequency Calculation

```
#model=ANI-1xnr
#freq
#device=gpu0

XYZ /path/to/optimized.xyz
```

### PES Scan

```
#model=uma
#scan
#device=gpu0

C   0.000   0.000   0.000
...

S 15 31 -0.05 50
S 13 34 -0.05 50
```

Scan format: `S <atom1> <atom2> <step_size> <n_steps>`

---

## Documentation

For detailed architecture and algorithm documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Citation

```
https://github.com/ClickFF/MAPLE
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Submit a pull request

---

## Acknowledgments

- [ASE](https://wiki.fysik.dtu.dk/ase/) - Atomic Simulation Environment
- [PyTorch](https://pytorch.org/)
- [AIMNet2](https://github.com/isayevlab/AIMNet2)
- [fair-chem](https://github.com/FAIR-Chem/fairchem)

---

**Version**: 0.1.0 | **Status**: Active Development | **Updated**: January 2026
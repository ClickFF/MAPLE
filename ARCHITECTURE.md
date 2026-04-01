# MAPLE Architecture Summary

**MAPLE**: MAchine-learning Potential for Landscape Exploration

A Python toolkit for molecular simulations using machine learning potentials, supporting geometry optimization, transition state searches, and reaction path analysis.

---

## 1. Overall Architecture

```
maple/
├── main.py                     # CLI entry point
├── function/
│   ├── engine.py               # Core engine orchestrator
│   ├── read/                   # Input parsing
│   │   ├── input_reader.py     # Main input file reader
│   │   ├── command_control.py  # Parameter parser
│   │   ├── header/             # Header parsing
│   │   └── filereader/         # XYZ file readers
│   ├── calculator/             # ML potential calculators
│   │   ├── set_calculator.py   # Calculator factory
│   │   ├── aimnet/             # AIMNet2 calculator
│   │   ├── ani/                # ANI calculator
│   │   ├── mace/               # MACE calculator
│   │   ├── uma/                # UMA calculator
│   │   └── extra_correction/   # D4, solvation corrections
│   ├── dispatcher/             # Job dispatchers
│   │   ├── dispatcher.py       # Main job dispatcher
│   │   ├── jobABC.py           # Base class for all jobs
│   │   ├── optimization/       # Geometry optimization (LBFGS, RFO)
│   │   ├── ts/                 # Transition state (NEB, PRFO, Dimer, String, AFIR)
│   │   ├── irc/                # IRC (GS method)
│   │   ├── frequency/          # Frequency analysis
│   │   ├── scan/               # PES scan
│   │   ├── sp/                 # Single point energy
│   │   └── hessian/            # Hessian calculation
│   ├── utility/                # Utility classes (Molecules)
│   ├── timer/                  # Timing utilities
│   └── units/                  # Unit conversions
```

---

## 2. Execution Flow

```
main.py (CLI)
    │
    ▼
engine.py
    │
    ├──► InputReader (read/input_reader.py)
    │       ├── Parse input file header
    │       ├── Read XYZ coordinates
    │       └── Parse command_control parameters
    │
    ├──► SetCalculator (calculator/set_calculator.py)
    │       └── Initialize ML potential (AIMNet2, ANI, MACE, UMA)
    │
    └──► Dispatcher (dispatcher/dispatcher.py)
            │
            └── Route to specific job handler:
                ├── 'opt'  → Optimization (LBFGS, RFO)
                ├── 'sp'   → SinglePoint
                ├── 'scan' → Scan
                ├── 'freq' → Frequency
                ├── 'ts'   → TransitionState (NEB, PRFO, Dimer, String, AFIR)
                └── 'irc'  → IRC (GS)
```

---

## 3. Parameter System

### 3.1 Three-Layer Architecture

```
Input File (command_control block)
        │
        ▼
CommandControl (dict)  ─────────────────────────┐
        │                                        │
        ▼                                        │
Dispatcher                                       │
        │                                        │
        ├── Set convergence thresholds           │
        │   (f_max_th, f_rms_th, dp_max_th, dp_rms_th)
        │                                        │
        ▼                                        │
Algorithm Class (e.g., NEB, LBFGS)               │
        │                                        │
        └── _init_params(ParamsClass, paras, aliases)
                    │
                    ▼
            DataClass Instance (e.g., NEBParams, LBFGSParams)
```

### 3.2 Base Class: `JobABC`

All algorithm classes inherit from `JobABC` which provides:

```python
class JobABC(ABC):
    def __init__(self, output: str)

    @abstractmethod
    def run(self)

    # Parameter utilities (case-insensitive)
    @staticmethod
    def _lower_keys(d: dict) -> dict
    @staticmethod
    def _select_subdict(paras: dict, aliases: tuple) -> dict
    @staticmethod
    def _update_dataclass_from_dict(dc_obj, d: dict)
    def _init_params(self, params_class, paras: dict, aliases: tuple)

    # Logging utilities
    def log_error(self, error_message: str)
    def log_info(self, info_message: list)
```

### 3.3 Standard Parameter Initialization

```python
class SomeAlgorithm(JobABC):
    def __init__(self, atoms, output, paras=None):
        super().__init__(output)
        self.params = self._init_params(SomeParams, paras, ("alias1", "alias2"))
```

### 3.4 Convergence Thresholds (Gaussian-style)

| Level       | f_max_th | f_rms_th | dp_max_th | dp_rms_th |
|-------------|----------|----------|-----------|-----------|
| extratight  | 0.00030  | 0.00020  | 0.00030   | 0.00020   |
| tight       | 0.00085  | 0.00055  | 0.00110   | 0.00075   |
| medium      | 0.00285  | 0.00190  | 0.00315   | 0.00210   |
| loose       | 0.00380  | 0.00250  | 0.00600   | 0.00400   |
| extraloose  | 0.00755  | 0.00500  | 0.01200   | 0.00800   |
| superloose  | 0.08500  | 0.05500  | 0.14500   | 0.09500   |

---

## 4. Job Types and Algorithms

### 4.1 Geometry Optimization (`opt`)

| Algorithm | File                            | Params Class   |
|-----------|---------------------------------|----------------|
| LBFGS     | `optimization/algorithm/LBFGS.py` | `LBFGSParams`  |
| RFO       | `optimization/algorithm/RFO.py`   | `RFOParams`    |
| SD        | `optimization/algorithm/SD.py`    | (functional)   |
| DIIS      | `optimization/algorithm/DIIS.py`  | (functional)   |

### 4.2 Transition State (`ts`)

| Algorithm | File                         | Params Class   | Description                    |
|-----------|------------------------------|----------------|--------------------------------|
| NEB       | `ts/algorithm/neb.py`        | `NEBParams`    | Nudged Elastic Band            |
| PRFO      | `ts/algorithm/PRFO.py`       | `PRFOParams`   | Partitioned RFO                |
| Dimer     | `ts/algorithm/dimer.py`      | `DimerParams`  | Dimer method                   |
| String    | `ts/algorithm/string.py`     | `GSMParams`    | Growing String Method          |
| AFIR      | `ts/algorithm/afir.py`       | `AFIRParams`   | Artificial Force Induced Reaction |
| DESCAFIR  | `ts/algorithm/descafir.py`   | `DESCAFIRParams` | Descent from AFIR              |

### 4.3 IRC (`irc`)

| Algorithm | File                      | Params Class |
|-----------|---------------------------|--------------|
| GS        | `irc/algorithm/gs.py`     | `GSParams`   |

### 4.4 Frequency (`freq`)

| Method | File                      | Params Class      |
|--------|---------------------------|-------------------|
| MW     | `frequency/frequency.py`  | `FrequencyParams` |

### 4.5 Others

- **Single Point (`sp`)**: `sp/sp.py`
- **Scan (`scan`)**: `scan/scan.py`

---

## 5. NEB Implementation Details

### 5.1 File: `ts/algorithm/neb.py`

The NEB (Nudged Elastic Band) implementation is one of the most comprehensive algorithms in MAPLE.

### 5.2 Key Components

```
neb.py
├── Utility Functions
│   ├── to_numpy_f64()          # Convert to float64 numpy
│   ├── vec1d()                 # Flatten to 1D vector
│   ├── kabsch_align()          # Rigid-body alignment
│   ├── write_xyz()             # Write XYZ trajectory
│   └── write_all_images_xyz()  # Write debug trajectory
│
├── IDPP (Image Dependent Pair Potential)
│   ├── IDPPParams              # IDPP parameters dataclass
│   ├── _pair_indices()         # Generate atom pair indices
│   └── _idpp_targets()         # Compute target distances
│
├── NEB Core
│   ├── NEBParams               # NEB parameters dataclass
│   ├── compute_dynamic_k()     # ORCA-style dynamic spring constants
│   ├── improved_tangent()      # Energy-weighted tangent
│   ├── cineb_tangent()         # Upwinding tangent for CINEB
│   ├── neb_forces()            # Compute NEB projected forces
│   └── rms_force()             # RMS force calculation
│
├── LBFGSDriver                 # L-BFGS optimizer for NEB
│   ├── two_loop()              # Two-loop recursion
│   ├── update()                # History update
│   ├── step_limit()            # Step size limiter
│   ├── should_stop()           # Convergence check
│   └── ci_should_stop()        # CINEB dual convergence
│
└── NEB Class (JobABC)
    ├── __init__()              # Initialize from Molecules
    ├── optimize_endpoints()    # Optional endpoint optimization
    ├── _pack_internal()        # Flatten images to vector
    ├── _unpack_internal()      # Restore images from vector
    ├── _align_path()           # Kabsch alignment
    ├── _compute_distances()    # Inter-image distances
    ├── _determine_insertion_plan()  # Smart interpolation planning
    ├── _insert_images_by_plan()     # Execute interpolation
    ├── _run_idpp_smoothing()   # IDPP initial path smoothing
    ├── cineb_forces()          # Climbing Image NEB forces
    ├── restart_run()           # CINEB refinement
    └── run()                   # Main execution
```

### 5.3 NEBParams Dataclass

```python
@dataclass
class NEBParams:
    n_images: int = 10              # Number of internal images
    k_min: float = 0.03             # Minimum spring constant
    k_max: float = 0.3              # Maximum spring constant
    use_dynamic_k: bool = True      # ORCA-style dynamic springs
    k_decay: float = 0.5            # Dynamic k decay factor
    max_iter: int = 500             # Maximum iterations
    lbfgs_m: int = 20               # L-BFGS memory size
    step0: float = 2e-2             # Initial step length
    ifidpp: int = 1                 # Use IDPP (1) or linear (0)
    neb_f_max_th: float = 9.5e-3    # Max force threshold
    neb_f_rms_th: float = 5e-3      # RMS force threshold
    initial_opt: bool = False       # Optimize endpoints first
    refine: str = None              # 'cineb' or 'nebts'
    cineb_f_max_th: float = 1e-2    # CINEB max force threshold
    cineb_f_rms_th: float = 1e-2    # CINEB RMS force threshold
    cilbfgs_m: int = 20             # CINEB L-BFGS memory
    cistep0: float = 5e-3           # CINEB step length
```

### 5.4 NEB Algorithm Flow

```
run()
│
├── 1. Process Input Images
│   ├── Validate image count (need >= 2)
│   ├── Align all images to reactant (Kabsch)
│   └── Determine if interpolation needed
│
├── 2. Path Interpolation (if needed)
│   ├── Compute inter-image distances
│   ├── Determine optimal insertion points (prioritize largest gaps)
│   ├── Linear interpolation
│   └── Optional IDPP smoothing (ifidpp=1)
│
├── 3. Optional Endpoint Optimization (initial_opt=True)
│
├── 4. Main NEB Optimization Loop (L-BFGS)
│   │
│   ├── For each iteration:
│   │   ├── Compute NEB forces (F_perp + F_spring)
│   │   ├── L-BFGS step direction
│   │   ├── Step size limiting
│   │   ├── Update positions
│   │   ├── Update L-BFGS history
│   │   └── Check convergence (max_f < neb_f_max_th AND rms_f < neb_f_rms_th)
│   │
│   └── Output:
│       ├── _mep.xyz (Minimum Energy Path)
│       ├── _hei.xyz (Highest Energy Image)
│       └── _image_traj.xyz (Trajectory)
│
└── 5. Optional CINEB Refinement (refine='cineb' or 'nebts')
    │
    ├── Freeze HEI index
    ├── Apply climbing image force: F_CI = F - 2(F·τ)τ
    ├── Dual convergence (regular images + climbing image)
    │
    └── Output:
        ├── _cineb_mep.xyz
        ├── _cineb_hei.xyz
        └── Optional: _nebts_ts.xyz (PRFO refined TS)
```

### 5.5 Key NEB Formulas

**Improved Tangent** (Henkelman-Jonsson):
```
if E_{i+1} > E_i > E_{i-1}:  τ = R_{i+1} - R_i
elif E_{i+1} < E_i < E_{i-1}: τ = R_i - R_{i-1}
else: τ = ΔE_+ · (R_{i+1} - R_i) + ΔE_- · (R_i - R_{i-1})
```

**NEB Projected Force**:
```
F_NEB = F_true⊥ + F_spring∥
F_true⊥ = F_true - (F_true · τ)τ
F_spring∥ = k(|R_{i+1} - R_i| - |R_i - R_{i-1}|)τ
```

**Climbing Image Force**:
```
F_CI = F_true - 2(F_true · τ)τ
```

**Dynamic Spring Constants** (ORCA-style):
```
k_i = k_max - (k_max - k_min) · exp(-k_decay · ΔE_i / ΔE_max)
```

---

## 6. ML Potentials

| Potential | Calculator Class              | Features                    |
|-----------|-------------------------------|------------------------------|
| AIMNet2   | `_aimnet2_calculator.py`      | General organic molecules    |
| ANI       | `_ani_calculator.py`          | CHNO molecules               |
| MACE      | `_mace_calculator.py`         | Universal potential          |
| UMA       | `_uma_calculator.py`          | Universal Materials          |

### Corrections
- **D4**: DFT-D4 dispersion correction
- **GBSA**: Implicit solvation (water, etc.)

---

## 7. Input File Format

```
%model aimnet2
%device cuda:0
%jobtype ts
%method neb
%xyz multi.xyz

$command_control
  level = medium
  neb
    n_images = 12
    k_max = 0.3
    refine = cineb
  end
$end
```

---

## 8. Output Files

| Suffix            | Description                           |
|-------------------|---------------------------------------|
| `_opt.xyz`        | Optimized structure                   |
| `_traj.xyz`       | Optimization trajectory               |
| `_mep.xyz`        | Minimum Energy Path (NEB)             |
| `_hei.xyz`        | Highest Energy Image (NEB)            |
| `_cineb_mep.xyz`  | CINEB refined MEP                     |
| `_cineb_hei.xyz`  | CINEB refined HEI                     |
| `_image_traj.xyz` | NEB iteration trajectory              |
| `_ts.xyz`         | Transition state structure            |
| `_irc_fwd.xyz`    | IRC forward path                      |
| `_irc_bwd.xyz`    | IRC backward path                     |
| `_freq.out`       | Frequency analysis results            |

---

## 9. Testing

```bash
# Run built-in tests
maple --test 1   # LBFGS optimization
maple --test 2   # NEB transition state
maple --test 3   # String method
maple --test 4   # Dimer method
maple --test 5   # RFO optimization
maple --test 6   # IRC GS
maple --test 7   # Frequency
maple --test 8   # PES Scan
```

---

## 10. Development Notes

### Adding a New Algorithm

1. Create params dataclass:
```python
@dataclass
class NewAlgorithmParams:
    param1: float = 1.0
    param2: int = 10
```

2. Create algorithm class inheriting from `JobABC`:
```python
class NewAlgorithm(JobABC):
    def __init__(self, atoms, output, paras=None):
        super().__init__(output)
        self.atoms = atoms
        self.params = self._init_params(NewAlgorithmParams, paras, ("newalg", "NewAlg"))

    def run(self):
        # Implementation
        pass
```

3. Register in dispatcher (e.g., `ts/ts.py` or `optimization/optimization.py`)

### Parameter Naming Convention
- Use lowercase with underscores: `max_iter`, `f_max_th`
- Convergence thresholds end with `_th`: `f_max_th`, `f_rms_th`, `dp_max_th`, `dp_rms_th`
- Use `verbose` (not `verbosity`) for output control

---

*Document generated: 2026-01-16*

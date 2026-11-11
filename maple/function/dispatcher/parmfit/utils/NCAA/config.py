"""Usage: parse user-facing NCAA input options."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..chgfit import ChargeFitConfig, build_charge_fit_config
from ..QMInterface import QMReferenceConfig, build_qm_reference_config
from ..TorsionFit import TorsionFitParams, build_torsion_fit_params
from ..TorsionFit.config import _parse_torsion_bonds
from ...runconfig import as_tracked, read_batch_size


SUPPORTED_PRO_FF = ("ff14SB", "ff19SB")
SUPPORTED_WATER_MODELS = ("tip3p", "spce", "tip4pew", "opc3", "opc", "fb3", "fb4")
SUPPORTED_BONDED_METHODS = ("mseminario", "seminario", "none")


@dataclass(frozen=True)
class NCAAAbinitioConfig:
    pdb_path: str
    target: str
    res_charge: int
    res_spin_multi: int
    charge_fit: ChargeFitConfig
    qm: QMReferenceConfig = field(default_factory=QMReferenceConfig)
    rn: str = "MOL"
    vib_scale: float = 1.0
    opt_max_iter: int = 256
    opt_max_step: float = 0.2
    torsion: TorsionFitParams = field(default_factory=TorsionFitParams)
    wat_ff: str = "tip3p"
    pro_ff: str = "ff14SB"
    bonded: str = "mseminario"
    ncaa_resids: tuple[str, ...] = ()
    ncaa_charges: tuple[int, ...] = ()
    ncaa_resnames: tuple[str, ...] = ()
    ncaa_mults: tuple[int, ...] = ()
    torsion_bonds_per_residue: tuple[tuple[tuple[int, int], ...], ...] | None = None
    batch_size: int | None = None


def _split_entries(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(token).strip() for token in value if str(token).strip()]
    return [token.strip() for token in str(value).split(",") if token.strip()]


def parse_ncaa_abinitio_config(
    raw: dict | None,
    *,
    pdb_path: str,
) -> NCAAAbinitioConfig:

    raw = as_tracked(raw)
    raw.set_group("NCAA run", "Non-standard amino acid declaration and residue building")
    ncaa_resids = _split_entries(raw.get("ncaa_resids", ""))
    if not ncaa_resids:
        raise ValueError(
            "NCAA requires ncaa_resids=... (one residue selector per entry, comma-separated)."
        )
    ncaa_charges = [int(token) for token in _split_entries(raw.get("ncaa_charges", ""))]
    if not ncaa_charges:
        ncaa_charges = [0 for _ in ncaa_resids]
    if len(ncaa_charges) != len(ncaa_resids):
        raise ValueError(
            f"ncaa_charges ({len(ncaa_charges)} entries) must pair 1:1 with ncaa_resids ({len(ncaa_resids)} entries)."
        )
    ncaa_resnames = [token.upper() for token in _split_entries(raw.get("ncaa_resnames", ""))]
    if ncaa_resnames and len(ncaa_resnames) != len(ncaa_resids):
        raise ValueError(
            f"ncaa_resnames ({len(ncaa_resnames)} entries) must pair 1:1 with ncaa_resids ({len(ncaa_resids)} entries)."
        )
    ncaa_mults = [int(token) for token in _split_entries(raw.get("ncaa_mults", ""))]
    if ncaa_mults and len(ncaa_mults) != len(ncaa_resids):
        raise ValueError(
            f"ncaa_mults ({len(ncaa_mults)} entries) must pair 1:1 with ncaa_resids ({len(ncaa_resids)} entries)."
        )

    raw_pro_ff = str(raw.get("pro_ff", "ff14SB")).strip()
    pro_ff = next((item for item in SUPPORTED_PRO_FF if item.lower() == raw_pro_ff.lower()), None)
    if pro_ff is None:
        raise ValueError(f"Unsupported protein ff {raw_pro_ff!r}; expected one of {', '.join(SUPPORTED_PRO_FF)}.")

    wat_ff = str(raw.get("wat_ff", "tip3p")).strip().lower()
    if wat_ff not in SUPPORTED_WATER_MODELS:
        raise ValueError(f"Unsupported water ff {wat_ff!r}; expected one of {', '.join(SUPPORTED_WATER_MODELS)}.")

    bonded = str(raw.get("bonded", "mseminario")).strip().lower()
    if bonded not in SUPPORTED_BONDED_METHODS:
        raise ValueError(
            f"Unsupported bonded method {bonded!r}; expected one of {', '.join(SUPPORTED_BONDED_METHODS)}."
        )
    vib_scale = float(raw.get("vib_scale", 1.0))

    # Per-residue torsion bond groups (; separated, matching ncaa_resids order).
    # Indices are residue-local 1-based serial-order (ACE/NME caps excluded);
    # converted to capped-model indices in _refine_ncaa_parameters.
    # Must parse BEFORE build_torsion_fit_params (which chokes on ;).
    raw_tb = str(raw.get("torsion_bonds", "")).strip()
    torsion_bonds_per_residue = None
    if raw_tb and ";" in raw_tb:
        torsion_bonds_per_residue = tuple(
            tuple(_parse_torsion_bonds(g) or ()) for g in raw_tb.split(";")
        )
        # Rewrite the dict entry (not tracked replace — get always reads dict)
        # so build_torsion_fit_params parses only the first group.
        raw["torsion_bonds"] = raw_tb.split(";")[0]
    elif raw_tb:
        parsed = _parse_torsion_bonds(raw_tb)
        torsion_bonds_per_residue = (parsed,) if parsed else None

    torsion = build_torsion_fit_params(raw)
    torsion.torsion_ensemble = False
    torsion._refresh_derived()
    raw.replace("torsion_ensemble", False)

    qm = build_qm_reference_config(raw)
    raw.set_group("charge fitting", "Charge fitting method, level and RESP backend")
    charge_fit = build_charge_fit_config(
        raw,
        default_method="resp",
        default_level="HF/6-31G(d)",
        default_nproc=qm.qm_nproc,
        default_mem=qm.qm_mem,
    )
    if charge_fit.method == "none":
        raise ValueError(
            "NCAA does not support chg_fit=none because its PDB input has no atomic charges."
        )

    raw.set_group("MLIP optimization", "MLIP geometry optimization controls")
    opt_max_iter = int(raw.get("opt_max_iter", 256))
    opt_max_step = float(raw.get("opt_max_step", 0.2))

    return NCAAAbinitioConfig(
        pdb_path=pdb_path,
        target=" ".join(ncaa_resids),
        res_charge=ncaa_charges[0],
        res_spin_multi=ncaa_mults[0] if ncaa_mults else 1,
        charge_fit=charge_fit,
        qm=qm,
        rn=ncaa_resnames[0] if ncaa_resnames else "MOL",
        vib_scale=vib_scale,
        batch_size=read_batch_size(raw),
        opt_max_iter=opt_max_iter,
        opt_max_step=opt_max_step,
        torsion=torsion,
        wat_ff=wat_ff,
        pro_ff=pro_ff,
        bonded=bonded,
        ncaa_resids=tuple(ncaa_resids),
        ncaa_charges=tuple(ncaa_charges),
        ncaa_resnames=tuple(ncaa_resnames),
        ncaa_mults=tuple(ncaa_mults),
        torsion_bonds_per_residue=torsion_bonds_per_residue,
    )

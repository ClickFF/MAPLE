"""Usage: write Amber and GROMACS parameter files from parmfit parameters."""

from __future__ import annotations

from copy import deepcopy
import os
from math import degrees

import numpy as np
from ase import Atoms

from .mechanics import build_mm_topology_cache
from .mol2_tools import write_updated_mol2
from .readparm import RefinementParameterSet


_KCAL_TO_KJ = 4.184
_ANG_TO_NM = 0.1
_BOX_PADDING_NM = 0.5
_MIN_BOX_LENGTH_NM = 0.5
_SIGMA_DENOM = 2.0 ** (1.0 / 6.0)
# Letters that are not element symbols come first. teLeap and ParmEd both read an
# unfamiliar type's leading character as an element symbol, so "U0" is taken for
# uranium and "N3" for nitrogen. The element letters stay in the pool as overflow
# because the two-character type space is too small to give any of them up.
_MAPLE_TYPE_LETTERS = (
    "Z", "J", "L", "M", "Q", "R", "T", "X", "E",
    "U", "V", "I", "K", "N", "P", "S", "W", "Y", "B",
)
_MAPLE_TYPE_SUFFIXES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ATOMIC_FALLBACK = {
    "H": (1, 1.008),
    "B": (5, 10.81),
    "C": (6, 12.01),
    "N": (7, 14.01),
    "O": (8, 16.00),
    "F": (9, 19.00),
    "P": (15, 30.97),
    "S": (16, 32.06),
    "CL": (17, 35.45),
    "BR": (35, 79.904),
    "I": (53, 126.90),
}
_FF_ATOM_TYPES = {
    # ==========================================
    # (Amino Acid - ff14/19SB & GAFF2 Custom)
    # ==========================================
    
    # (Hydrogen)
    "H": "sp3", "H1": "sp3", "H2": "sp3", "H3": "sp3", "H4": "sp3", "H5": "sp3", 
    "HA": "sp3", "HC": "sp3", "HO": "sp3", "HP": "sp3", "HS": "sp3", "HW": "sp3", "HZ": "sp3",
    "h1": "sp3", "h2": "sp3", "h3": "sp3", "h4": "sp3", "h5": "sp3", "ha": "sp3", 
    "hc": "sp3", "hn": "sp3", "ho": "sp3", "hp": "sp3", "hs": "sp3", "hw": "sp3", "hx": "sp3",
    # (Carbon)
    "C": "sp2", "C*": "sp2", "C0": "sp3", "2C": "sp3", "3C": "sp3", "C4": "sp2", "C5": "sp2", "C8": "sp3",
    "CA": "sp2", "CB": "sp2", "CC": "sp2", "CD": "sp2", "CH": "sp3", "CI": "sp3", "CJ": "sp2", 
    "CK": "sp2", "CM": "sp2", "CN": "sp2", "CO": "sp2", "CP": "sp2", "CQ": "sp2", "CR": "sp2", 
    "CS": "sp2", "CT": "sp3", "CV": "sp2", "CW": "sp2", "CX": "sp3", "CY": "sp2", "CZ": "sp2", "XC": "sp3",
    "c": "sp2", "c1": "sp2", "c2": "sp2", "c3": "sp3", "c5": "sp3", "c6": "sp3",
    "ca": "sp2", "cc": "sp2", "cd": "sp2", "ce": "sp2", "cf": "sp2", "cg": "sp2", 
    "ch": "sp2", "cp": "sp2", "cq": "sp2", "cs": "sp2", "cu": "sp2", "cv": "sp2", 
    "cx": "sp2", "cy": "sp2", "cz": "sp2",
    # (Nitrogen)
    "N": "sp2", "N*": "sp2", "N2": "sp2", "N3": "sp3", "NA": "sp2", "NB": "sp2", 
    "NC": "sp2", "NP": "sp2", "NQ": "sp2", "NT": "sp3", "NY": "sp2",
    "n": "sp2", "n+": "sp3", "n1": "sp2", "n2": "sp2", "n3": "sp3", "n4": "sp3", "n5": "sp3", 
    "n6": "sp3", "n7": "sp3", "n8": "sp3", "n9": "sp3", "na": "sp2", "nb": "sp2", "nc": "sp2", 
    "nd": "sp2", "ne": "sp2", "nf": "sp2", "nh": "sp2", "ni": "sp2", "nj": "sp2", "nk": "sp3", 
    "nl": "sp3", "nm": "sp2", "nn": "sp2", "no": "sp2", "np": "sp3", "nq": "sp3", "ns": "sp2", 
    "nt": "sp2", "nu": "sp2", "nv": "sp2", "nx": "sp3", "ny": "sp3", "nz": "sp3",
    # (Oxygen)
    "O": "sp2", "O2": "sp2", "OH": "sp3", "OP": "sp2", "OS": "sp3", "OW": "sp3",
    "o": "sp2", "o2": "sp2", "oh": "sp3", "op": "sp3", "oq": "sp3", "os": "sp3", "ow": "sp3",
    # (Sulfur & Phosphorus)
    "S": "sp3", "SH": "sp3", 
    "s": "sp2", "s2": "sp2", "s3": "sp3", "s4": "sp3", "s6": "sp3", "sh": "sp3", 
    "sp": "sp3", "sq": "sp3", "ss": "sp3", "sx": "sp3", "sy": "sp3",
    "P": "sp3", "LP": "sp3", "EP": "sp3", 
    "p2": "sp2", "p3": "sp3", "p4": "sp3", "p5": "sp3", "pb": "sp3", "pc": "sp3", "pd": "sp3", 
    "pe": "sp3", "pf": "sp3", "px": "sp3", "py": "sp3", 
    # (Halogens)
    "F": "sp3", "Cl": "sp3", "Br": "sp3", "I": "sp3",
    "f": "sp3", "cl": "sp3", "br": "sp3", "i": "sp3",
    # (Ions) Amber spells ions "Zn2+"/"Na+"/"C0", never as a bare element symbol.
    # Bare symbols collide with the organic types above -- "CA"/"NA"/"CO" are the
    # ff14SB ring carbon and nitrogens and "ca"/"cu"/"ni" the gaff2 aromatic and
    # conjugated ones -- and a repeated key in a dict literal silently keeps the
    # last value, so spelling them that way retyped seven aromatic types as sp3.
    "MG": "sp3", "CU": "sp3", "FE": "sp3", "Zn": "sp3",
    "Li+": "sp3", "Na+": "sp3", "K+": "sp3", "Rb+": "sp3", "Cs+": "sp3",
    "F-": "sp3", "Cl-": "sp3", "Br-": "sp3", "I-": "sp3",
    "Be2+": "sp3", "Cu2+": "sp3", "Ni2+": "sp3", "Pt2+": "sp3", "Zn2+": "sp3",
    "Co2+": "sp3", "Pd2+": "sp3", "Ag2+": "sp3", "Cr2+": "sp3", "Fe2+": "sp3",
    "Mg2+": "sp3", "V2+": "sp3", "Mn2+": "sp3", "Hg2+": "sp3", "Cd2+": "sp3",
    "Yb2+": "sp3", "Ca2+": "sp3", "Sn2+": "sp3", "Pb2+": "sp3", "Eu2+": "sp3",
    "Sr2+": "sp3", "Sm2+": "sp3", "Ba2+": "sp3", "Ra2+": "sp3",
    "Al3+": "sp3", "Ce3+": "sp3", "Cr3+": "sp3", "Dy3+": "sp3", "Er3+": "sp3",
    "Eu3+": "sp3", "Fe3+": "sp3", "Gd3+": "sp3", "In3+": "sp3", "La3+": "sp3",
    "Lu3+": "sp3", "Nd3+": "sp3", "Pr3+": "sp3", "Sm3+": "sp3", "Tb3+": "sp3",
    "Tl3+": "sp3", "Tm3+": "sp3", "Y3+": "sp3",
    "Ce4+": "sp3", "Hf4+": "sp3", "Pu4+": "sp3", "Th4+": "sp3", "U4+": "sp3", "Zr4+": "sp3",
}

# Snapshot of two-character uppercase/alnum Amber atom types observed in the
# bundled leap parm/lib/prep data. MAPLE generated types must avoid these so
# NCAA/protein crossterms such as CX/XC keep their force-field meaning.
_AMBER_RESERVED_TWO_CHAR_TYPES = frozenset(
    """
    2C 3C 6C 6D 6E 6F 6H 6I 6J 6K 6L 6M 6N 6P 6Q 6R 6S 6T 6V 6W 6Y 8C A1 A3
    A5 A6 A7 A9 AA AB AC AE AG AL AN AP AR AS B1 B2 BA BB BE BG BP BR BS BT
    BX BY C0 C1 C2 C3 C4 C5 C6 C7 C8 C9 CA CB CC CD CE CF CG CH CI CJ CK CL
    CM CN CO CP CQ CR CS CT CU CV CW CX CY CZ D1 D2 D3 D4 DA DC DE DG DR DS
    DT DU EC ED EP EU F1 F2 F3 F4 F5 F6 FE FZ G1 G2 G3 G5 G6 G7 G9 GA GC GD
    GE GL GN GT H0 H1 H2 H3 H4 H5 H6 H7 H8 H9 HA HB HC HD HE HF HG HH HK HM
    HN HO HP HR HS HT HW HX HZ IB ID II IM IN IP LA LI LJ LP LU M0 M1 M2 M4
    MC MD MG MN MW MY N1 N2 N3 N4 N5 N6 N7 N9 NA NB NC ND NE NF NG NH NI NL
    NO NP NR NT NY NZ O1 O2 O3 O4 O5 O6 O7 O8 O9 OA OB OD OE OF OG OH OK OL
    OM ON OP OQ OR OS OT OV OW OX OZ P2 PA PB PC PD PE PH PI PR PS PT Q1 Q2
    Q3 Q4 QC QK QL QM QN QR R0 R1 R2 R3 R4 RA RB RC RE RG RO RU S1 S2 S3 S4
    SA SC SD SE SF SG SH SM SO SP SR SS ST SX T3 T5 TA TB TC TG TH TJ TL TM
    TN TO TP U1 U2 U3 U4 U5 UD UE UN V2 VS XC Y1 Y3 YC ZN
    """.split()
) | frozenset(
    atom_type
    for atom_type in _FF_ATOM_TYPES
    if len(atom_type) == 2 and atom_type.isupper() and atom_type.isalnum()
)


def _section(title: str, body: list[str]) -> list[str]:
    return [f"[ {title} ]\n"] + body + ["\n"]


def _comment(line: str) -> str:
    return f"; {line}\n"


def _normalize_symbol(symbol: str) -> str:
    if len(symbol) <= 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def _atomic_number_and_mass(symbol: str) -> tuple[int, float]:
    normalized = _normalize_symbol(symbol)
    try:
        from ase.data import atomic_masses, atomic_numbers  # type: ignore

        atomic_number = int(atomic_numbers[normalized])
        return atomic_number, float(atomic_masses[atomic_number])
    except Exception:
        key = normalized.upper()
        if key not in _ATOMIC_FALLBACK:
            raise ValueError(f"GROMACS export does not know atomic number/mass for element '{symbol}'.")
        return _ATOMIC_FALLBACK[key]


def _box_lengths_nm(atoms: Atoms) -> tuple[float, float, float]:
    positions_nm = np.asarray(atoms.get_positions(), dtype=float) * _ANG_TO_NM
    mins = np.min(positions_nm, axis=0)
    maxs = np.max(positions_nm, axis=0)
    lengths = (maxs - mins) + _BOX_PADDING_NM
    lengths = np.maximum(lengths, _MIN_BOX_LENGTH_NM)
    return float(lengths[0]), float(lengths[1]), float(lengths[2])


def _title_from_base(output_base: str, title: str | None) -> str:
    if title:
        return str(title)
    return os.path.basename(os.path.splitext(output_base)[0]) or "MAPLE parmfit export"


def _format_defaults_lines() -> list[str]:
    return [
        _comment("nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ"),
        "1               2               yes             0.5     0.83333333\n",
    ]


def _format_atomtypes_lines(paramset: RefinementParameterSet, atoms: Atoms, meta: dict) -> list[str]:
    lines = [_comment("name    at.num    mass    charge ptype  sigma      epsilon")]
    symbols = atoms.get_chemical_symbols()
    seen: set[str] = set()
    for nonbond in paramset.nonbonds:
        if nonbond.atom_type in seen:
            continue
        seen.add(nonbond.atom_type)
        atom_index = nonbond.atom - 1
        atomic_number, mass = _atomic_number_and_mass(symbols[atom_index])
        if nonbond.rmin_half is None or nonbond.epsilon is None:
            meta["omitted_counts"]["atomtypes"] += 1
            warning = f"atom type {nonbond.atom_type} omitted from [ atomtypes ] because LJ parameters are missing"
            meta["warnings"].append(warning)
            lines.append(_comment(warning))
            continue
        sigma_nm = (2.0 * float(nonbond.rmin_half) / _SIGMA_DENOM) * _ANG_TO_NM
        epsilon_kj = float(nonbond.epsilon) * _KCAL_TO_KJ
        lines.append(
            f"{nonbond.atom_type:<12} {atomic_number:>5d}  {mass:>10.6f}  0.00000000  A  "
            f"{sigma_nm:>12.8f}  {epsilon_kj:>12.7f}\n"
        )
    return lines


def _format_atoms_lines(paramset: RefinementParameterSet, atoms: Atoms) -> list[str]:
    lines = [
        _comment("nr       type  resnr residue  atom   cgnr    charge       mass"),
        _comment("residue    1 MOL rtp MOL q 0.0"),
    ]
    symbols = atoms.get_chemical_symbols()
    for index, (mol2_atom, nonbond) in enumerate(zip(paramset.mol2.atoms, paramset.nonbonds), start=1):
        _, mass = _atomic_number_and_mass(symbols[index - 1])
        lines.append(
            f"{index:>5d}  {nonbond.atom_type:>9s}  {1:>5d}  {'MOL':>6s}  {mol2_atom.name:>5s}  {index:>5d}"
            f"  {nonbond.charge:>10.8f}  {mass:>10.6f}\n"
        )
    return lines


def _format_bonds_lines(paramset: RefinementParameterSet, meta: dict) -> list[str]:
    lines = [_comment("ai     aj funct         c0         c1")]
    for bond in paramset.bonds:
        if bond.kBond is None or bond.rEq is None:
            meta["omitted_counts"]["bonds"] += 1
            warning = f"bond {bond.atoms} omitted from [ bonds ] because kBond/rEq is missing"
            meta["warnings"].append(warning)
            lines.append(_comment(warning))
            continue
        r0_nm = float(bond.rEq) * _ANG_TO_NM
        k_gmx = 2.0 * float(bond.kBond) * _KCAL_TO_KJ * 100.0
        lines.append(f"{bond.atoms[0]:>6d} {bond.atoms[1]:>7d} {1:>6d} {r0_nm:>12.5f} {k_gmx:>14.6f}\n")
    return lines


def _format_pairs_lines(paramset: RefinementParameterSet) -> list[str]:
    cache = build_mm_topology_cache(paramset)
    lines = [_comment("ai     aj funct")]
    for atom_i, atom_j in sorted(cache.scaled_14):
        lines.append(f"{atom_i:>6d} {atom_j:>7d} {1:>6d}\n")
    return lines


def _format_angles_lines(paramset: RefinementParameterSet, meta: dict) -> list[str]:
    lines = [_comment("ai     aj     ak funct         c0         c1")]
    for angle in paramset.angles:
        if angle.kTheta is None or angle.thetaEq is None:
            meta["omitted_counts"]["angles"] += 1
            warning = f"angle {angle.atoms} omitted from [ angles ] because kTheta/thetaEq is missing"
            meta["warnings"].append(warning)
            lines.append(_comment(warning))
            continue
        theta_deg = degrees(float(angle.thetaEq))
        k_gmx = 2.0 * float(angle.kTheta) * _KCAL_TO_KJ
        lines.append(
            f"{angle.atoms[0]:>6d} {angle.atoms[1]:>7d} {angle.atoms[2]:>7d} {1:>6d} "
            f"{theta_deg:>12.6f} {k_gmx:>14.6f}\n"
        )
    return lines


def _period_to_mult(period: float, label: str, meta: dict) -> int:
    rounded = int(round(float(period)))
    if abs(float(period) - rounded) > 1.0e-8:
        warning = f"{label} period {period} was rounded to integer multiplicity {rounded} for GROMACS export"
        meta["warnings"].append(warning)
    return rounded


def _format_dihedrals_lines(paramset: RefinementParameterSet, meta: dict) -> list[str]:
    lines = [_comment("ai     aj     ak     al funct         c0         c1    mult")]
    for dihedral in paramset.dihedrals:
        if not dihedral.terms:
            meta["omitted_counts"]["dihedrals"] += 1
            warning = f"proper dihedral {dihedral.atoms} omitted from [ dihedrals ] because no torsion terms are assigned"
            meta["warnings"].append(warning)
            lines.append(_comment(warning))
            continue
        for term in dihedral.terms:
            lines.append(
                f"{dihedral.atoms[0]:>6d} {dihedral.atoms[1]:>7d} {dihedral.atoms[2]:>7d} {dihedral.atoms[3]:>7d} {1:>6d} "
                f"{degrees(float(term.phase)):>12.7f} {(float(term.kPhi) * _KCAL_TO_KJ):>12.7f} "
                f"{_period_to_mult(term.period, f'proper dihedral {dihedral.atoms}', meta):>5d}\n"
            )
    for improper in paramset.impropers:
        if not improper.terms:
            meta["omitted_counts"]["impropers"] += 1
            continue
        for term in improper.terms:
            lines.append(
                f"{improper.atoms[0]:>6d} {improper.atoms[1]:>7d} {improper.atoms[2]:>7d} {improper.atoms[3]:>7d} {4:>6d} "
                f"{degrees(float(term.phase)):>12.7f} {(float(term.kPhi) * _KCAL_TO_KJ):>12.7f} "
                f"{_period_to_mult(term.period, f'improper {improper.atoms}', meta):>5d}\n"
            )
    return lines


def _format_system_lines(title: str) -> list[str]:
    return [_comment("Name"), f"{title}\n"]


def _format_molecules_lines() -> list[str]:
    return [_comment("Compound       #mols"), f"{'MOL':<16}{1:>5d}\n"]


def allocate_maple_atom_types(count: int, existing_types: set[str]) -> dict[int, str]:
    maple_types: dict[int, str] = {}
    blocked_types = set(existing_types) | set(_AMBER_RESERVED_TWO_CHAR_TYPES)
    next_index = 0
    for atom_index in range(1, count + 1):
        while next_index < len(_MAPLE_TYPE_LETTERS) * len(_MAPLE_TYPE_SUFFIXES):
            candidate = (
                f"{_MAPLE_TYPE_LETTERS[next_index // len(_MAPLE_TYPE_SUFFIXES)]}"
                f"{_MAPLE_TYPE_SUFFIXES[next_index % len(_MAPLE_TYPE_SUFFIXES)]}"
            )
            next_index += 1
            if candidate not in blocked_types:
                maple_types[atom_index] = candidate
                break
        else:
            raise ValueError("MAPLE Amber export ran out of unique two-character atom types.")
    return maple_types


def _element_from_mass(mass: float) -> str:
    """Recover an element symbol from an atom type's mass.

    The refinement mol2 atoms carry no element column, so the frcmod MASS section --
    which the export already validates for every type -- is the only per-type
    element evidence available.
    """
    try:
        from ase.data import atomic_masses, chemical_symbols  # type: ignore

        table = [
            (float(reference), chemical_symbols[number])
            for number, reference in enumerate(atomic_masses)
            if number and float(reference) == float(reference)
        ]
    except Exception:
        table = [(reference, symbol) for symbol, (_number, reference) in _ATOMIC_FALLBACK.items()]
    return _normalize_symbol(min(table, key=lambda item: abs(item[0] - mass))[1])


def format_tleap_add_atom_types(
    atom_type_rows: list[tuple[str, str, str, str]],
) -> list[str]:
    lines = ["addAtomTypes {\n"]
    for atom_name, element, old_type, maple_type in atom_type_rows:
        hybridization = _FF_ATOM_TYPES.get(old_type)
        if hybridization is None:
            raise ValueError(f"Could not determine tleap hybridization for atom {atom_name} with old atom type {old_type!r}.")
        lines.append(f'    {{ "{maple_type}" "{_normalize_symbol(element)}" "{hybridization}" }}\n')
    lines.append("}\n")
    return lines


def format_refinement_tleap(
    atom_type_rows: list[tuple[str, str, str, str]],
    *,
    mol2_name: str,
    frcmod_name: str,
    prmtop_name: str,
    inpcrd_name: str,
    unit: str = "lig",
) -> list[str]:
    """Gas-phase tleap script for a refinement export.

    addAtomTypes must precede loadamberparams and loadmol2: teLeap applies the
    table only to types it has not resolved yet, and a block placed after them is
    ignored without an error -- every MAPLE type then lands in the topology with
    ATOMIC_NUMBER -1, or with whatever element its leading letter suggests.
    """
    lines = ["source leaprc.gaff2\n"]
    if atom_type_rows:
        lines.extend(format_tleap_add_atom_types(atom_type_rows))
    lines.append(f"loadamberparams {frcmod_name}\n")
    lines.append(f"{unit} = loadmol2 {mol2_name}\n")
    lines.append(f"saveamberparm {unit} {prmtop_name} {inpcrd_name}\n")
    lines.append("quit\n")
    return lines


def write_top(
    paramset: RefinementParameterSet,
    atoms: Atoms,
    top_path: str,
    title: str | None = None,
) -> dict:
    meta = {
        "warnings": [],
        "omitted_counts": {
            "atomtypes": 0,
            "bonds": 0,
            "angles": 0,
            "dihedrals": 0,
            "impropers": 0,
            "nonbonds": 0,
        },
        "written_sections": [
            "defaults",
            "atomtypes",
            "moleculetype",
            "atoms",
            "bonds",
            "pairs",
            "angles",
            "dihedrals",
            "system",
            "molecules",
        ],
    }
    title_str = _title_from_base(top_path, title)
    lines = [
        _comment(f"File {os.path.basename(top_path)} was generated by MAPLE parmfit"),
        _comment("This is a standalone topology file"),
    ]
    lines.extend(_section("defaults", _format_defaults_lines()))
    lines.extend(_section("atomtypes", _format_atomtypes_lines(paramset, atoms, meta)))
    lines.extend(_section("moleculetype", [_comment("Name            nrexcl"), f"{'MOL':<12}{3:>5d}\n"]))
    lines.extend(_section("atoms", _format_atoms_lines(paramset, atoms)))
    lines.extend(_section("bonds", _format_bonds_lines(paramset, meta)))
    lines.extend(_section("pairs", _format_pairs_lines(paramset)))
    lines.extend(_section("angles", _format_angles_lines(paramset, meta)))
    lines.extend(_section("dihedrals", _format_dihedrals_lines(paramset, meta)))
    lines.extend(_section("system", _format_system_lines(title_str)))
    lines.extend(_section("molecules", _format_molecules_lines()))

    if meta["warnings"]:
        warning_header = [_comment("Warnings during GROMACS export:")]
        warning_header.extend(_comment(warning) for warning in meta["warnings"])
        lines = warning_header + ["\n"] + lines

    with open(top_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    return meta


def write_gro(
    paramset: RefinementParameterSet,
    atoms: Atoms,
    gro_path: str,
    title: str | None = None,
) -> str:
    title_str = _title_from_base(gro_path, title)
    positions_nm = np.asarray(atoms.get_positions(), dtype=float) * _ANG_TO_NM
    box_x, box_y, box_z = _box_lengths_nm(atoms)

    with open(gro_path, "w", encoding="utf-8") as handle:
        handle.write(f"{title_str}\n")
        handle.write(f"{len(atoms):5d}\n")
        for index, (mol2_atom, xyz_nm) in enumerate(zip(paramset.mol2.atoms, positions_nm), start=1):
            atom_name = mol2_atom.name[-5:]
            handle.write(
                f"{1:5d}{'MOL':<5}{atom_name:>5}{index:5d}{xyz_nm[0]:8.3f}{xyz_nm[1]:8.3f}{xyz_nm[2]:8.3f}\n"
            )
        handle.write(f"{box_x:10.5f}{box_y:10.5f}{box_z:10.5f}\n")
    return gro_path


def write_gromacs_files(
    paramset: RefinementParameterSet,
    atoms: Atoms,
    output_base: str,
    title: str | None = None,
) -> tuple[str, str, dict]:
    base = os.path.splitext(output_base)[0]
    top_path = base + "_maple.top"
    gro_path = base + "_maple.gro"
    meta = write_top(paramset, atoms, top_path, title=title)
    write_gro(paramset, atoms, gro_path, title=title)
    return top_path, gro_path, meta


def write_amber_files(
    paramset: RefinementParameterSet,
    atoms: Atoms,
    input_mol2_path: str,
    output_base: str,
) -> tuple[str, str, str]:
    base = os.path.splitext(output_base)[0]
    mol2_path = base + "_maple.mol2"
    frcmod_path = base + "_maple.frcmod"
    tleap_path = base + "_maple_tleap.in"

    _validate_amber_export_inputs(paramset)

    existing_types = {atom.atom_type for atom in paramset.mol2.atoms}
    maple_types = allocate_maple_atom_types(len(paramset.mol2.atoms), existing_types)

    charges_by_atom = {
        nonbond.atom: float(nonbond.charge)
        for nonbond in paramset.nonbonds
    }
    write_updated_mol2(
        input_mol2_path,
        mol2_path,
        positions=atoms.get_positions(),
        atom_types=[
            maple_types[index]
            for index in range(1, len(paramset.mol2.atoms) + 1)
        ],
        charges=[
            charges_by_atom[index]
            for index in range(1, len(paramset.mol2.atoms) + 1)
        ],
    )

    mapped = deepcopy(paramset)
    maple_mass_params: dict[str, float] = {}
    for bond in mapped.bonds:
        bond.atom_types = tuple(maple_types[index] for index in bond.atoms)
    for angle in mapped.angles:
        angle.atom_types = tuple(maple_types[index] for index in angle.atoms)
    for dihedral in mapped.dihedrals:
        dihedral.atom_types = tuple(maple_types[index] for index in dihedral.atoms)
    for improper in mapped.impropers:
        improper.atom_types = tuple(maple_types[index] for index in improper.atoms)
    for nonbond in mapped.nonbonds:
        nonbond.atom_type = maple_types[nonbond.atom]
    for atom_index, mol2_atom in enumerate(paramset.mol2.atoms, start=1):
        maple_mass_params[maple_types[atom_index]] = paramset.frcmod.mass_params[mol2_atom.atom_type]

    from .interface import write_refined_frcmod

    write_refined_frcmod(mapped, frcmod_path, mass_params=maple_mass_params)

    gas_base = os.path.basename(base) + "_maple_gas"
    atom_type_rows = [
        (
            mol2_atom.name,
            _element_from_mass(paramset.frcmod.mass_params[mol2_atom.atom_type]),
            mol2_atom.atom_type,
            maple_types[atom_index],
        )
        for atom_index, mol2_atom in enumerate(paramset.mol2.atoms, start=1)
    ]
    with open(tleap_path, "w") as handle:
        handle.writelines(
            format_refinement_tleap(
                atom_type_rows,
                mol2_name=os.path.basename(mol2_path),
                frcmod_name=os.path.basename(frcmod_path),
                prmtop_name=gas_base + ".prmtop",
                inpcrd_name=gas_base + ".inpcrd",
            )
        )
    return mol2_path, frcmod_path, tleap_path


def _validate_amber_export_inputs(paramset: RefinementParameterSet) -> None:
    atom_count = len(paramset.mol2.atoms)
    expected_atoms = set(range(1, atom_count + 1))
    nonbond_atoms = {nonbond.atom for nonbond in paramset.nonbonds}
    missing_nonbond_atoms = sorted(expected_atoms - nonbond_atoms)
    if missing_nonbond_atoms:
        missing = ", ".join(str(atom) for atom in missing_nonbond_atoms)
        raise ValueError(f"Cannot write Amber NONBON: missing nonbond rows for atom(s) {missing}.")

    for atom in paramset.mol2.atoms:
        if atom.atom_type not in paramset.frcmod.mass_params:
            raise ValueError(f"Cannot write Amber MASS for atom type {atom.atom_type!r}.")

    for nonbond in paramset.nonbonds:
        if nonbond.atom not in expected_atoms:
            raise ValueError(f"Cannot write Amber NONBON for atom {nonbond.atom}: atom index is out of range.")
        if nonbond.rmin_half is None or nonbond.epsilon is None:
            raise ValueError(
                f"Cannot write Amber NONBON for atom {nonbond.atom} ({nonbond.atom_type}): "
                "missing rmin_half/epsilon."
            )

    for bond in paramset.bonds:
        if bond.kBond is None or bond.rEq is None:
            raise ValueError(f"Cannot write Amber BOND {'-'.join(map(str, bond.atoms))}: missing kBond/rEq.")
    for angle in paramset.angles:
        if angle.kTheta is None or angle.thetaEq is None:
            raise ValueError(f"Cannot write Amber ANGLE {'-'.join(map(str, angle.atoms))}: missing kTheta/thetaEq.")
    for dihedral in paramset.dihedrals:
        if not dihedral.terms:
            raise ValueError(f"Cannot write Amber DIHE {'-'.join(map(str, dihedral.atoms))}: missing torsion terms.")
    for improper in paramset.impropers:
        if not improper.terms:
            raise ValueError(f"Cannot write Amber IMPROPER {'-'.join(map(str, improper.atoms))}: missing torsion terms.")

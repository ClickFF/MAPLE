"""
GROMACS-style MDP file parser for MAPLE MD parameters.

Format:
    key = value   ; optional comment
    key = value   # optional comment

Keys are case-insensitive and stripped of whitespace.
Values are coerced to int, float, or bool as appropriate;
otherwise left as str.
"""

import os


def parse_mdp(path: str) -> dict:
    """
    Parse a GROMACS-style MDP file and return a dict of parameters.

    Handles:
    - Comments introduced by ';' or '#'
    - key = value pairs (case-insensitive keys, lowercased in output)
    - Type coercion: int, float, bool ('yes'/'no'/'true'/'false'), str
    - Blank lines and comment-only lines ignored
    - Auto-append .mdp suffix if file not found without it

    Returns:
        dict mapping lowercase key to coerced value

    Raises:
        FileNotFoundError: if path does not exist (with or without .mdp suffix)
        ValueError: if a line has no '=' separator
    """
    # Try original path first, then with .mdp suffix
    actual_path = path
    if not os.path.exists(path):
        mdp_path = path if path.endswith('.mdp') else f"{path}.mdp"
        if os.path.exists(mdp_path):
            actual_path = mdp_path
        else:
            raise FileNotFoundError(f"MDP file not found: {path} (also tried {mdp_path})")

    result = {}
    with open(actual_path) as f:
        for lineno, raw in enumerate(f, 1):
            # Strip inline comments (; or #)
            for comment_char in (';', '#'):
                idx = raw.find(comment_char)
                if idx != -1:
                    raw = raw[:idx]
            line = raw.strip()
            if not line:
                continue
            if '=' not in line:
                raise ValueError(
                    f"{path}:{lineno}: expected 'key = value', got: {line!r}"
                )
            key, _, val = line.partition('=')
            key = key.strip().lower()
            val = val.strip()
            if not key:
                raise ValueError(f"{path}:{lineno}: empty key in MDP entry")
            if not val:
                raise ValueError(f"{path}:{lineno}: empty value for key '{key}'")
            result[key] = _coerce(val)
    return result


def _coerce(val: str):
    """Coerce a string value to int, float, bool, or leave as str."""
    if val.lower() in ('yes', 'true'):
        return True
    if val.lower() in ('no', 'false'):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val

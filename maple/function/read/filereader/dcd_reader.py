"""
DCD (CHARMM/NAMD) binary trajectory reader.

Reads DCD files written by CHARMM, NAMD, and other MD packages.
Returns ASE Atoms objects for each frame in the trajectory.

Format reference:
    - CHARMM source: dcdlib.c
    - NAMD source: dcdlib.C
    - VMD plugin: molfile_dcdplugin.c
"""

import struct
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from ase import Atoms


# DCD format constants
_DCD_HEADER_SIZE = 84
_DCD_TITLE_BLOCK_SIZE = 160
_DCD_CORD_MAGIC = 84


class DCDReader:
    """
    Read CHARMM/NAMD DCD binary trajectory files.

    DCD files store atomic coordinates in a compact binary format.
    This reader extracts positions and unit cell information for each frame.

    Usage:
        >>> reader = DCDReader("trajectory.dcd", natoms=1000)
        >>> first_frame = reader.read_frame(0)
        >>> all_frames = reader.read_all()
        >>> reader.close()

    Or use as context manager:
        >>> with DCDReader("trajectory.dcd") as reader:
        ...     frame = reader.read_frame(0)

    Parameters
    ----------
    path : str or Path
        Path to DCD file
    natoms : int, optional
        Number of atoms. If not provided, will be read from file header.
    """

    def __init__(self, path: Union[str, Path], natoms: Optional[int] = None):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"DCD file not found: {path}")

        self._file = open(self.path, "rb")
        self._header = None
        self._nframes = None
        self._natoms = None

        # Read header on initialization
        self._read_header()

        # Validate natoms if provided
        if natoms is not None and natoms != self._natoms:
            raise ValueError(
                f"Atom count mismatch: expected {natoms}, got {self._natoms}"
            )

    def _read_header(self):
        """Read and parse the DCD header."""
        f = self._file

        # Read header marker
        marker_data = f.read(4)
        if len(marker_data) != 4:
            raise ValueError("Invalid DCD file: too short")
        marker = struct.unpack('<i', marker_data)[0]
        if marker != 84:
            raise ValueError(f"Invalid DCD file: expected header marker 84, got {marker}")

        # Read header data (84 bytes = 21 int32)
        header_data = f.read(84)
        if len(header_data) != 84:
            raise ValueError("Invalid DCD file: incomplete header")
        self._header = np.frombuffer(header_data, dtype=np.int32)

        # Validate CORD magic
        if self._header[0] != _DCD_CORD_MAGIC:
            raise ValueError("Invalid DCD file: missing CORD magic number")

        # Read header end marker
        end_marker = f.read(4)
        if len(end_marker) != 4:
            raise ValueError("Invalid DCD file: incomplete header")
        if struct.unpack('<i', end_marker)[0] != 84:
            raise ValueError("Invalid DCD file: header end marker mismatch")

        # Extract key fields
        self._nframes = self._header[2]  # NSET
        self._natoms = self._header[6]   # NATOMS

        # Read title block (we don't parse it, just skip)
        # Format: marker(4) + title(160) + marker(4)
        f.read(4)
        f.read(160)
        f.read(4)

        # Read natom block
        f.read(4)
        natoms_check = struct.unpack('<i', f.read(4))[0]
        f.read(4)

        if natoms_check != self._natoms:
            raise ValueError(
                f"DCD header inconsistency: {self._natoms} != {natoms_check}"
            )

    @property
    def natoms(self) -> int:
        """Number of atoms in the trajectory."""
        return self._natoms

    @property
    def nframes(self) -> int:
        """Number of frames in the trajectory."""
        return self._nframes

    @property
    def timestep(self) -> float:
        """Timestep in femtoseconds (from DELTA field)."""
        # DELTA is stored in picoseconds as int32
        delta_ps = self._header[9]
        return float(delta_ps) * 1000.0  # ps -> fs

    def read_frame(self, frame_index: int) -> Atoms:
        """
        Read a specific frame from the trajectory.

        Parameters
        ----------
        frame_index : int
            Frame index (0-based)

        Returns
        -------
        Atoms
            ASE Atoms object with positions and cell info

        Raises
        ------
        IndexError
            If frame_index is out of range
        """
        if frame_index < 0 or frame_index >= self._nframes:
            raise IndexError(
                f"Frame index {frame_index} out of range [0, {self._nframes})"
            )

        # Calculate frame size
        coord_block = 4 + self._natoms * 4 + 4  # marker + data + marker
        frame_size = 56 + 3 * coord_block  # cell + 3 coord blocks

        # Calculate position of frame data
        # Header: 92 + 168 + 12 = 272 bytes
        header_size = 272
        frame_offset = header_size + frame_index * frame_size

        # Seek to frame position
        self._file.seek(frame_offset)

        # Read unit cell
        # DCD cell order: A, gamma, B, beta, alpha, C
        marker = struct.unpack('<i', self._file.read(4))[0]
        if marker != 48:
            raise ValueError(f"Invalid cell marker: expected 48, got {marker}")
        cell_data = np.frombuffer(self._file.read(48), dtype=np.float64)
        self._file.read(4)  # end marker

        # Check if cell is all zeros (non-periodic or triclinic with zero box)
        if np.allclose(cell_data, 0.0):
            # No periodic cell
            cell = None
            pbc = [False, False, False]
        else:
            # Convert DCD cell order to ASE cellpar
            # DCD: A, gamma, B, beta, alpha, C
            # ASE: a, b, c, alpha, beta, gamma
            a, gamma, b, beta, alpha, c = cell_data
            cellpar = [a, b, c, alpha, beta, gamma]
            # Create cell from cellpar (handles triclinic)
            from ase.geometry import cellpar_to_cell
            cell = cellpar_to_cell(cellpar)
            pbc = [True, True, True]

        # Read coordinates
        # X coordinates
        marker = struct.unpack('<i', self._file.read(4))[0]
        expected_bytes = self._natoms * 4
        if marker != expected_bytes:
            raise ValueError(
                f"Invalid X coord marker: expected {expected_bytes}, got {marker}"
            )
        x = np.frombuffer(self._file.read(expected_bytes), dtype=np.float32)
        self._file.read(4)  # end marker

        # Y coordinates
        marker = struct.unpack('<i', self._file.read(4))[0]
        if marker != expected_bytes:
            raise ValueError(
                f"Invalid Y coord marker: expected {expected_bytes}, got {marker}"
            )
        y = np.frombuffer(self._file.read(expected_bytes), dtype=np.float32)
        self._file.read(4)  # end marker

        # Z coordinates
        marker = struct.unpack('<i', self._file.read(4))[0]
        if marker != expected_bytes:
            raise ValueError(
                f"Invalid Z coord marker: expected {expected_bytes}, got {marker}"
            )
        z = np.frombuffer(self._file.read(expected_bytes), dtype=np.float32)
        self._file.read(4)  # end marker

        # Combine coordinates into (natoms, 3) array
        positions = np.stack([x, y, z], axis=1).astype(np.float64)

        # Create Atoms object
        # Note: DCD doesn't store element information or symbols
        # User needs to provide this separately or set it after reading
        atoms = Atoms(positions=positions)

        if cell is not None:
            atoms.set_cell(cell)
            atoms.set_pbc(pbc)

        return atoms

    def read_all(self) -> List[Atoms]:
        """
        Read all frames from the trajectory.

        Returns
        -------
        List[Atoms]
            List of ASE Atoms objects, one per frame
        """
        return [self.read_frame(i) for i in range(self._nframes)]

    def read_slice(self, start: int, stop: int, step: int = 1) -> List[Atoms]:
        """
        Read a slice of frames from the trajectory.

        Parameters
        ----------
        start : int
            Starting frame index (inclusive)
        stop : int
            Ending frame index (exclusive)
        step : int, default=1
            Step size

        Returns
        -------
        List[Atoms]
            List of ASE Atoms objects
        """
        indices = range(start, stop, step)
        return [self.read_frame(i) for i in indices]

    def close(self):
        """Close the DCD file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self) -> int:
        """Return the number of frames in the trajectory."""
        return self._nframes


def read_dcd(
    path: Union[str, Path],
    natoms: Optional[int] = None,
    frame_index: Optional[int] = None,
) -> Union[Atoms, List[Atoms]]:
    """
    Convenience function to read DCD trajectory files.

    Parameters
    ----------
    path : str or Path
        Path to DCD file
    natoms : int, optional
        Number of atoms (auto-detected if not provided)
    frame_index : int, optional
        If provided, read only this frame. Otherwise read all frames.

    Returns
    -------
    Atoms or List[Atoms]
        Single Atoms object if frame_index is specified,
        otherwise list of all frames

    Examples
    --------
        >>> # Read all frames
        >>> trajectory = read_dcd("trajectory.dcd")

        >>> # Read specific frame
        >>> frame = read_dcd("trajectory.dcd", frame_index=0)

        >>> # Read with context manager
        >>> with DCDReader("trajectory.dcd") as reader:
        ...     first_frame = reader.read_frame(0)
        ...     last_frame = reader.read_frame(-1)
    """
    with DCDReader(path, natoms) as reader:
        if frame_index is not None:
            return reader.read_frame(frame_index)
        else:
            return reader.read_all()

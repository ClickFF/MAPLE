"""
DCD (CHARMM/NAMD) binary trajectory writer.

DCD format is a FORTRAN unformatted binary format used by CHARMM, NAMD,
and X-PLOR for storing molecular dynamics trajectories. This module
implements writing DCD files compatible with CHARMM v24 and later.

Format reference:
    - CHARMM source: dcdlib.c
    - NAMD source: dcdlib.C (uses same format)
    - VMD plugin: molfile_dcdplugin.c

Binary layout (little-endian):
    - Header block (84 bytes)
    - Title block (optional, 2x 80-byte strings)
    - NATOM block (4 bytes)
    - Per frame:
        - Unit cell (6 float64) = 48 bytes + record markers
        - X coordinates (NATOMS float32)
        - Y coordinates (NATOMS float32)
        - Z coordinates (NATOMS float32)

All records are wrapped with 4-byte record markers (FORTRAN unformatted).
"""

import struct
from pathlib import Path
from typing import Optional
import numpy as np
from ase import Atoms


# DCD format constants
_DCD_HEADER_SIZE = 84
_DCD_TITLE_BLOCK_SIZE = 160  # 2 x 80-byte strings
_DCD_NATOM_BLOCK_SIZE = 4
_DCD_CORD_MAGIC = 84  # CORD identifier in first header field
_CHARMM_VERSION = 24  # CHARMM version flag at position 49

# FORTRAN record marker size (4 bytes on most platforms)
_REC_MARKER_SIZE = 4


class DCDWriter:
    """
    Write CHARMM/NAMD compatible DCD binary trajectory files.

    DCD files store atomic coordinates (and optionally unit cell dimensions)
    in a compact binary format. Coordinates are stored as float32 (Angstrom),
    making DCD files ~3x smaller than equivalent XYZ files.

    Usage:
        >>> writer = DCDWriter("traj.dcd", natoms=1000, timestep=0.5)
        >>> writer.write_frame(atoms, step=100)
        >>> writer.close()

    Or use as context manager:
        >>> with DCDWriter("traj.dcd", natoms=1000) as w:
        ...     w.write_frame(atoms, step=0)
    """

    def __init__(
        self,
        path: str | Path,
        natoms: int,
        timestep: float = 1.0,
        is_periodic: bool = False,
        first_step: int = 0,
    ):
        """
        Initialize DCD writer and write header.

        Parameters
        ----------
        path : str or Path
            Output DCD file path
        natoms : int
            Number of atoms in the system
        timestep : float, default=1.0
            Timestep in femtoseconds (fs). Stored in DELTA field as picoseconds.
        is_periodic : bool, default=False
            Whether the system is periodic. If True, unit cell dimensions
            are written for each frame.
        first_step : int, default=0
            Step number of the first frame (for restart/append scenarios).
            The internal NSET field is initialized to 0 and incremented
            with each written frame; ISTART is set to first_step.
        """
        self.path = Path(path)
        self.natoms = natoms
        self.timestep = timestep
        self.is_periodic = is_periodic
        self.first_step = first_step

        # Frame tracking
        self._nframes = 0      # frames written so far
        self._nsteps = 0       # steps counter for step numbers

        # Open file and write header
        self._file = open(self.path, "wb")
        self._write_header()

    def _write_header(self):
        """Write the DCD header."""
        # CHARMM DCD header format (from dcdlib.c and VMD molfile_dcdplugin.c):
        # The header is written as:
        #   4-byte marker (84)
        #   80 bytes of header data (20 int32, NOT 22!)
        #   4-byte marker (84)
        #
        # Wait, that's confusing. Let me look at the actual working code.
        # From MDTraj which works:
        #   They write 84 bytes between markers, which is 21 int32s.
        #   The positions are:
        #     [0] = 84 (CORD)
        #     [1] = NPRIV
        #     [2] = NSAVC
        #     ...
        #     [20] = CHARMM_VERSION (24)
        #   That's 21 elements * 4 = 84 bytes. Perfect.
        #
        # The 21-element header is:
        #   0: 84 (CORD magic)
        #   1: NPRIV (0)
        #   2: NSAVC (1)
        #   3: NSEL (1)
        #   4-5: 0
        #   6: NATOMS
        #   7-8: 0
        #   9: DELTA (timestep in ps)
        #   10-19: 0
        #   20: 24 (CHARMM_VERSION)

        delta_ps = int(round(self.timestep / 1000.0))

        # 21-element header (84 bytes)
        hdr = np.zeros(21, dtype=np.int32)
        hdr[0] = 84              # CORD magic
        hdr[1] = 0               # NPRIV
        hdr[2] = 1               # NSAVC
        hdr[3] = 1               # NSEL
        hdr[4] = 0
        hdr[5] = 0
        hdr[6] = self.natoms
        hdr[7] = 0
        hdr[8] = 0
        hdr[9] = delta_ps
        # hdr[10] through hdr[19] are already 0
        hdr[20] = 24             # CHARMM_VERSION

        # Write as FORTRAN unformatted record: marker, data, marker
        marker = struct.pack('<i', 84)  # 84 bytes of header data
        self._file.write(marker)
        self._file.write(hdr.tobytes())
        self._file.write(marker)

        # Title block (2 x 80-character strings)
        # We write descriptive titles
        title1 = f"MAPLE MD trajectory: {self.path.name}".encode('ascii')[:80]
        title2 = f"Created by MAPLE MD module".encode('ascii')[:80]
        # Pad to exactly 80 bytes each
        title1 = title1.ljust(80, b'\x00')
        title2 = title2.ljust(80, b'\x00')

        marker = struct.pack('<i', 160)  # 2 x 80 = 160 bytes
        self._file.write(marker)
        self._file.write(title1)
        self._file.write(title2)
        self._file.write(marker)

        # NATOM block
        marker = struct.pack('<i', 4)
        self._file.write(marker)
        self._file.write(struct.pack('<i', self.natoms))
        self._file.write(marker)

    def write_frame(self, atoms: Atoms, step: int = None):
        """
        Write a single frame to the DCD file.

        Parameters
        ----------
        atoms : ase.Atoms
            Atomic configuration. Coordinates must be in Angstrom (ASE default).
        step : int, optional
            Step number for logging/identification. Not stored in DCD,
            but can be used for frame numbering.
        """
        if step is not None:
            self._nsteps = step
        else:
            self._nsteps += 1

        # Get positions in Angstrom (ASE native)
        positions = atoms.get_positions()
        if positions.shape != (self.natoms, 3):
            raise ValueError(
                f"Atom count mismatch: expected {self.natoms}, "
                f"got {positions.shape[0]}"
            )

        # For DCD, we write X, Y, Z as separate float32 arrays
        x = positions[:, 0].astype(np.float32)
        y = positions[:, 1].astype(np.float32)
        z = positions[:, 2].astype(np.float32)

        # Unit cell (if periodic)
        # DCD cell order: A, gamma, B, beta, alpha, C (NOT a,b,c,alpha,beta,gamma!)
        # This odd order comes from CHARMM's history.
        if self.is_periodic and any(atoms.pbc):
            cellpar = atoms.cell.cellpar()  # [a, b, c, alpha, beta, gamma]
            # DCD order: A (0), gamma (5), B (1), beta (4), alpha (3), C (2)
            # Indices:  0          5         1       4        3        2
            dcd_cell = np.array([
                cellpar[0],  # A
                cellpar[5],  # gamma
                cellpar[1],  # B
                cellpar[4],  # beta
                cellpar[3],  # alpha
                cellpar[2],  # C
            ], dtype=np.float64)

            # Write crystal data as FORTRAN record
            # 6 doubles = 48 bytes
            marker = struct.pack('<i', 48)
            self._file.write(marker)
            self._file.write(dcd_cell.tobytes())
            self._file.write(marker)
        else:
            # Even for non-periodic, some DCD readers expect a zero cell
            # Write 6 zeros
            zero_cell = np.zeros(6, dtype=np.float64)
            marker = struct.pack('<i', 48)
            self._file.write(marker)
            self._file.write(zero_cell.tobytes())
            self._file.write(marker)

        # Write X coordinates
        nbytes = self.natoms * 4  # float32 = 4 bytes each
        marker = struct.pack('<i', nbytes)
        self._file.write(marker)
        self._file.write(x.tobytes())
        self._file.write(marker)

        # Write Y coordinates
        self._file.write(marker)
        self._file.write(y.tobytes())
        self._file.write(marker)

        # Write Z coordinates
        self._file.write(marker)
        self._file.write(z.tobytes())
        self._file.write(marker)

        self._nframes += 1

    def close(self):
        """
        Close the DCD file and update header with final frame count.

        The NSET field (header position 2, data index 2) is updated with
        the total number of frames written. This requires seeking back
        to the header and overwriting.
        """
        if self._file is None:
            return  # Already closed

        # For append mode files, we must reopen in r+b mode to write header
        # because append mode always writes at end regardless of seek position
        file_mode = getattr(self._file, 'mode', '')
        if 'a' in file_mode:
            # Close current handle and reopen for read-write
            self._file.close()
            self._file = open(self.path, "r+b")

        # Update NSET (number of sets/frames) in header
        # The header is: marker(4) + 21*4 = 88 bytes
        # NSET is at position 2 (after marker + HDR[0] + HDR[1])
        # Seek to: 4 (marker) + 2*4 (HDR[0], HDR[1]) = 12

        self._file.seek(4 + 2 * 4)  # Skip marker + HDR[0] + HDR[1]
        nset_bytes = struct.pack('<i', self._nframes)
        self._file.write(nset_bytes)

        self._file.close()
        self._file = None

    @classmethod
    def open_for_append(cls, path: str | Path) -> "DCDWriter":
        """
        Open an existing DCD file for appending.

        Reads the header to extract natoms, timestep, and periodic info,
        then positions the file pointer for writing new frames.

        Parameters
        ----------
        path : str or Path
            Existing DCD file path

        Returns
        -------
        DCDWriter
            Writer configured for appending to the existing file

        Raises
        ------
        ValueError
            If the file doesn't exist or is not a valid DCD file
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"DCD file does not exist: {path}")

        with open(path, "rb") as f:
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
            hdr = np.frombuffer(header_data, dtype=np.int32)

            # Read header end marker
            end_marker = f.read(4)
            if len(end_marker) != 4:
                raise ValueError("Invalid DCD file: incomplete header")
            if struct.unpack('<i', end_marker)[0] != 84:
                raise ValueError("Invalid DCD file: header end marker mismatch")

            # Validate CORD magic (first element is 84, last (20) should not necessarily be 84 in the data)
            # The last element of the 21-element header is CHARMM_VERSION at position 20
            if hdr[0] != 84:
                raise ValueError("Invalid DCD file: missing CORD magic number at position 0")
            if len(hdr) != 21:
                raise ValueError(f"Invalid DCD file: expected 21 header elements, got {len(hdr)}")

            natoms = hdr[6]
            # DELTA at hdr[9]
            delta_ps = hdr[9]
            timestep = delta_ps * 1000.0  # ps -> fs

            # Read title block (skip)
            f.read(4)  # marker
            f.read(160)  # title data (2 x 80)
            f.read(4)  # marker

            # Read natoms block
            f.read(4)  # marker
            natoms_check = struct.unpack('<i', f.read(4))[0]
            f.read(4)  # marker

            if natoms != natoms_check:
                raise ValueError(
                    f"DCD header inconsistency: {natoms} != {natoms_check}"
                )

            # Count existing frames by seeking to end and calculating
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()

            # Calculate frame size:
            #   cell: 4 + 48 + 4 = 56 bytes
            #   X: 4 + natoms*4 + 4
            #   Y: 4 + natoms*4 + 4
            #   Z: 4 + natoms*4 + 4
            coord_block = 4 + natoms * 4 + 4  # marker + data + marker
            frame_size = 56 + 3 * coord_block  # cell + 3 coord blocks

            # Header size:
            #   header: 4 + 84 + 4 = 92
            #   title: 4 + 160 + 4 = 168
            #   natom: 4 + 4 + 4 = 12
            header_size = 92 + 168 + 12

            traj_size = file_size - header_size
            nframes = traj_size // frame_size

        # Create writer instance and re-open file
        instance = cls.__new__(cls)
        instance.path = path
        instance.natoms = natoms
        instance.timestep = timestep
        instance.is_periodic = True  # Assume periodic (we always write cell)
        instance.first_step = 0
        instance._nframes = nframes
        instance._nsteps = nframes  # Approximate step number
        instance._file = open(path, "ab")  # Append binary mode

        return instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

"""Universal MDAnalysis Universe creation utilities.

Simple, focused utilities following UNIX philosophy: do one thing well.
Agents should read .h5 metadata to understand what's available, then call
these functions with explicit paths.
"""

from pathlib import Path
from typing import Union

import MDAnalysis as mda


def get_mda_universe(
    topology: Union[str, Path],
    trajectory: Union[str, Path],
) -> mda.Universe:
    """Get MDAnalysis Universe from topology and trajectory files.

    Simple function that creates a Universe from explicit file paths.
    Format is inferred from file extensions.

    Args:
        topology: Path to topology/structure file
        trajectory: Path to trajectory file

    Returns:
        MDAnalysis Universe object

    Raises:
        FileNotFoundError: If topology or trajectory file doesn't exist
        ValueError: If file format cannot be determined or is unsupported
    """
    topology = Path(topology)
    trajectory = Path(trajectory)

    # Validate files exist
    if not topology.exists():
        raise FileNotFoundError(
            f"Topology file not found: {topology}\n"
            f"Please verify the path is correct and the file exists in the working directory."
        )

    if not trajectory.exists():
        raise FileNotFoundError(
            f"Trajectory file not found: {trajectory}\n"
            f"Please verify the path is correct and the file exists in the working directory."
        )

    # Create universe
    try:
        if (
            topology.suffix.lower() == ".lammps-data"
            and trajectory.suffix.lower() == ".dcd"
        ):
            u = mda.Universe(
                str(topology),
                str(trajectory),
                topology_format="DATA",
                format="LAMMPS",
                atom_style="id type x y z",
                timeunit="ps",
            )
        else:
            raise ValueError("File formats not supported")
    except Exception as e:
        raise ValueError(
            f"Failed to create MDAnalysis Universe.\n"
            f"Topology: {topology}\n"
            f"Trajectory: {trajectory}\n"
            f"Error: {str(e)}"
        )

    return u

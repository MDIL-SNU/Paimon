from copy import copy

import numpy as np
from ase.atoms import Atoms
from ase.data import atomic_masses
from MDAnalysis import Universe
from MDAnalysis.coordinates.timestep import Timestep


def mda_to_atoms(
    u: Universe,
    timestep: Timestep | int,
    type_to_atomic_number: dict[int, int] | None = None,
    pbc: bool = True,
    verbose: bool = True,
) -> Atoms:
    """
    Convert an MDAnalysis timestep to an ASE Atoms object.
    If type_to_atomic_number is not provided, atomic elements will be inferred 
    by matching atomic masses from the .masses attribute.

    Parameters
    ----------
    u : Universe
        MDAnalysis Universe containing atom information and trajectory.
    timestep : int or Timestep
        Index or Timestep object specifying the frame to extract.
    type_to_atomic_number : dict[int, int], optional
        Mapping from atom types to atomic numbers, used if `u.atoms.types` exists.
    pbc : bool, default=True
        Whether periodic boundary conditions (PBC) should be applied.
    verbose : bool, default=True
        If True, prints status messages during conversion.

    Returns
    -------
    Atoms
        ASE Atoms object with positions, optional velocities, forces, and cell.

    Raises
    ------
    ValueError
        If atomic elements can't be determined or required data (positions or cell) is missing.
    """

    def guess_element_from_masses(masses: list[float]) -> list[int]:
        mass_table = np.array(copy(atomic_masses))
        mass_diff = np.abs(np.tile(mass_table, (len(masses), 1)).T - masses)
        if np.any(np.min(mass_diff, axis=0) >= 0.05):
            raise ValueError(
                "Some atom(s) mass has no match in standard atomic mass table. Maybe isotopes?"
            )
        return np.argmin(mass_diff, axis=0)

    atom_group = u.atoms
    """ TODO
    if hasattr(atom_group, "elements"):
        print(
            "[mda_to_atoms]: Use elements attr to identify elements"
        ) if verbose else None
        raise NotImplementedError()  # TODO
    """
    if type_to_atomic_number and hasattr(atom_group, "types"):
        print(
            "[mda_to_atoms]: Use types attr to identify elements"
        ) if verbose else None
        atomic_numbers = [type_to_atomic_number[t] for t in atom_group.types]  # type: ignore
    elif hasattr(atom_group, "masses"):
        print(
            "[mda_to_atoms]: Use masses attr to identify elements"
        ) if verbose else None
        atomic_numbers = guess_element_from_masses(atom_group.masses)  # type: ignore
    else:
        raise ValueError("Can't identify elements of the system")

    snapshot: Timestep = (
        u.trajectory[timestep] if isinstance(timestep, int) else timestep
    )

    atoms = Atoms(numbers=atomic_numbers)
    if snapshot.has_positions:
        atoms.set_positions(snapshot.positions.copy())
    else:
        raise ValueError("Positions are not found")

    if snapshot.has_velocities:
        print("[mda_to_atoms]: velocities are copied") if verbose else None
        atoms.set_velocities(snapshot.velocities.copy())

    if snapshot.has_forces:
        print("[mda_to_atoms]: forces are copied to .arrays") if verbose else None
        atoms.set_array("forces", snapshot.forces.copy())

    if hasattr(snapshot, "triclinic_dimensions"):
        print("[mda_to_atoms]: cell shapes are copied") if verbose else None
        atoms.set_cell(snapshot.triclinic_dimensions.copy())
    elif pbc:
        raise ValueError("pbc True, but dimesions are not found from the snapshot")

    atoms.set_pbc(pbc)
    atoms.info = {
        "time": snapshot.time,
        "dt": snapshot.dt,
    }
    return atoms

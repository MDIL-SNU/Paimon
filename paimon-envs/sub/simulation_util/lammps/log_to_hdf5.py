"""
LAMMPS log to HDF5 converter with canonical field mapping.
"""

import h5py
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

from simulation_util.io.hdf5 import is_valid_attr_value
from simulation_util.lammps.log import LogFile


FIELD_MAP = {
    "Step": ("step", "timestep"),
    "Elapsed": ("elapsed", "timestep"),
    "Temp": ("temperature", "K"),
    "Press": ("pressure", "bar"),
    "PotEng": ("potential_energy", "eV"),
    "KinEng": ("kinetic_energy", "eV"),
    "TotEng": ("total_energy", "eV"),
    "Volume": ("volume", "angstrom^3"),
    "Density": ("density", "g/cm^3"),
    "Lx": ("box_length_x", "angstrom"),
    "Ly": ("box_length_y", "angstrom"),
    "Lz": ("box_length_z", "angstrom"),
    "Pxx": ("pressure_xx", "bar"),
    "Pyy": ("pressure_yy", "bar"),
    "Pzz": ("pressure_zz", "bar"),
    "Pxy": ("pressure_xy", "bar"),
    "Pxz": ("pressure_xz", "bar"),
    "Pyz": ("pressure_yz", "bar"),
}


_CONTEXT_INJECTION = """\
When you read trajectory using get_mda_universe, 'type' and 'mass' is accessible as a per-atom field.
The type can be interpreted by type_map metadata.

For example, type_map: "C", "O", "H"; type 1 corresponds to carbon.
```python
u = get_mda_universe(
    topology=h5_dir / topology_file,
    trajectory=h5_dir / trajectory_file
)
atom_C = u.select_atoms("type 1")  # select element carbon.
```
"""


def write_lammps_thermo_to_hdf5(
    log_path: str | Path,
    output_h5: str | Path,
    type_map: list[str],
    run_index: int | None = None,
    run_note: str | list[str] = "",
    note: str | None = "",
    **run_attrs,
) -> None:
    """
    Extract thermodynamic data from LAMMPS log and write to HDF5.

    Maps LAMMPS fields (TotEng, Press, Temp) to canonical names.

    Args:
        log_path: LAMMPS log file path
        output_h5: HDF5 output file (created if new, appended if exists)
        type_map: type map ["Li", "C", "O"]
        run_index: Which run to extract (0-indexed). If None, writes ALL runs.
        run_note: Description for run(s). Can be:
                  - str: applied to all runs
                  - list[str]: one per run (length must match number of runs)
        note: Global note. Ignored if the h5 already have a global note.
        **run_attrs: Metadata stored as run attributes (topology_file, trajectory_file,
                     random_seed, etc.). Applied to all runs unless run_note is a list.

    Returns:
        Summary message

    Examples:
        # Single run (most common)
        write_lammps_thermo_to_hdf5(
            "npt.log", "data.h5",
            topology_file="structure.lammps-data",
            trajectory_file="traj.dcd",
            run_note="NPT at 298K and 1 bar"
        )

        # Replicates: multiple logs into one .h5
        for seed in [12345, 23456, 34567]:
            write_lammps_thermo_to_hdf5(
                f"npt_seed{seed}.log", "replicates.h5",
                random_seed=seed,
                trajectory_file=f"traj_{seed}.dcd"
            )

        # Sequential: multiple runs in one log with different notes
        write_lammps_thermo_to_hdf5(
            "seq.log", "data.h5",
            run_note=["Minimize", "NPT equil", "NVT prod"]
        )
    """
    log_path = Path(log_path)
    output_h5 = Path(output_h5)

    if not type_map:
        raise ValueError("type_map is required. Provide the list matching the element order of `pair_coeff` or the lammps-data (topology) file. Example: type_map['Ac', 'U'].")

    log = LogFile(str(log_path))
    if not log.runs:
        raise ValueError(f"ERROR: No runs found in {log_path.name}")

    # Determine which runs to write
    if run_index is not None:
        if run_index >= len(log.runs):
            raise ValueError(f"ERROR: run_index={run_index} but log has {len(log.runs)} run(s)")
        runs_to_write = [run_index]
    else:
        runs_to_write = list(range(len(log.runs)))

    for idx in runs_to_write:
        if "Step" not in log.runs[idx]:
            warnings.warn(f"Run {idx} in {log_path.name} missing 'Step' field.")

    # Parse run_note
    if isinstance(run_note, list):
        if len(run_note) != len(runs_to_write):
            raise ValueError(f"ERROR: run_note list has {len(run_note)} items but writing {len(runs_to_write)} run(s)")
        run_notes = run_note
    else:
        if len(runs_to_write) > 1:
            raise ValueError(f"ERROR: more than one runs ({len(runs_to_write)}) to write but run_note is not a list.")
        run_notes = [run_note]

    for k, v in run_attrs.items():
        if not is_valid_attr_value(v):
            raise ValueError(f"run_attr '{k}' has unsupported type {type(v).__name__} for HDF5 attribute.")

    # Open or create HDF5
    if output_h5.exists():
        h5f = h5py.File(output_h5, "a")
        if "runs" not in h5f:
            h5f.create_group("runs")
    else:
        h5f = h5py.File(output_h5, "w")
        h5f.attrs["agent"] = "LAMMPS agent"
        h5f.attrs["created_at"] = datetime.now().isoformat(timespec='minutes')
        note_to_add = _CONTEXT_INJECTION
        if note:
            note_to_add = note_to_add + "\n" + note
        h5f.attrs["note"] = note_to_add
        h5f.create_group("runs")

    runs_grp = h5f["runs"]
    existing = [k for k in runs_grp.keys() if k.startswith("run_")]
    next_idx = len(existing)

    # Write runs
    total_steps = 0
    for i, idx in enumerate(runs_to_write):
        run_data = log.runs[idx]

        # Create run group
        run_grp = runs_grp.create_group(f"run_{next_idx + i}")

        # Add metadata
        if run_notes[i] is not None:
            run_grp.attrs["run_note"] = run_notes[i]
        for key, value in run_attrs.items():
            run_grp.attrs[key] = value
        type_map_str = ', '.join(f'"{x}"' for x in type_map)
        run_grp.attrs["type_map"] = type_map_str

        # Write datasets
        for lammps_field, values in run_data.items():
            if lammps_field in FIELD_MAP:
                canonical_name, units = FIELD_MAP[lammps_field]
            else:
                canonical_name = lammps_field.lower()
                units = ""

            dset = run_grp.create_dataset(canonical_name, data=np.array(values))
            if units:
                dset.attrs["units"] = units

        total_steps += len(run_data["Step"]) if "Step" in run_data else 0

    h5f.close()

    n_written = len(runs_to_write)
    final_idx = next_idx + n_written - 1
    print(f"[write_lammps_thermo_to_hdf5] Wrote {n_written} run(s) from {log_path.name} to {output_h5.name} (run_{next_idx} to run_{final_idx}, {total_steps} steps)")

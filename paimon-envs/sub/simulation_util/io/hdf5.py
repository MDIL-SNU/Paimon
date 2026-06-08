import h5py
from pathlib import Path
from typing import Union


def is_valid_attr_value(value) -> bool:
    try:
        with h5py.File("probe.h5", "w", driver="core", backing_store=False) as f:
            f.attrs["_probe"] = value
        return True
    except (TypeError, ValueError, OSError):
        return False


def summarize_hdf5(hdf5_path: Union[str, Path]) -> str:
    """
    Universal HDF5 file inspector for LLM agents.

    Returns structured summary of file metadata, groups, and datasets.
    Clean output without decorative formatting.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        Structured summary or error message
    """
    hdf5_path = Path(hdf5_path)

    if not hdf5_path.exists():
        return f"ERROR: File not found: {hdf5_path}"

    lines = [f"HDF5 File: {hdf5_path.name}"]

    try:
        with h5py.File(hdf5_path, "r") as f:
            # File-level attributes
            lines.append("\nmetadata:")
            if len(f.attrs) > 0:
                for key in sorted(f.attrs.keys()):
                    value = f.attrs[key]
                    lines.append(f"  {key}: {value}")
            else:
                lines.append("  (none)")

            # Recursive structure exploration
            lines.append("\nHDF5 structure:")

            def explore_group(group, indent=0):
                prefix = "  " * indent
                items = []
                groups = []
                datasets = []

                for name in sorted(group.keys()):
                    item = group[name]
                    if isinstance(item, h5py.Group):
                        groups.append(name)
                    elif isinstance(item, h5py.Dataset):
                        datasets.append(name)

                for name in groups:
                    subgroup = group[name]
                    items.append(f"{prefix}{name}/ (group)")
                    if len(subgroup.attrs) > 0:
                        for attr_key in sorted(subgroup.attrs.keys()):
                            attr_val = subgroup.attrs[attr_key]
                            items.append(f"{prefix}  {attr_key}: {attr_val}")
                    items.extend(explore_group(subgroup, indent + 1))

                for name in datasets:
                    dset = group[name]
                    shape_str = "scalar" if dset.shape == () else f"shape={dset.shape}"
                    units = dset.attrs.get("units", "Not specified")
                    line = f"{prefix}{name} ({shape_str}, units={units})"
                    items.append(line)
                    for attr_key in sorted(dset.attrs.keys()):
                        if attr_key == "units":
                            continue
                        items.append(f"{prefix}  {attr_key}: {dset.attrs[attr_key]}")

                return items

            structure = explore_group(f, indent=1)
            if structure:
                lines.extend(structure)
            else:
                lines.append("  (empty)")

            # Summary
            lines.append("\nSUMMARY:")

            def count_items(group):
                n_groups = 0
                n_datasets = 0
                for item in group.values():
                    if isinstance(item, h5py.Group):
                        n_groups += 1
                        sub_g, sub_d = count_items(item)
                        n_groups += sub_g
                        n_datasets += sub_d
                    elif isinstance(item, h5py.Dataset):
                        n_datasets += 1
                return n_groups, n_datasets

            n_groups, n_datasets = count_items(f)
            lines.append(f"  Total groups: {n_groups}")
            lines.append(f"  Total datasets: {n_datasets}")

    except Exception as e:
        lines.append(f"\nERROR: {str(e)}")

    return "\n".join(lines)

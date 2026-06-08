"""Function bodies and interface to use env.python_call"""
import json

from paimon.world import get_env
from .environment import Environment


_list_dir_fn_body = """
import os
import json
import time

directory = k["directory"]
files_only = k["files_only"] == "True"
skip_hidden = k["skip_hidden"] == "True"

results = []
try:
    for entry in os.scandir(directory):
        if files_only and entry.is_dir():
            continue
        if skip_hidden and entry.name.startswith("."):
            continue
        file_info = {}
        file_info["name"] = entry.name
        file_info["path"] = entry.path
        file_info["size"] = entry.stat().st_size if entry.is_file else 0
        file_info["mtime"] = time.ctime(entry.stat().st_mtime)
        file_info["is_dir"] = entry.is_dir()

        results.append(file_info)
except FileNotFoundError:
    return f"Error: Directory '{directory}' not found."
except PermissionError:
    return f"Error: Permission denied to access '{directory}'."
except Exception as e:
    return f"Error: {e}"

return json.dumps(results)
"""


async def pycall_list_dir(
    env: Environment | str,
    directory: str,
    files_only: bool = True,
    skip_hidden: bool = True,
    sub_wd: str | None = None,
    venv_name: str | None = None,
) -> list[dict[str, str]] | str:
    if isinstance(env, str):
        env = get_env(env)
    ret = await env.python_call(
        _list_dir_fn_body,
        sub_wd=sub_wd,
        venv_name=venv_name,
        func_kwargs={
            "directory": directory,
            "files_only": "True" if files_only else "False",
            "skip_hidden": "True" if skip_hidden else "False",
        },
    )
    assert isinstance(ret, str)
    if ret.startswith("Error:"):
        return ret

    ret = json.loads(ret)
    assert isinstance(ret, list)
    return ret


_summarize_structure_fn_body = """
import os.path as osp
from collections import Counter

import numpy as np
from ase import io
from ase.atoms import Atoms

filepath = k["filepath"]
fname = osp.basename(filepath)
try:
    stct = io.read(filepath)  # read only first structure for speed
except Exception:
    return f"Something went wrong! Failed to read file '{fname}'"

report_lines = []
stct = stct if isinstance(stct, Atoms) else stct[0]

pbc = stct.get_pbc()
pbc_str = ", ".join(["True" if p else "False" for p in pbc])
pbc_status = f"PBC: {pbc_str} (x, y, z)"
report_lines.append(pbc_status)

if np.all(pbc):
    try:
        volume = stct.get_volume()
    except ValueError:
        return "Warning! The structure is pbc True but no lattice vectors"
    report_lines.append(f"Cell volume: {volume:.3f} Å³")

symbols = stct.get_chemical_symbols()
element_counts = Counter(symbols)
total_atoms = len(symbols)
report_lines.append(f"Total number of atoms: {total_atoms}")

element_summary = ", ".join(
    [f"{elem}: {count}" for elem, count in sorted(element_counts.items())]
)
report_lines.append(f"Elements: {element_summary}")

short_contacts = 0
ths = 0.55
positions = stct.get_positions()

if stct.get_pbc().any():
    from ase.neighborlist import neighbor_list

    try:
        i, j, _ = neighbor_list("ijd", stct, cutoff=ths)
        mask = i != j
        short_contacts = np.sum(mask)
    except Exception:
        return "Warning! Failed to run neighbor list analysis."
else:
    distances = np.linalg.norm(
        positions[:, np.newaxis] - positions[np.newaxis, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    short_contacts = np.sum(distances < ths)

if short_contacts:
    report_lines.extend([
        f"Warning! unphysically short contacts found (< {ths} Å).",
        "Invalid structure.",
        "Retry subtask to obtain valid structure."
    ])
else:
    report_lines.append("No unphysically short contacts found.")

report_lines = [f"<{fname}>"] + report_lines + [f"</{fname}>"]

return "\\n".join(report_lines)
"""


async def pycall_summarize_structure(
    env: Environment | str,
    filepath: str,
    sub_wd: str | None = None,
    venv_name: str | None = None,
) -> str:
    if isinstance(env, str):
        env = get_env(env)
    ret = await env.python_call(
        _summarize_structure_fn_body,
        timeout=120,
        sub_wd=sub_wd,
        venv_name=venv_name,
        func_kwargs={"filepath": filepath},
    )
    assert isinstance(ret, str)
    return ret


_convert_extxyz_to_lammps_data_fn_body = """
import ase.io
import os.path as osp
extxyz_path = k["extxyz_path"]
lammps_data_fname = k["lammps_data_filename"]

if osp.basename(lammps_data_fname) != lammps_data_fname:
    return f"Error: the given {lammps_data_fname} is a path, not a filename."

if not lammps_data_fname.endswith(".lammps-data"):
    return f"Error: the given {lammps_data_fname} does not end with .lammps-data."

try:
    atoms = ase.io.read(extxyz_path, index=":")
except FileNotFoundError:
    return f"Error: file not found — {extxyz_path}"

if len(atoms) > 1:
    return f"Error: more than one structure was found ({len(atoms)})."

atoms = atoms[0]
try:
    ase.io.write(lammps_data_fname, atoms, format="lammps-data", masses=True)
except Exception as e:
    return f"Error: an error occurred while writing a LAMMPS data file: {e}"

return "SUCCESS"
"""


async def pycall_extxyz_to_lammps_data(
    env: Environment | str,
    extxyz_path: str,
    lammps_data_filename: str,
    sub_wd: str | None = None,
    venv_name: str | None = None,
) -> str:
    if isinstance(env, str):
        env = get_env(env)
    ret = await env.python_call(
        _convert_extxyz_to_lammps_data_fn_body,
        timeout=60,
        sub_wd=sub_wd,
        venv_name=venv_name,
        func_kwargs={
            "extxyz_path": extxyz_path,
            "lammps_data_filename": lammps_data_filename,
        },
    )
    assert isinstance(ret, str)
    return ret


_summarize_hdf5_fn_body = """
from simulation_util.io.hdf5 import summarize_hdf5

hdf5_path = k["hdf5_path"]

try:
    summary = summarize_hdf5(hdf5_path)
    return summary
except Exception as e:
    return f"ERROR: {str(e)}"
"""


_quick_introspect_fn_body = """
import json
import os
import site
import sysconfig

from simulation_util.introspection import run_quick_introspect

# Parse parameters from k
code_content = k.get("code_content") or None
class_hint = k.get("class_hint") or None
method_hint = k.get("method_hint") or None
package_path = k.get("package_path") or None
function_hint = k.get("function_hint") or None
module_hint = k.get("module_hint") or None
repo_hint = k.get("repo_hint") or None
max_suggestions = int(k.get("max_suggestions", 10))
no_imports = k.get("no_imports", "False") == "True"

# Resolve relative package_path against site-packages (on remote environment)
if package_path:
    try:
        candidate_paths = []
        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            candidate_paths.append(purelib)
        try:
            for p in site.getsitepackages():
                if p not in candidate_paths:
                    candidate_paths.append(p)
        except Exception:
            pass
        if not os.path.isabs(package_path):
            for root in candidate_paths:
                joined = os.path.join(root, package_path)
                if os.path.exists(joined):
                    package_path = joined
                    break
    except Exception:
        pass

try:
    report, found_any = run_quick_introspect(
        code_content=code_content,
        class_hint=class_hint,
        method_hint=method_hint,
        package_path=package_path,
        function_hint=function_hint,
        module_hint=module_hint,
        repo_hint=repo_hint,
        max_suggestions=max_suggestions,
        no_imports=no_imports,
    )
    return json.dumps({"success": found_any, "report": report})
except Exception as e:
    return json.dumps({"success": False, "report": str(e)})
"""


async def pycall_summarize_hdf5(
    env: Environment | str,
    hdf5_path: str,
    sub_wd: str | None = None,
    venv_name: str | None = None,
) -> str:
    """Universal HDF5 file inspector - works with any .h5 file structure.

    Recursively explores the HDF5 file and returns a human-readable summary
    of its structure, metadata, and datasets. Works with any agent's output
    (LAMMPS agent, MD analyzer, ASE agent, Packmol agent, etc.).

    Args:
        env: Environment instance or environment ID
        hdf5_path: Path to HDF5 file
        sub_wd: Optional sub-working directory
        venv_name: Virtual environment name to use

    Returns:
        Human-readable summary string or error message
    """
    if isinstance(env, str):
        env = get_env(env)
    ret = await env.python_call(
        _summarize_hdf5_fn_body,
        timeout=60,
        sub_wd=sub_wd,
        venv_name=venv_name,
        func_kwargs={"hdf5_path": hdf5_path},
    )
    assert isinstance(ret, str)
    return ret


async def pycall_quick_introspect(
    env: Environment | str,
    code_content: str | None = None,
    class_hint: str | None = None,
    method_hint: str | None = None,
    package_path: str | None = None,
    function_hint: str | None = None,
    module_hint: str | None = None,
    repo_hint: str | None = None,
    max_suggestions: int = 10,
    no_imports: bool = False,
    sub_wd: str | None = None,
    venv_name: str | None = None,
) -> str:
    """Run quick introspection on the agent's environment.

    Uses Jedi for static discovery first (no side effects), then runtime
    import/inspect fallback. Returns a JSON string with the introspection report.

    Args:
        env: Environment instance or environment ID
        code_content: Code content for import diagnostics
        class_hint: Fuzzy or exact class name hint
        method_hint: Fuzzy or exact method name hint
        package_path: Path to the package directory (resolved on remote)
        function_hint: Fuzzy or exact function name hint
        module_hint: Module name hint (requires function_hint)
        repo_hint: Top-level import module name
        max_suggestions: Maximum number of suggestions to return
        no_imports: Whether to silence import diagnostics
        sub_wd: Optional sub-working directory

    Returns:
        JSON string with success status and report
    """
    if isinstance(env, str):
        env = get_env(env)

    func_kwargs = {
        "code_content": code_content or "",
        "class_hint": class_hint or "",
        "method_hint": method_hint or "",
        "package_path": package_path or "",
        "function_hint": function_hint or "",
        "module_hint": module_hint or "",
        "repo_hint": repo_hint or "",
        "max_suggestions": str(max_suggestions),
        "no_imports": "True" if no_imports else "False",
    }

    ret = await env.python_call(
        _quick_introspect_fn_body,
        timeout=120,
        sub_wd=sub_wd,
        func_kwargs=func_kwargs,
        venv_name=venv_name,
    )
    assert isinstance(ret, str)
    return ret

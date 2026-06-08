"""Trace a LAMMPS restart chain to find the original topology (.lammps-data) file."""

import fnmatch
import re
from pathlib import Path

MAX_CHAIN_DEPTH = 30


def parse_lammps_commands(script: Path) -> list[tuple[str, str]]:
    """Return list of (command, rest_of_line) from a LAMMPS script.

    Rejects scripts with `include` or `variable` constructs.
    Strips comments and blank lines.
    """
    commands: list[tuple[str, str]] = []
    text = script.read_text()

    # Reject variable definitions / substitutions
    if re.search(r"\$\{", text) or re.search(r"\$[A-Za-z]", text):
        # Allow only if no variable *definition* and no substitution in
        # read_data / read_restart / write_restart lines.
        pass  # We check per-command below.

    for raw_line in text.splitlines():
        line = raw_line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        cmd = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

        if cmd == "include":
            raise ValueError(f"'include' command not supported in {script}")

        if cmd == "variable":
            # We only care if variables affect read_data/read_restart/write_restart.
            # Record it but continue; we reject substitution in relevant commands.
            continue

        commands.append((cmd, rest))

    return commands


def _reject_substitution(value: str, script: Path) -> None:
    if "${" in value or re.search(r"\$[A-Za-z]", value):
        raise ValueError(
            f"variable substitution in path '{value}' not supported in {script}"
        )


def find_read_data(commands: list[tuple[str, str]], script: Path) -> str | None:
    for cmd, rest in commands:
        if cmd == "read_data":
            path = rest.split()[0]
            _reject_substitution(path, script)
            return path
    return None


def find_read_restart(commands: list[tuple[str, str]], script: Path) -> str | None:
    for cmd, rest in commands:
        if cmd == "read_restart":
            path = rest.split()[0]
            _reject_substitution(path, script)
            return path
    return None


def find_write_restarts(commands: list[tuple[str, str]], script: Path) -> list[str]:
    results = []
    for cmd, rest in commands:
        if cmd == "write_restart":
            path = rest.split()[0]
            _reject_substitution(path, script)
            results.append(path)
    return results


def restart_pattern_matches(write_pattern: str, restart_filename: str) -> bool:
    """Check if a write_restart pattern matches a restart filename.

    Handles LAMMPS wildcard '%' (replaced by processor id) and '*' (replaced by step).
    For matching purposes, '%' and '*' in the pattern match any substring.
    """
    # Convert LAMMPS pattern to a glob-like pattern
    glob_pattern = write_pattern.replace("%", "*")
    return fnmatch.fnmatch(restart_filename, glob_pattern)


def find_writer_script(restart_path: Path, from_script: Path) -> Path:
    """Find the unique .in script that writes the given restart file."""
    restart_dir = restart_path.parent
    restart_name = restart_path.name

    if not restart_dir.is_dir():
        raise FileNotFoundError(f"directory {restart_dir} does not exist")

    candidates: list[Path] = []
    for in_file in sorted(restart_dir.glob("*.in")):
        if in_file == from_script:
            continue
        commands = parse_lammps_commands(in_file)
        for pattern in find_write_restarts(commands, in_file):
            pattern_name = Path(pattern).name
            if restart_pattern_matches(pattern_name, restart_name):
                candidates.append(in_file)
                break

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"no .in script writes restart '{restart_name}' in {restart_dir}"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple .in scripts write restart '{restart_name}' "
            f"in {restart_dir}: {[str(c) for c in candidates]}"
        )

    return candidates[0]


def resolve_from_input(script_path: Path | str, as_relative_path: bool = True) -> str:
    """
    Resolve the original LAMMPS data file referenced by an input script.

    Follows any `read_restart` chain starting from `script_path` until a
    `read_data` command is found.

    Parameters
    ----------
    script_path : Path
        Path to a LAMMPS input script

    as_relative_path : bool, default=True
        Return a path relative to current working directory

    Returns
    -------
    Path
        Path to the resolved LAMMPS data file
    """
    if isinstance(script_path, str):
        script_path = Path(script_path)
    current_script = script_path.resolve()

    for _ in range(MAX_CHAIN_DEPTH):
        commands = parse_lammps_commands(current_script)

        data_ref = find_read_data(commands, current_script)
        if data_ref is not None:
            data_path = (current_script.parent / data_ref).resolve()
            if not data_path.exists():
                raise FileNotFoundError(f"topology file {data_path} does not exist")
            if as_relative_path:
                return str(data_path.relative_to(Path.cwd(), walk_up=True))
            return str(data_path)

        restart_ref = find_read_restart(commands, current_script)
        if restart_ref is None:
            raise ValueError(
                f"script {current_script} contains neither read_data nor read_restart"
            )

        restart_path = (current_script.parent / restart_ref).resolve()
        if not restart_path.exists():
            raise FileNotFoundError(f"restart file {restart_path} does not exist")

        writer_script = find_writer_script(restart_path, current_script)
        current_script = writer_script.resolve()

    raise RecursionError(f"restart chain exceeded maximum depth ({MAX_CHAIN_DEPTH})")

"""Agent that create packmol input file and run packmol code for MD input."""

from typing import Annotated, Literal

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

from paimon.knowledge.library import get_knowledge
from paimon.util.context import get_env_with_sub_wd
from paimon.world.common_tools import (
    write_file_tool,
    run_python_tool,
    run_bash_tool,
)
from paimon.agent.agent_config import AgentConfig


async def estimate_box_length(
    ctx: Context,
    molecule_filenames: Annotated[
        list[str],
        "Names of molecule files (extxyz format).",
    ],
    molecule_counts: Annotated[
        list[int],
        "The number of molecules to pack for each molecule files. Must have the same length as molecule_filenames.",
    ],
) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    python_code = """
#!/usr/bin/env python3
import sys
import math
from ase import io
from ase.data import vdw_radii
import numpy as np

# Expect COUNT FILE [COUNT FILE ...]
args = sys.argv[1:]
if len(args) % 2 != 0:
    print("Usage: calc_size.py COUNT FILE [COUNT FILE ...]")
    sys.exit(1)

total_volume = 0.0
print("Files used")
for i in range(0, len(args), 2):
    count = int(args[i])
    fname = args[i+1]
    print(f"filename : {fname}, molecule count :{count}")
    # read the molecule
    atoms  = io.read(fname, format='extxyz')
    # get vdw radii and compute atomic volumes
    radii   = np.array([vdw_radii[atom.number] for atom in atoms])
    volumes = (4.0/3.0) * math.pi * radii**3
    # accumulate count × molecule volume
    total_volume += count * volumes.sum()
print(f"Total volume summed by van der Waals volume per atoms : {total_volume} Å^3")
# cube‐root and safety factor
L = total_volume**(1/3)
print(f"cube root of the total volume : {L} Å")
L = L * 1.1
print(f"Box length with 1.1 buffer: {L} Å")
"""
    if len(molecule_counts) != len(molecule_filenames):
        return (
            "Error: molecule_counts and molecule_filenames have different lengths."
        )
    flat_args = []
    for c, f in zip(molecule_counts, molecule_filenames):
        flat_args += [str(c), f]
    argstr = " ".join(flat_args)

    filename = "tmp_calc_box_length.py"
    env.write_file(content=python_code, remote_path=filename, sub_wd=sub_wd)
    result = env.run(
        f"python -u {filename} {argstr}",
        wrap_for_llm=True,
        sub_wd=sub_wd,
        venv_name=venv_name,
    )
    result_rm = env.run(
        f"rm {filename}", wrap_for_llm=True, sub_wd=sub_wd, venv_name=venv_name
    )
    return result


estimate_box_length_tool = FunctionTool.from_defaults(
    name="estimate_box_length",
    description="Calculates the box length for packing. Box length are calculated by summing the van der Waals volume of each molecule, followed by cube root and an extra 1.1 multiplication to the box length for 10 % buffer. Returns the box length in ångströms (Å).",  # noqa: E501
    async_fn=estimate_box_length,
)


async def run_packmol(
    ctx: Context,
    packmol_input_filename: Annotated[
        str,
        "Filename of the prewritten Packmol input file.",
    ],
    packmol_log_filename: Annotated[
        str,
        "Filename for the full Packmol log.",
    ],
) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    cmd = f"packmol < {packmol_input_filename}"
    result = env.run(
        cmd, timeout=3600, wrap_for_llm=False, sub_wd=sub_wd, venv_name=venv_name
    )
    full_text = result.stdout
    lines = full_text.splitlines()
    env.write_file(
        content=full_text, remote_path=packmol_log_filename, sub_wd=sub_wd
    )
    if result.return_code != 0:
        start_idx = next(
            (i for i, ln in enumerate(lines) if "error" in ln.lower()), None
        )
        if start_idx is not None:
            start = max(0, start_idx - 8)
            crucial_text = "\n".join(lines[start:])
        else:
            tail_n = 100
            crucial_text = "\n".join(lines[-tail_n:])
        error_message = f"""
ERROR while running packmol
<stdout>
{crucial_text}
</stdout>
<stderr>
{result.stderr}
</stderr>
"""
        return error_message

    filtered = [
        line
        for line in lines
        if "Success!" in line or line.strip().startswith("Running time:")
    ]
    if not any("Success!" in line for line in filtered):
        filtered.insert(0, "fail")
    if not any(line.strip().startswith("Running time:") for line in filtered):
        filtered.append("no Running time")
    crucial_text = "\n".join(filtered)
    return crucial_text.strip()


run_packmol_tool = FunctionTool.from_defaults(
    name="run_packmol",
    description="Runs Packmol using a prewritten Packmol input file. Returns key excerpts from the full Packmol log.",  # noqa: E501
    async_fn=run_packmol,
)


def get_packmol_example(
    example_name: Annotated[
        Literal[
            "mixture",
            "interface",
            "bilayer",
            "spherical_vesicle",
            "solvated_protein",
        ],
        "Name of the example.",
    ],
) -> str:
    """Look up a commented Packmol example input script by name."""
    return get_knowledge(f"agents/packmol_agent/examples/{example_name}")


PACKMOL_EXAMPLES = {
    "mixture": "Simple mixture of water and urea in a box",
    "interface": "Hormone molecule fixed at a water/chloroform interface",
    "bilayer": "Lipid bilayer with atom constraints for orientation",
    "spherical_vesicle": "Double-layered spherical lipid vesicle",
    "solvated_protein": "Fixed protein solvated with water and ions",
}

_example_list = "\n".join(f"  - {k}: {v}" for k, v in PACKMOL_EXAMPLES.items())
get_packmol_example_tool = FunctionTool.from_defaults(
    name="get_packmol_example",
    description=(
        "Returns a commented Packmol example input script. "
        "Available examples:\n" + _example_list
    ),
    fn=get_packmol_example,
)


def config() -> AgentConfig:
    doc_text = get_knowledge("agents/packmol_agent/userguide")

    system_prompt = """\
{{common}}
{{tips}}
"""  # noqa: E501

    description = """\
<general>
Generates a atomistic structure using Packmol.

PACKMOL creates an initial point for molecular dynamics simulations by packing molecules in defined regions of space. The packing guarantees that short range repulsive interactions do not disrupt the simulations.
</general>

<typical outputs>
- `packed.extxyz`
</typical outputs>
"""  # noqa: E501
    name = "Packmol agent"
    canonical_name = name.lower().replace(" ", "_")

    return AgentConfig(
        name=name,
        description=description,  # noqa: E501
        tools=[
            estimate_box_length_tool,
            run_packmol_tool,
            get_packmol_example_tool,
            write_file_tool,
            run_python_tool,
            run_bash_tool,
        ],
        system_prompt=system_prompt.replace("{{common}}", "{{common}}")
        .replace("{{tips}}", get_knowledge(f"agents/{canonical_name}/tips"))
        .replace("{{userguide}}", doc_text),
        task_types=["Structure Generation", "Preparation"],
    )

"""Agent that write and execute ASE simulation code."""

import asyncio

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

from paimon.knowledge.library import get_knowledge
from paimon.util.context import get_env_with_sub_wd
from paimon.world.common_tools import (
    write_file_tool,
    run_bash_tool,
    switch_venv_tool,
)
from paimon.agent.agent_config import AgentConfig
from paimon.world.code_search_tools import (
    search_package_code_tool,
    quick_introspect_tool,
    runtime_probe_snippet_tool,
)


async def run_python(ctx: Context, code: str, filename: str) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    if len(filename) == 0:
        raise ValueError("The filename is empty.")
    env.write_file(content=code, remote_path=filename, sub_wd=sub_wd)
    return env.run(
        f"python -u {filename}",
        wrap_for_llm=True,
        sub_wd=sub_wd,
        timeout=1200,
        venv_name=venv_name,
    )


# With 20 min timeout and different description
run_python_tool = FunctionTool.from_defaults(
    name="run_python",
    description="""\
Write and execute Python code. Copy the complete code here. Use 'print' statements to obtain information during the simulation. This script, along with any logs, structures, or results, can be saved as files and will persist. Always use this tool to execute Python code.""",  # noqa: E501
    async_fn=run_python,
)


def config() -> AgentConfig:
    system_prompt = """\
{{common}}
{{tips}}
"""  # noqa: E501

    description = """\
<general>
Performs lightweight relaxation, atomistic modeling, analysis, and scripting using ASE.

The Atomic Simulation Environment (ASE) is a set of tools and Python modules for setting up, manipulating, running, visualizing and analyzing atomistic simulations.
</general>
"""  # noqa: E501
    name = "ASE agent"
    canonical_name = name.lower().replace(" ", "_")

    return AgentConfig(
        name=name,
        description=description,
        tools=[
            write_file_tool,
            run_python_tool,
            run_bash_tool,
            search_package_code_tool,
            runtime_probe_snippet_tool,
            quick_introspect_tool,
            switch_venv_tool,
        ],
        system_prompt=system_prompt.replace("{{common}}", "{{common}}").replace(
            "{{tips}}", get_knowledge(f"agents/{canonical_name}/tips")
        ),
        ff_prompt_key="ase",
        task_types=[
            "Relaxation",
            "Analysis",
            "Structure Generation",
            "Preparation",
            "Property Computation",
        ],
    )

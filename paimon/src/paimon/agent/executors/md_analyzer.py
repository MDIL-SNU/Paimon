from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

from paimon.knowledge.library import get_knowledge
from paimon.util.context import get_env_with_sub_wd
from paimon.world.common_tools import run_bash_tool, inspect_h5_tool
from paimon.world.retrieval_tools import query_python_library_tool
from paimon.agent.agent_config import AgentConfig


# TODO: refactor to remove duplicates code between common tools (+md_postprocessor)
async def run_python(ctx: Context, python_code: str, filename: str) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    env.write_file(content=python_code, remote_path=filename, sub_wd=sub_wd)
    return env.run(
        f"python -u {filename}",
        wrap_for_llm=True,
        sub_wd=sub_wd,
        timeout=600,
        venv_name=venv_name,
    )


run_python_tool = FunctionTool.from_defaults(
    name="run_python",
    description="""\
Write given code to the working dirctory and execute Python code. Please note that you need to copy the complete code here.
""",  # noqa: E501
    async_fn=run_python,
)


def config(use_slurm: bool = False) -> AgentConfig:
    system_prompt = """\
{{common}}
{{tips}}
"""  # noqa: E501

    description = """\
<general>
Performs one or more analyses or computations on MD trajectories or thermodynamic outputs using freud. (MDAnalysis as an interface).

The freud Python library provides a simple, flexible, powerful set of tools for analyzing trajectories obtained from molecular dynamics or Monte Carlo simulations.
</general>

<typical outputs>
- Computed physical quantities as requested
- Resulting tabular data in .h5 format
</typical outputs>
"""  # noqa: E501
    name = "MD analyzer"
    canonical_name = name.lower().replace(" ", "_")

    return AgentConfig(
        name=name,
        description=description,
        tools=[
            run_python_tool,
            query_python_library_tool,
            run_bash_tool,
            inspect_h5_tool,
        ],
        system_prompt=system_prompt.replace("{{common}}", "{{common}}").replace(
            "{{tips}}", get_knowledge(f"agents/{canonical_name}/tips")
        ),
        task_types=["Analysis", "Property Computation"],
    )

"""Agent that write LAMMPS input and run LAMMPS code for MD simulation."""

from typing import Annotated

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

from paimon.knowledge.library import get_knowledge
from paimon.util.context import get_env_with_sub_wd
from paimon.agent.agent_config import AgentConfig
from paimon.world.common_tools import (
    write_file_tool,
    run_python_tool,
    run_bash_tool,
    switch_venv_tool,
)
from paimon.world.slurm_tools import (
    submit_and_wait,
    simple_odin_submit_and_wait_tool,
    simple_kias_submit_and_wait_tool,
    kisti_private_submit_and_wait_tool,
    simple_odin_submit_and_wait_tool_wait_in_tool,
)
from paimon.world.remote_python import pycall_extxyz_to_lammps_data, pycall_list_dir
from paimon.rag import LAMMPSRAGSystem
from paimon.util.log import debug
from paimon import cfg, models as mdl
import paimon.rag.format as rag_format


_lammps_rag_system = None


def _get_lammps_rag_system() -> LAMMPSRAGSystem:
    """Get or initialize LAMMPS RAG system."""
    global _lammps_rag_system
    if _lammps_rag_system is None:
        # Use config for all settings
        _lammps_rag_system = LAMMPSRAGSystem(force_rebuild=False)
        _lammps_rag_system.build_index()

    return _lammps_rag_system


async def retrieve_lammps_doc(ctx: Context, query: str) -> str:
    """Search LAMMPS docs for a specific query and return the most relevant
    command."""
    env, sub_wd, _ = await get_env_with_sub_wd(ctx)
    rag = _get_lammps_rag_system()

    # Get current subtask instruction for task context
    agent_state = await ctx.store.get("agent_state")
    task_description = getattr(agent_state, "instruction", query)

    normalized = query.strip().replace(" ", "_").replace("/", "_")
    exact_result = rag.get_document_by_name(normalized, env=env, sub_wd=sub_wd)

    if exact_result is not None:  # check for direct matches
        debug(f"[retrieve_lammps_doc] Exact match found: {normalized}")
        doc = await rag_format.format_annotated(
            original_query=query,
            result=exact_result,
            task_description=task_description,
        )
        return doc

    debug(f"[retrieve_lammps_doc] proceed to RAG: {query}")
    result, candid = await rag.retrieve(
        query=query, top_k_stage1=6, env=env, sub_wd=sub_wd
    )

    if not result:
        return """
No documentation is selected for the query.

Candidate command names:
{candid_str}
""".format(candid_str=", ".join([cc.command_name for cc in candid]))

    doc = await rag_format.format_annotated(
        original_query=query,
        result=result,
        task_description=task_description,
    )
    return doc


async def retrieve_lammps_docs_for_task(ctx: Context, task_description: str) -> str:
    """Retrieve comprehensive LAMMPS docs for complex simulation tasks using
    AI-powered query decomposition."""
    rag = _get_lammps_rag_system()
    result = await rag.retrieve_complex(
        task_description=task_description,
        top_k_per_subquery=4,
    )

    return result.formatted_output


async def convert_extxyz_data(
    ctx: Context,
    extxyz_path: str,
    lammps_data_filename: str,
) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    ret = await pycall_extxyz_to_lammps_data(
        env,
        extxyz_path=extxyz_path,
        lammps_data_filename=lammps_data_filename,
        sub_wd=sub_wd,
        venv_name=venv_name,
    )
    if ret == "SUCCESS":
        ret2 = env.sys_run(
            f"awk '/^Masses$/{{f=1}} /^Atoms[[:space:]]/{{exit}} f' {lammps_data_filename}",
            sub_wd=sub_wd,
        )
        return ret + "\n" + ret2
    return ret


async def write_lammps_script(
    ctx: Context,
    content: str,
    lammps_script_filename: str = "lammps.in",
    generate_job_script: Annotated[
        bool,
        "If True, writes the job_script.j file for submission.",
    ] = True,
) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    env.write_file(
        content=content, remote_path=lammps_script_filename, sub_wd=sub_wd
    )

    ls_bf = await pycall_list_dir(
        env,
        directory=".",
        files_only=True,
        skip_hidden=True,
        sub_wd=sub_wd,
        venv_name=venv_name,
    )
    assert isinstance(ls_bf, list)
    ls_bf = set([f["name"] for f in ls_bf])

    result = env.run(
        f"lmp -skiprun -in {lammps_script_filename}",
        wrap_for_llm=False,
        sub_wd=sub_wd,
        timeout=60,
        venv_name=venv_name,
    )

    ls_af = await pycall_list_dir(
        env,
        directory=".",
        files_only=True,
        skip_hidden=True,
        sub_wd=sub_wd,
        venv_name=venv_name,
    )
    assert isinstance(ls_af, list)
    ls_af = set([f["name"] for f in ls_af])

    for byprod in ls_af - ls_bf:
        env.sys_run(f"rm {byprod}", sub_wd=sub_wd)

    if result.return_code != 0:
        error_output = result.stdout.strip()
        return f"Dry run failed. The input script ({lammps_script_filename}) has been saved for correction. \n\nError output:\n{error_output}"

    message = f"Dry run completed successfully. The LAMMPS input script {lammps_script_filename} is valid."
    if generate_job_script:
        env.write_file(
            content=f"lmp -in {lammps_script_filename}",
            remote_path="job_script.j",
            sub_wd=sub_wd,
        )
        message += "\nA job submission script `job_script.j` has been created. Run simulation via the submit_and_wait tool."
    else:
        message += "\nTo run the simulation, create a job submission script and then submit it with submit_and_wait tool."

    return message


async def lammps_pre_init_hook(
    prompt: str, agent_state: mdl.SubtaskAgentState, subtask: mdl.SubtaskWithDir
) -> str:
    """Pre-initialization hook that retrieves LAMMPS documentation.

    This hook is called after the LAMMPS agent is created but before
    it starts executing. It uses the RAG system to retrieve relevant
    documentation based on the subtask instruction.

    Args:
        prompt: Complete agent prompt (will be extended with retrieved docs)
        agent_state: Agent state (for future use)
        subtask: Subtask information (contains instruction, working dir, etc.)

    Returns:
        Modified prompt with <retrieved_knowledge> section appended

    Raises:
        Exception: If RAG retrieval fails (will cause subtask to fail)
    """
    rag = _get_lammps_rag_system()

    # Retrieve comprehensive documentation using query decomposition
    result = await rag.retrieve_complex(
        task_description=subtask.instruction,
        top_k_per_subquery=4,
    )

    num_queries = len(result.subqueries)
    debug(f"[lammps_pre_init_hook] Retrieved {num_queries} subquery results")

    # Format for prompt injection with XML section
    output = []
    output.append("")
    output.append("<retrieved_knowledge>")
    output.append(
        "The following LAMMPS documentation has been automatically "
        "retrieved based on your task:"
    )
    output.append("")
    output.append(result.formatted_output)
    output.append("</retrieved_knowledge>")

    return prompt + "\n".join(output)


retrieve_lammps_docs_for_task_tool = FunctionTool.from_defaults(
    name="retrieve_lammps_docs_for_task",
    description=(
        "Retrieve LAMMPS docs for complex tasks by decomposing them into "
        "primitive queries. Use for multi-step simulations. This call is expensive and should not be used unless necessary."
    ),
    async_fn=retrieve_lammps_docs_for_task,
)


retrieve_lammps_doc_tool = FunctionTool.from_defaults(
    name="retrieve_lammps_doc",
    description="""\
Search and retrieve LAMMPS documentation, describe what you want to do as if explaining to a colleague who will find the right command for you.
You can also use it to search with the exact command name.

Explanation examples:
- "run simulation at constant pressure and temperature"  
- "calculate temperature of a subset of atoms"
- "remove atoms inside a spherical region"

Exact command examples:
- "fix_nve"
- "compute_ke_atom"

Note: this tool returns only single document (command) per use.
""",  # noqa: E501
    async_fn=retrieve_lammps_doc,
)


convert_extxyz_data_tool = FunctionTool.from_defaults(
    name="convert_extxyz_data",
    description=(
        "Converts an extxyz file into a LAMMPS data file and returns its element-type mapping."
    ),
    async_fn=convert_extxyz_data,
)


write_lammps_script_tool = FunctionTool.from_defaults(
    name="write_lammps_script",
    description="Writes a LAMMPS input script and performs a dry run to check for errors. Optionally generates a shell script for LAMMPS execution. Returns the result of the dry run.",
    async_fn=write_lammps_script,
)


def config() -> AgentConfig:
    system_prompt = """\
{{common}}
{{tips}}
"""  # noqa: E501

    slurm_submit_tool = submit_and_wait
    if cfg.slurm_policy == "odin":
        debug("SLURM SUBMIT AS ODIN")
        slurm_submit_tool = simple_odin_submit_and_wait_tool
    elif cfg.slurm_policy == "odin_wait_in_tool":
        debug("SLURM SUBMIT AS ODIN WAIT IN TOOL")
        slurm_submit_tool = simple_odin_submit_and_wait_tool_wait_in_tool
    elif cfg.slurm_policy == "kias":
        debug("SLURM SUBMIT AS KIAS")
        slurm_submit_tool = simple_kias_submit_and_wait_tool
    elif cfg.slurm_policy == "kisti_private":
        debug("SLURM SUBMIT AS KISTI_PRIVATE")
        slurm_submit_tool = kisti_private_submit_and_wait_tool
    else:
        raise ValueError(f"Unknown slurm policy: {cfg.slurm_policy}")

    tools = [
        write_lammps_script_tool,
        # retrieve_lammps_docs_for_task_tool,
        retrieve_lammps_doc_tool,
        convert_extxyz_data_tool,
        write_file_tool,
        run_python_tool,
        run_bash_tool,
        switch_venv_tool,
        slurm_submit_tool,
    ]

    description = """\
<general>
Performs a molecular dynamics (MD) simulation using LAMMPS.

LAMMPS is a classical molecular dynamics simulation code focusing on materials modeling. It was designed to run efficiently on parallel computers and to be easy to extend and modify.

In instruction, do not include any analysis tasks that require postprocessing of simulation results. Create a separate [[Analysis]] for such cases.
</general>

<typical outputs>
- Any tabular data (e.g. thermodynamic data): `data.h5` (with path to the topology file)
- Trajectory: `traj.dcd` (note: this format contains only atomic positions)
- Restart file to continue the MD simulation: `last.restart`
</typical outputs>
"""  # noqa: E501
    name = "LAMMPS agent"
    canonical_name = name.lower().replace(" ", "_")

    return AgentConfig(
        name=name,
        description=description,
        tools=tools,  # type: ignore
        system_prompt=system_prompt.replace("{{common}}", "{{common}}").replace(
            "{{tips}}", get_knowledge(f"agents/{canonical_name}/tips")
        ),
        ff_prompt_key="lammps",
        task_types=["MD"],
        critic_gate_tool_names=["submit_and_wait"],
        # pre_init_hook=lammps_pre_init_hook,
    )

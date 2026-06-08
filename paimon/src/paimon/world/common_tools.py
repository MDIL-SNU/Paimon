from typing import Literal, Annotated

from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.utils import create_schema_from_function

from paimon.models import SubtaskAgentState, SubtaskFailReason, TaskStatus, Value
from paimon.util.context import get_env_with_sub_wd
from paimon.util.log import debug, debug_var


async def run_python(ctx: Context, code: str, filename: str) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    if len(filename) == 0:
        raise ValueError("The filename is empty.")
    env.write_file(content=code, remote_path=filename, sub_wd=sub_wd)
    return env.run(
        f"python -u {filename}",
        wrap_for_llm=True,
        sub_wd=sub_wd,
        timeout=120,
        venv_name=venv_name,
    )


run_python_tool = FunctionTool.from_defaults(
    name="run_python",
    description="""\
Write given code to the working dirctory and execute Python code. Please note that you need to copy the complete code here.
""",  # noqa: E501
    async_fn=run_python,
)


# DO NOT USE
async def run_bash(ctx: Context, command: str) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    return env.run(command, wrap_for_llm=True, sub_wd=sub_wd, venv_name=venv_name)


run_bash_tool = FunctionTool.from_defaults(
    name="run_bash",
    description="""\
Execute a Bash command. You will receive the command’s stdout and stderr. Do not use this tool to write files.
""",
    async_fn=run_bash,
)


async def write_file(ctx: Context, content: str, filename: str) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    if len(filename) == 0:
        raise ValueError("The filename is empty.")
    if len(filename) == 0:
        return "[Error] filename should not empty"
    env.write_file(content=content, remote_path=filename, sub_wd=sub_wd)
    return f"{filename} has been successfully saved to the working directory."


write_file_tool = FunctionTool.from_defaults(
    name="write_file",
    description="""Write a file to the working directory.""",
    async_fn=write_file,
)


async def abort_task(
    ctx: Context, reason: Literal["ambiguous", "give_up"], message_to_user: str
):
    state: SubtaskAgentState = await ctx.store.get("agent_state")
    state.message_to_planner = message_to_user
    state.task_status = TaskStatus.FAIL
    state.subtask_fail_reason = SubtaskFailReason.ABORT
    debug(f"[abort_task] agent aborted task {reason}")
    await ctx.store.set("agent_state", state)
    return message_to_user  # more natural in memory


abort_task_tool = FunctionTool.from_defaults(
    name="abort_task",
    description="""\
Use this if the task cannot be completed with valid results.
1. give_up: You attempted the task but failed, and recovery options are exhausted.
2. ambiguous: Do not proceed. Use this if:
  - Required instructions (e.g., ensemble, duration) are missing.
  - Proceeding may cause invalid or unnecessary results.
""",  # noqa: E501
    async_fn=abort_task,
    return_direct=True,
)


async def complete_task(
    ctx: Context,
    message_to_user: str,
    file_usage_summary: str,
    **submitted_values,
):
    state: SubtaskAgentState = await ctx.store.get("agent_state")
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)  # type: ignore

    outputs_ok = True
    err_msg = ""
    for output_file in state.required_output_files:
        for fname in output_file.enumerate():
            if not env.file_exists(fname, sub_wd=sub_wd):
                outputs_ok = False
                err_msg += f'The requested output file "{output_file}" does not exist.\n'

    for output_value in state.required_output_values:
        if output_value.name not in submitted_values:
            err_msg += f'The requested output value "{output_value.name}" has not been provided.\n'
            outputs_ok = False

    ret = None
    if outputs_ok:
        state.message_to_planner = message_to_user
        state.file_usage_summary = file_usage_summary
        state.output_values = submitted_values  # may contain more
        state.task_status = TaskStatus.SUCCESS
        ret = "The task is complete. The results have been reported to the user."
    else:
        state.task_status = TaskStatus.HOLD
        ret = "Task completion failed:\n" + err_msg  # to agent, does not end loop

    await ctx.store.set("agent_state", state)
    return ret


def complete_task_tool_factory(required_output_values: list[str] | list[Value]):
    required_output_values = [
        v.name if isinstance(v, Value) else v for v in required_output_values
    ]

    schema = create_schema_from_function(
        name="complete_task",
        func=complete_task,
        additional_fields=[(v, float) for v in required_output_values],
        ignore_fields=["ctx", "submitted_values"],
    )
    return FunctionTool.from_defaults(
        name="complete_task",
        description="""\
Use this only when the task has been successfully executed and produced **valid** and **complete** results, whether positive or negative.
- Report results (files, values, or outputs if requested) to the user.
- Do not use if the task failed, produced invalid results, or is incomplete.
- Results may show success (criteria met) or failure (criteria not met), but they must be trustworthy and complete.
    """,  # noqa: E501
        async_fn=complete_task,
        fn_schema=schema,
        return_direct=True,
    )


async def inspect_h5(ctx: Context, h5_file: str) -> str:
    """Universal HDF5 file inspector - works with ANY .h5 file structure.

    Recursively explores the HDF5 file and returns a human-readable summary
    of its structure, metadata, and datasets. Works with any agent's output
    (LAMMPS agent, MD analyzer, ASE agent, Packmol agent, etc.).

    Args:
        h5_file: Path to HDF5 file (e.g., "results.h5", "../01_TaskName/data.h5")

    Returns:
        Human-readable summary of file structure and metadata
    """
    from paimon.world.remote_python import pycall_summarize_hdf5

    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)

    ret = await pycall_summarize_hdf5(
        env=env,
        hdf5_path=h5_file,
        sub_wd=sub_wd,
        venv_name=venv_name,
    )

    return ret


inspect_h5_tool = FunctionTool.from_defaults(
    name="inspect_h5",
    description="""\
Universal HDF5 file inspector - works with ANY .h5 file structure.

Use this to explore and understand .h5 files.

Returns a complete summary showing:
- File-level metadata (agent, created_at, note, etc.)
- Hierarchical structure (groups and datasets)
- Dataset shapes, types, units, and descriptions
- Summary statistics

Example usage:
- inspect_h5("../01_MDSimulation/simulation.h5")
- inspect_h5("../02_Analysis/rdf_data.h5")
""",
    async_fn=inspect_h5,
)


async def switch_venv(ctx: Context, venv_name: str) -> str:
    """Switch to a different Python virtual environment."""
    from paimon.agent.registry import get_agent_config
    from paimon.knowledge.library import get_knowledge

    env, _, current_venv = await get_env_with_sub_wd(ctx)
    # TODO: What if current_vent is same as the given venv_name?

    available = list(env._venv_map.keys())
    if venv_name not in available:
        return f"[Error] Unknown venv: {venv_name}. Available: {available}"

    # Update agent state (NOT global state)
    agent_state = await ctx.store.get("agent_state")
    old_venv = agent_state.current_venv
    agent_state.current_venv = venv_name
    await ctx.store.set("agent_state", agent_state)

    result = f"Switched venv: {old_venv} -> {venv_name}"

    # Return FF knowledge for the new environment
    ff_family = venv_name
    config = get_agent_config(agent_state.agent_name)
    if config.ff_prompt_key:
        try:
            ff_knowledge = get_knowledge(
                f"forcefield/{ff_family}/{config.ff_prompt_key}"
            )
            result += "\n\nThe force field guide for the switched environment is below. Use this instead of the previous FF instructions.\n\n" + ff_knowledge  # noqa: E501
        except FileNotFoundError:
            pass

    return result


# TODO: description should be auto-generated from paimon-envs/ structure
# TODO: description should be more general (not specific to ML potentials)
switch_venv_tool = FunctionTool.from_defaults(
    name="switch_venv",
    description="""\
Switch to a different Python environment. Available environments:
- base: General Python environment with ASE, numpy, etc.
- sevennet: Environment with SevenNet (SevenNetCalculator)
- mace: Environment with MACE (mace_mp, MACECalculator)
Use this before running code that requires a specific ML potential.""",
    async_fn=switch_venv,
)


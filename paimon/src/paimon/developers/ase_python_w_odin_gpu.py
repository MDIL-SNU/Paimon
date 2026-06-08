import asyncio

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

from paimon.models import SubtaskAgentState
from paimon.util.context import get_env_with_sub_wd
from paimon.world.slurm import get_slurm_tracker
from paimon.util.log import debug, debug_var, debug_assert
from paimon.world.environment import MAX_BASH_OUTPUT_CHARS


PARTITION = "gpu"


async def run_python(ctx: Context, code: str, filename: str) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    env.write_file(content=code, remote_path=filename, sub_wd=sub_wd)

    sbatch_command = (
        f"sbatch "
        f"--job-name={env.id}_ase "
        f"--ntasks-per-node=1 "
        f"--partition={PARTITION} "
        f"--nodes=1 "
        f"--gres=gpu:1 "
        f"--time=00-00:20 "
        '--output=".slurm-%j.out" '
        '--error=".slurm-%j.err" '
    )
    setup_script = env.get_current_setup_script(venv_name)
    sbatch_command += (
        f'--wrap "source ~/.bash_profile &> /dev/null '
        f"&& source /home/{env.user_name}/{setup_script} "
        f'&& python {filename} 1> _stdout 2> _stderr"'
    )

    result = env.run(sbatch_command, wrap_for_llm=False, sub_wd=sub_wd, venv_name=venv_name)

    debug_assert(result.return_code == 0, "[ase] non zero return code submitted job")

    slurm = await get_slurm_tracker()
    job_id = int(result.stdout.strip().split()[-1])
    slurm.enroll_job(
        env.id,
        job_id,
        filename,
        {"partition": PARTITION},
    )
    await asyncio.sleep(slurm.poll_interval_sec + 0.1)

    # Hope the job ends shortly after the submit
    job_state = None
    while not job_state or job_state not in [
        "FAILED",
        "COMPLETED",
        "TIMEOUT",
        "NODE_FAIL",
        "UNKNOWN",
    ]:
        try:
            job_state = slurm.job_states[env.id][job_id]["state"]
            await asyncio.sleep(slurm.poll_interval_sec)
        except KeyError as e:
            debug(f"[ase] waiting job state to be updated: {e}")
            continue
    res = {}
    res["stdout"] = env.sys_run("cat _stdout", sub_wd=sub_wd)
    res["stderr"] = env.sys_run("cat _stderr", sub_wd=sub_wd)

    debug_assert(job_state != "NODE_FAIL", "Node failure")
    header = "<system>"
    if job_state == "COMPLETED":
        header += "The script executed successfully."
    elif job_state == "FAILED":
        header += "The script encountered an error."
    elif job_state == "UNKNOWN":
        header += "The script ended."
    elif job_state == "TIMEOUT":
        header += "The script timed out after 20 minutes."
    header += "</system>"

    ret = []
    for x in ("stdout", "stderr"):
        val = res[x]
        if len(val) > MAX_BASH_OUTPUT_CHARS:
            val = f"<system>The outputs are truncated!</system>\n{val[:MAX_BASH_OUTPUT_CHARS]}"
        ret.append(
            """\
<{}>
{}
</{}>
""".format(x, val.rstrip(), x)
            if val
            else "(no {})".format(x)
        )
    ret = "\n".join(ret)

    env.sys_run("rm _stdout && rm _stderr", sub_wd=sub_wd)

    return header + "\n" + ret


TARGET_TOOL = "run_python"


def wrap_tool(tool: FunctionTool, agent_name: str):
    if not (tool.metadata.get_name() == "run_python" and agent_name == "ASE agent"):
        return tool

    f = tool.async_fn

    async def wrapper(*args, **kwargs):
        ctx: Context = kwargs["ctx"]
        agent_state: SubtaskAgentState = await ctx.store.get("agent_state")
        if agent_state.agent_name != "ASE agent":
            return await f(*args, **kwargs)
        else:
            return await run_python(*args, **kwargs)

    debug("[ASE_PYTHON_IN_ODIN] applied")
    tool._async_fn = wrapper
    return tool

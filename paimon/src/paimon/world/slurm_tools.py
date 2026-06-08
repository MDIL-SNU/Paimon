"""Tools to work with slurm. Currenlt assuming only one job per agent (context)"""

import asyncio

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

from paimon.util.context import get_env_with_sub_wd

from paimon import cfg
from paimon.models import TaskStatus
from paimon.util.log import debug, debug_var, debug_assert
from paimon.world.slurm import get_slurm_tracker

"""
Currently only lammps agent with wait until finish only.
Other tools are not removed, as maybe needed for future reference

Not used: server_status, wait_short, wait_long, get_job_state, cancle_job

TODO: slurm_job_id to agent_state
"""


async def _tail_slurm_outputs(
    env, sub_wd, job_id: int, venv_name: str | None = None
) -> list[str]:
    rets = ["<system>The last ten lines of slurm.out and slurm.err.</system>"]
    for suffix in ["out", "err"]:
        tail = env.run(
            f"tail -n 10 slurm-{job_id}.{suffix}",
            wrap_for_llm=False,
            sub_wd=sub_wd,
            no_history=True,
            venv_name=venv_name,
        ).stdout
        if len(tail) > 0:
            rets.append(f"""\
<slurm-{job_id}.{suffix}>
{tail}
</slurm-{job_id}.{suffix}>
""")
        else:
            rets.append(f"<system>slurm-{job_id}.{suffix} is empty</system>")

    return rets


# TODO: server and job status can be more effectively informed to agent without tools
async def query_server_status() -> str:
    slurm = await get_slurm_tracker()
    return slurm.server_state


server_status_tool = FunctionTool.from_defaults(
    name="query_server_status",
    description="""\
Query the current state of the SLURM-managed cluster. Returns partition list, node/core availability, and allocation status in structured format. Use this before job submission to validate partition or resource availability.""",
    async_fn=query_server_status,
)


# I think this tools is obsolute
async def get_job_state(ctx: Context) -> str:
    debug("[slurm_tools] get_job_state tool called")
    env, _, venv_name = await get_env_with_sub_wd(ctx)
    job_id = await ctx.store.get("slurm_job_id", None)
    if job_id is None:
        return "No jobs submitted"

    slurm = await get_slurm_tracker()
    job_state = slurm.job_states[env.id][job_id]["state"]
    return f"Current job state: {job_state}"


get_job_state_tool = FunctionTool.from_defaults(
    name="get_job_state",
    description=""""Get current job state. Job_id(s) is automatically inferred from the recored.""",
    async_fn=get_job_state,
)


async def submit_job(
    ctx: Context,
    script_path: str,
    partition: str,
    ntasks_per_node: int = 1,
    ngpus: int = 1,
    nnodes: int = 1,
    cancel_if_pending: bool = True,
) -> str:
    debug("[slurm_tools] submit_job tool called")
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)

    sbatch_command = (
        f"sbatch "
        f"--job-name={env.id} "
        f"--ntasks-per-node={ntasks_per_node} "
        f"--partition={partition} "
        f"--nodes={nnodes} "
        '--output="slurm-%j.out" '
        '--error="slurm-%j.err" '
    )
    if ngpus > 0:
        sbatch_command += f"--gres=gpu:{ngpus} "
    setup_script = env.get_current_setup_script(venv_name)
    sbatch_command += f'--wrap "source ~/.bash_profile &> /dev/null && source /home/{env.user_name}/{setup_script}&& bash {script_path}"'

    result = env.run(sbatch_command, wrap_for_llm=False, sub_wd=sub_wd, venv_name=venv_name)

    if result.return_code != 0:
        debug(f"[slurm_tools] non zero return code submitted job, {result}")
        return str(result)

    slurm = await get_slurm_tracker()
    job_id = int(result.stdout.strip().split()[-1])
    slurm.enroll_job(
        env.id,
        job_id,
        script_path,
        {"partition": partition},
    )
    await asyncio.sleep(slurm.poll_interval_sec + 0.1)

    job_state = None
    while not job_state or job_state == "WAITING_UPDATE":
        try:
            job_state = slurm.job_states[env.id][job_id]["state"]
            await asyncio.sleep(slurm.poll_interval_sec)
        except KeyError as e:
            debug(f"[slurm_tools] waiting job state to be updated: {e}")
            continue

    agent_state = await ctx.store.get("agent_state")
    if cancel_if_pending and job_state == "PENDING":
        debug(f"[slurm_tools] submit_job {job_id} in PENDING")
        result = env.run(
            f"scancel {job_id}",
            wrap_for_llm=False,
            sub_wd=sub_wd,
            assert_failure=True,
            venv_name=venv_name,
        )
        ret = f"The partition {partition} is full. Try another partition."

    elif job_state in ["FAILED", "COMPLETED", "UNKNOWN"]:  # job immediately ended
        debug(f"[slurm_tools] submit_job {job_id} in fail or completed state")
        rets = [
            f"Job submitted. The job id is {job_id}. The job ended in the state {job_state}.",
        ]
        tail = await _tail_slurm_outputs(env, sub_wd, job_id, venv_name)
        rets.extend(tail)
        ret = "\n".join(rets)

    elif job_state == "RUNNING":  # job in long running state
        debug(f"[slurm_tools] submit_job {job_id} in running state")
        rets = [
            f"Job submitted. The job id is {job_id}. State: RUNNING.",
        ]
        tail = await _tail_slurm_outputs(env, sub_wd, job_id, venv_name)
        rets.extend(tail)
        agent_state.slurm_job_id = job_id
        ret = "\n".join(rets)

    elif job_state == "PENDING":
        debug(f"[slurm_tools] submit_job {job_id} in pending state")
        rets = [
            f"Job submitted. The job id is {job_id}. State: PENDING.",
        ]
        agent_state.slurm_job_id = job_id
        ret = "\n".join(rets)

    else:  # PENDING?
        debug(f"[slurm_tools] submit_job {job_id} in unexpected state {job_state}")
        ret = f"Job submitted. The job id is {job_id}. Current state: {job_state}."
    await ctx.store.set("agent_state", agent_state)
    return ret


submit_job_tool = FunctionTool.from_defaults(
    name="submit_job",
    description="""Submit a SLURM job with specified script and resource configuration. Returns the SLURM job ID. Output is saved as slurm-%j.out and slurm-%j.err.""",
    async_fn=submit_job,
)


async def cancel_job(ctx: Context) -> str:
    debug("[slurm_tools] cancel_job tool called")
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    job_id = await ctx.store.get("slurm_job_id", None)
    if job_id is None:
        return "No job submitted"

    result = env.run(
        f"scancel {job_id}", wrap_for_llm=False, sub_wd=sub_wd, venv_name=venv_name
    )
    if result.return_code == 0:
        await ctx.store.set("slurm_job_id", None)
        return f"Job id: {job_id} canceled. Current job state: CANCELED"
    else:
        debug(f"[slurm_tools] cancel job: job cancel failed: {job_id}")
        return f"Failed to cancel job: {result.stderr}"


cancel_job_tool = FunctionTool.from_defaults(
    name="cancel_job",
    description="""Cancel the last submitted SLURM job.""",
    async_fn=cancel_job,
)


async def submit_and_wait(
    ctx: Context,
    script_path: str,
    partition: str,
    ntasks_per_node: int = 1,
    ngpus: int = 1,
    nnodes: int = 1,
) -> str:
    state = await ctx.store.get("agent_state")

    submit_job_msg = await submit_job(
        ctx,
        script_path,
        partition,
        ntasks_per_node=ntasks_per_node,
        ngpus=ngpus,
        nnodes=nnodes,
    )
    # Valid for starting long wait
    state = await ctx.store.get("agent_state")
    if state.slurm_job_id:
        # submitted => in PEDNING or RUNNING state
        state.task_status = TaskStatus.WAIT
        state.max_wait_minutes = 1440
        await ctx.store.set("agent_state", state)
        return f"{submit_job_msg}.\nStart waiting ..."
    else:
        # submitted => the job is immediatly failed or some other reason
        state.task_status = TaskStatus.HOLD
        await ctx.store.set("agent_state", state)
        return submit_job_msg


submit_and_wait_tool = FunctionTool.from_defaults(
    name="submit_and_wait",
    description="""\
Submit a SLURM job with the given script_path, partition, ntasks_per_node, ngpus, and nnodes, then block until the job completes (or until log updates are available). Output is saved to slurm-%j.out and slurm-%j.err.
""",
    async_fn=submit_and_wait,
    return_direct=True,
)


async def _submit_and_wait_odin_gpu(
    ctx: Context,
    script_path: str,
) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    state = await ctx.store.get("agent_state")

    if not env.file_exists(script_path, sub_wd):
        state.task_status = TaskStatus.HOLD
        await ctx.store.set("agent_state", state)
        return f"No such file: {script_path}"

    priority = ["gpu5", "gpu4", "gpu3", "gpu2"]
    pending_default = "gpu5"
    all_pending = True

    submit_job_msg = None
    for parti in priority:
        msg = await submit_job(
            ctx,
            script_path=script_path,
            partition=parti,
            cancel_if_pending=True,
        )
        if "Try another partition" in msg:
            # The job is canceled due to pending state => try next partition
            continue
        all_pending = False
        submit_job_msg = msg
        break

    if all_pending:
        submit_job_msg = await submit_job(
            ctx,
            script_path=script_path,
            partition=pending_default,
            cancel_if_pending=False,
        )
    assert submit_job_msg, "submit_job_msg is empty"


    if state.slurm_job_id:
        # The job is in RUNNING or PENDING state
        state.task_status = TaskStatus.WAIT
        state.max_wait_minutes = 1440
        await ctx.store.set("agent_state", state)
        return f"{submit_job_msg}.\nStart waiting ..."
    else:
        # The job is immediatly completed or failed, so there is nothing to be waited
        state.task_status = TaskStatus.HOLD
        await ctx.store.set("agent_state", state)
        return submit_job_msg


simple_odin_submit_and_wait_tool = FunctionTool.from_defaults(
    name="submit_and_wait",
    description="""\
Submit a SLURM job to a GPU node. Wait until the job completes (or until log updates are available). Output is saved to slurm-%j.out and slurm-%j.err.
""",
    async_fn=_submit_and_wait_odin_gpu,
    return_direct=True,
)


async def _submit_and_wait_odin_gpu_wait_in_tool(
    ctx: Context,
    script_path: str,
) -> str:
    priority = ["gpu5", "gpu4", "gpu3", "gpu2"]
    pending_default = "gpu5"
    all_pending = True

    submit_job_msg = None
    for parti in priority:
        msg = await submit_job(
            ctx,
            script_path=script_path,
            partition=parti,
            cancel_if_pending=True,
        )
        if "Try another partition" in msg:
            # The job is canceled due to pending state => try next partition
            continue
        all_pending = False
        submit_job_msg = msg
        break

    if all_pending:
        submit_job_msg = await submit_job(
            ctx,
            script_path=script_path,
            partition=pending_default,
            cancel_if_pending=False,
        )
    assert submit_job_msg, "submit_job_msg is empty"
    state = await ctx.store.get("agent_state")

    slurm = await get_slurm_tracker()
    return_str = ""
    while True:
        job = slurm.job_states[state.env_id][state.slurm_job_id]
        job_state = job["state"]
        if job_state not in ["RUNNING", "PENDING"]:
            return_str = f"Job {state.slurm_job_id} is no longer running or pending (state={job_state}). Check the updated files in your working directory and make the next decision."
            state.slurm_job_id = None
            await ctx.store.set("agent_state", state)
            break
        await asyncio.sleep(10)

    state = await ctx.store.get("agent_state")
    state.task_status = TaskStatus.HOLD
    await ctx.store.set("agent_state", state)
    return return_str



simple_odin_submit_and_wait_tool_wait_in_tool = FunctionTool.from_defaults(
    name="submit_and_wait",
    description="""\
Submit a SLURM job to a GPU node. Wait until the job completes (or until log updates are available). Output is saved to slurm-%j.out and slurm-%j.err.
""",
    async_fn=_submit_and_wait_odin_gpu_wait_in_tool,
    return_direct=True,
)

async def _submit_and_wait_kias_gpu(
    ctx: Context,
    script_path: str,
) -> str:
    priority = ["a40", "a100"]
    pending_default = "a40"
    all_pending = True

    submit_job_msg = None
    for parti in priority:
        msg = await submit_job(
            ctx,
            script_path=script_path,
            partition=parti,
            cancel_if_pending=True,
        )
        if "Try another partition" in msg:
            # The job is canceled due to pending state => try next partition
            continue
        all_pending = False
        submit_job_msg = msg
        break

    if all_pending:
        submit_job_msg = await submit_job(
            ctx,
            script_path=script_path,
            partition=pending_default,
            cancel_if_pending=False,
        )
    assert submit_job_msg, "submit_job_msg is empty"

    state = await ctx.store.get("agent_state")

    if state.slurm_job_id:
        # The job is in RUNNING or PENDING state
        state.task_status = TaskStatus.WAIT
        state.max_wait_minutes = 1440
        await ctx.store.set("agent_state", state)
        return f"{submit_job_msg}.\nStart waiting ..."
    else:
        # The job is immediatly completed or failed, so there is nothing to be waited
        state.task_status = TaskStatus.HOLD
        await ctx.store.set("agent_state", state)
        return submit_job_msg


simple_kias_submit_and_wait_tool = FunctionTool.from_defaults(
    name="submit_and_wait",
    description="""\
Submit a SLURM job to a GPU node. Wait until the job completes (or until log updates are available). Output is saved to slurm-%j.out and slurm-%j.err.
""",
    async_fn=_submit_and_wait_kias_gpu,
    return_direct=True,
)


async def _submit_and_wait_kisti_private(
    ctx: Context,
    script_path: str,
) -> str:
    state = await ctx.store.get("agent_state")
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)

    empty_part = None
    for part in ("node1", "node2", "node3", "node4"):
        ret = env.sys_run(f"ssh {part} 'check_gpu'")
        debug(f"CHECK_GPU RET: {ret}")
        if ret.strip() == "0":
            empty_part = part
            break

    assert sub_wd
    wd = env.get_sub_wd_path(sub_wd)
    setup_script = env.get_current_setup_script(venv_name)
    cmd = f"source /home/{env.user_name}/{setup_script} && cd {wd} && bash {script_path}"
    result = env._conn.run(
        f"ssh {empty_part} '{cmd}'", hide=True, warn=True, env=env.envs, timeout=None
    )
    debug_var(result, "RESULT_SLURM")

    job_id = getattr(_submit_and_wait_kisti_private, "job_id", 43821) + 1
    _submit_and_wait_kisti_private.job_id = job_id

    msg_prefix = (
        f"Job submitted. The job id is {job_id}. State: RUNNING\nStart waiting  ..."
    )

    if result.return_code == 0:
        state = "COMPLETED"
    else:
        state = "FAILED"

    break_reason = f"Job {job_id} is no longer running or pending (state={state})."
    poll_msg = f"\nYou're receiving a polling result. Reason: {break_reason}. Check the updated files in your working directory and make the next decision."

    ret_msg = msg_prefix + poll_msg

    # write stdout and stderr
    env.write_file(result.stdout, f"slurm-{job_id}.out", sub_wd=sub_wd)
    env.write_file(result.stderr, f"slurm-{job_id}.err", sub_wd=sub_wd)
    return ret_msg


kisti_private_submit_and_wait_tool = FunctionTool.from_defaults(
    name="submit_and_wait",
    description="""\
Submit a SLURM job to a GPU node. Wait until the job completes (or until log updates are available). Output is saved to slurm-%j.out and slurm-%j.err.
""",
    async_fn=_submit_and_wait_kisti_private,
    return_direct=True,
)

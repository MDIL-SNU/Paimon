from llama_index.core.workflow import Context

from paimon.models import PlanState, SubtaskAgentState
from paimon.world import get_env, Environment


async def get_state(ctx: Context) -> PlanState | SubtaskAgentState:
    st: SubtaskAgentState | None = await ctx.store.get("agent_state", None)
    if st:
        return st
    else:
        plan_st: PlanState = await ctx.store.get_state()  # type: ignore
        return plan_st


async def get_env_with_sub_wd(
    ctx: Context,
) -> tuple[Environment, str | None, str]:
    """Returns environment, sub working directory, and venv name

    Parameters
    ----------
    ctx
        llamaindex context

    Returns
    -------
    environment
        The Environment instance
    sub_wd
        Sub working directory path (or None)
    venv_name
        Virtual environment name to use
    """
    agent_state = await ctx.store.get("agent_state", None)
    if not agent_state:
        plan_state = await ctx.store.get_state()
        env_id = plan_state.env_id
        sub_wd = None
        venv_name = "NONE"  # Planner
    else:
        env_id = agent_state.env_id
        sub_wd = agent_state.sub_wd
        venv_name = agent_state.current_venv

    env = get_env(env_id)
    return env, sub_wd, venv_name

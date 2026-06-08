from random import random
from typing import Any

from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.llms.llm import LLM

import paimon.agent.registry as agent_registry
import paimon.llm
from paimon.models import SubtaskWithDir, SubtaskAgentState
from paimon.world import get_env
from paimon import cfg
from paimon.util.log import debug, warning
from paimon.world.web_search_tools import web_search_tool


async def get_single_agent_workflow_with_context(
    agent_name: str,
    env_id: str,
    subtask: SubtaskWithDir,
    *,
    with_critic: bool = False,
    with_todo_list: bool = False,
    with_web_search: bool = False,
    llm: LLM | str = cfg.default_expert_llm,
    agent_kwargs: dict[str, Any] | None = None,
    original_prompt: str | None = None,
    **workflow_kwargs,
) -> tuple[AgentWorkflow, Context, str]:
    """Get Agent workflow with some llama_index's Context pre-configured

    Parameters
    ----------
    agent_name
        agent name
    env_id
        optional environment id initialized to the Context
    subtask_name
        optional subtask name initialized to the Context
    agent_kwargs
        optional agent kwargs, passed to AgentConfig.get_agent
    original_prompt
        optional initial prompt (if provided, pre_init_hook can modify it)
    workflow_kwargs
        optional workflow kwargs

    Returns
    -------
    AgentWorkflow
        AgentWorkflow instance of llama_index
    Context
        context with env_id, agent_name, subtask attached
    str
        modified prompt (if prompt was provided and hook was registered)
    """
    from paimon.world.common_tools import (
        abort_task_tool,
        complete_task_tool_factory,
    )

    import paimon.world.todo_tools as todo_tools

    agent_kwargs = agent_kwargs or {}

    metadata = {
        "role": "executor",
        "env_id": env_id,
        "agent_name": agent_name,
        "sub_wd": subtask.sub_wd,
    }
    if isinstance(llm, str):
        llm = paimon.llm.get_llm(llm, metadata=metadata)

    complete_task_tool = complete_task_tool_factory(subtask.output_values)

    addi_tools = [complete_task_tool, abort_task_tool]
    addi_system_prompt = agent_kwargs.pop("additional_system_prompt", "")

    if with_todo_list:
        addi_tools.append(todo_tools.generate_todo_list_tool)
        addi_tools.append(todo_tools.mark_item_complete_tool)
        addi_system_prompt += "\n" + todo_tools.AGENT_SYSTEM_PROMPT_TODO

    if with_web_search:
        addi_tools.append(web_search_tool)

    agent_kwargs = agent_kwargs or {}
    agent_config = agent_registry.get_agent_config(agent_name)

    agent = agent_config.get_agent(
        llm=llm,
        interactive_mode=False,
        # Don't understand why LSP complains
        additional_tools=addi_tools,  # type: ignore
        additional_system_prompt=addi_system_prompt,
        as_paimon_agent=True,
        with_critic=with_critic,
        **agent_kwargs,
    )

    agent.list_wd_for_tool_call = True  # type: ignore
    assert agent.system_prompt

    determined_venv = "base"

    agent_wf = AgentWorkflow([agent], **workflow_kwargs)
    state = SubtaskAgentState(
        env_id=env_id,
        agent_name=agent_name,
        system_prompt=agent.system_prompt,
        sub_wd=subtask.sub_wd,
        instruction=subtask.instruction,
        required_output_files=subtask.output_files,
        required_output_values=subtask.output_values,
        current_venv=determined_venv,
    )

    # Apply pre-initialization hook if registered and prompt provided
    modified_prompt = original_prompt or ""  # Default to original prompt
    if agent_config.pre_init_hook and original_prompt:
        debug(f"[subtask_agent] Running pre-init hook for {agent_name}")
        try:
            modified_prompt = await agent_config.pre_init_hook(
                original_prompt, state, subtask
            )
        except Exception as e:
            error_msg = f"Pre-init hook failed for {agent_name}: {str(e)}"
            warning(error_msg)
            raise  # Re-raise for start_subtask to handle

    ctx = Context(workflow=agent_wf)
    await ctx.store.set("agent_state", state)
    return agent_wf, ctx, modified_prompt


async def get_mock_agent_workflow_with_context(
    agent_name: str,
    env_id: str,
    subtask: SubtaskWithDir,
    *,
    success_chance: float = 1.0,
    mock_instruction: str | None = None,
    llm: LLM | str = "fast",
    agent_kwargs: dict[str, Any] | None = None,
    **workflow_kwargs,
) -> tuple[AgentWorkflow, Context]:
    """
    Generate mock agent

    if mock_instruction is not given, be success of fail based on the success_chance
    if mock_instruction is given, ignore success_chance and insert the inst. to agent
    """
    from paimon.world.common_tools import (
        abort_task_tool,
        complete_task_tool_factory,
    )

    assert success_chance >= 0.0 and success_chance <= 1.0
    mock_status = (
        mock_instruction or "success" if random() < success_chance else "give_up"
    )

    complete_task_tool = complete_task_tool_factory(subtask.output_values)

    agent_desc = agent_registry.describe_agent(agent_name)["description"]

    mock_agent_cfg = agent_registry.get_agent_config("Mock agent")
    agent_kwargs = agent_kwargs or {}
    mock_agent = mock_agent_cfg.get_agent(
        llm=llm,
        additional_tools=[complete_task_tool, abort_task_tool],
        **agent_kwargs,
    )
    assert mock_agent.system_prompt is not None
    mock_agent.system_prompt = mock_agent.system_prompt.format(
        agent_name=agent_name,
        agent_desc=agent_desc,
        mock_status=mock_status,
    )
    mock_agent.name = agent_name

    agent_wf = AgentWorkflow([mock_agent], **workflow_kwargs)
    state = SubtaskAgentState(
        env_id=env_id,
        agent_name=agent_name,
        sub_wd=subtask.sub_wd,
        required_output_files=subtask.output_files,
        required_output_values=subtask.output_values,
    )
    ctx = Context(workflow=agent_wf)
    await ctx.store.set("agent_state", state)

    # Generate empty output files
    env = get_env(env_id)
    sub_wd = subtask.sub_wd
    for output_file in subtask.output_files:
        for filename in output_file.enumerate():
            env.sys_run(f"touch {filename}", sub_wd=sub_wd)

    return agent_wf, ctx


# Move this function to somewhere else
async def get_standalone_agent_workflow_with_context(
    agent_name: str,
    env_id: str,
    *,
    with_todo_list: bool = False,
    with_web_search: bool = cfg.web_search_config.interactive,
    llm: LLM | str = cfg.default_expert_llm,
    agent_kwargs: dict[str, Any] | None = None,
    **workflow_kwargs,
) -> tuple[AgentWorkflow, Context]:
    import paimon.world.todo_tools as todo_tools
    agent_kwargs = agent_kwargs or {}

    addi_tools = []
    addi_system_prompt = agent_kwargs.pop("additional_system_prompt", "")
    if with_todo_list:
        addi_tools.append(todo_tools.generate_todo_list_tool)
        addi_tools.append(todo_tools.mark_item_complete_tool)
        addi_system_prompt += "\n" + todo_tools.AGENT_SYSTEM_PROMPT_TODO

    if with_web_search:
        addi_tools.append(web_search_tool)

    agent_config = agent_registry.get_agent_config(agent_name)
    agent = agent_config.get_agent(
        llm=llm,
        interactive_mode=True,
        additional_tools=addi_tools,  # type: ignore
        additional_system_prompt=addi_system_prompt,
        as_paimon_agent=True,
        with_critic=False,
    )  # type: ignore
    agent.tool_required = False  # type: ignore
    agent.list_wd_for_tool_call = True  # type: ignore

    assert agent.system_prompt

    determined_venv = "base"

    agent_wf = AgentWorkflow([agent], **workflow_kwargs)
    state = SubtaskAgentState(
        env_id=env_id,
        agent_name=agent_name,
        system_prompt=agent.system_prompt,
        sub_wd="01_working_directory",
        instruction="",
        required_output_files=[],
        required_output_values=[],
        current_venv=determined_venv,
    )
    agent_ctx = Context(workflow=agent_wf)
    await agent_ctx.store.set("agent_state", state)
    return agent_wf, agent_ctx

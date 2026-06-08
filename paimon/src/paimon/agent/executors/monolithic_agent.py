"""Monolithic agent with all tools from registered agents."""

from typing import Annotated

from llama_index.core.llms.llm import LLM
from llama_index.core.tools import AsyncBaseTool, FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent

from paimon.agent.agent_config import AgentConfig
from paimon.agent.registry import collect_all_tools, get_registry
from paimon.knowledge.library import get_knowledge, get_library
from paimon.agent.paimon_agent import PaimonAgent
from paimon.world.critic import attach_critic_committee
from paimon.world.expert_tools import retrieve_expert_knowledge_tool
from paimon.world.slurm_tools import (
    simple_odin_submit_and_wait_tool,
    simple_kias_submit_and_wait_tool,
    kisti_private_submit_and_wait_tool,
    simple_odin_submit_and_wait_tool_wait_in_tool,
)
from paimon.util.log import debug
from paimon import cfg
import paimon.llm

name = "Monolithic agent"

description = """\
<general>
A universal agent with access to all available tools for atomistic simulations.
</general>

<instruction_requirements>
Accepts any task that specialized agents can handle. Provide clear instructions.
</instruction_requirements>
"""


def _get_domain_expertise() -> str:
    """Compose domain expertise sections from registered agents.

    FF knowledge is delivered dynamically via switch_venv
    at runtime, not baked into the system prompt.
    """
    debug("[_get_domain_expertise] Building domain expertise")
    sections = []
    sections.append("<domain_expertise>")
    sections.append(
        "You have access to expertise from multiple domains. "
        "Refer to the relevant section based on your subtask needs."
    )
    sections.append("")

    registry = get_registry()
    for agent_name, agent_cfg in registry.items():
        # Skip auxiliary agents and the monolithic agent itself
        if agent_cfg.auxiliary or agent_name == name:
            continue

        # Derive canonical name (lowercase, replace spaces with underscores)
        canonical_name = agent_name.lower().replace(" ", "_")
        if agent_name == "LLaMP (Materials Project) agent":
            canonical_name = "llamp_agent"  # Exceptions due to parantheses

        domain_tag = f"{canonical_name}_domain"

        # Try to load tips knowledge for this agent
        try:
            tips = get_knowledge(f"agents/{canonical_name}/tips")
            sections.append(f"<{domain_tag}>")
            sections.append(tips)
            sections.append(f"</{domain_tag}>")
            sections.append("")
        except FileNotFoundError:
            continue

    sections.append("</domain_expertise>")
    return "\n".join(sections)


async def _monolithic_switch_venv(
    ctx: Context,
    venv_name: Annotated[
        str,
        "Target venv name (e.g. 'sevennet', 'mace', 'base')",
    ],
) -> str:
    """Switch venv and return all FF guides for that env.

    Returns every guide for the FF family (ase, lammps, etc.)
    since the monolithic agent handles all domains.
    """
    from paimon.world.common_tools import get_env_with_sub_wd

    env, _, _ = await get_env_with_sub_wd(ctx)
    available = list(env._venv_map.keys())
    if venv_name not in available:
        return f"[Error] Unknown venv: {venv_name}. Available: {available}"

    agent_state = await ctx.store.get("agent_state")
    old_venv = agent_state.current_venv
    agent_state.current_venv = venv_name
    await ctx.store.set("agent_state", agent_state)

    result = f"Switched venv: {old_venv} -> {venv_name}"

    ff_family = venv_name
    prefix = f"forcefield/{ff_family}/"
    library = get_library()
    guides = []
    for key, content in sorted(library.items()):
        debug(f"Checking library key for FF guide: {key}")
        if not key.startswith(prefix):
            continue
        guide_name = key[len(prefix) :]
        if guide_name == "planner":
            continue
        guides.append(f"<{guide_name}_guide>\n{content}\n</{guide_name}_guide>")
        debug(
            f"Added guide for {guide_name} from library key: {key}, content head: {content[:300]}..."
        )
    if guides:
        result += (
            "\n\nThe force field guides for the switched environment are below. "
            "Use these instead of any previous FF instructions.\n\n"
            + "\n\n".join(guides)
        )
    return result


_monolithic_switch_venv_tool = FunctionTool.from_defaults(
    name="switch_venv",
    description="""\
Switch to a different Python environment. Available:
- base: General Python environment with ASE, numpy, etc.
- sevennet: Environment with SevenNet (SevenNetCalculator)
- mace: Environment with MACE (mace_mp, MACECalculator)
Use this before running code that requires a specific ML potential.""",
    async_fn=_monolithic_switch_venv,
)


def _collect_all_task_types() -> list[str]:
    """Collect all unique task types from non-auxiliary agents."""
    task_types_set = set()
    registry = get_registry()

    for agent_name, agent_cfg in registry.items():
        # Skip auxiliary agents and the monolithic agent itself
        if agent_cfg.auxiliary or agent_name == name:
            continue
        if agent_cfg.task_types:
            task_types_set.update(agent_cfg.task_types)

    return sorted(task_types_set)


def _apply_slurm_policy(tools: list[AsyncBaseTool]) -> list[AsyncBaseTool]:
    """Replace submit_and_wait tool based on SLURM policy config."""
    original_count = len(tools)
    tools_without_slurm = [
        t for t in tools if t.metadata.get_name() != "submit_and_wait"
    ]
    removed_count = original_count - len(tools_without_slurm)
    debug(f"[_apply_slurm_policy] Removed {removed_count} submit_and_wait tool(s)")

    slurm_tool = _get_slurm_submit_tool()
    tools_without_slurm.append(slurm_tool)
    debug(
        f"[_apply_slurm_policy] Added {slurm_tool.metadata.get_name()} for policy: {cfg.slurm_policy}"
    )
    return tools_without_slurm


def _get_slurm_submit_tool() -> FunctionTool:
    """Select submit_and_wait tool based on slurm_policy config."""
    if cfg.slurm_policy == "odin":
        debug("SLURM SUBMIT AS ODIN")
        return simple_odin_submit_and_wait_tool
    elif cfg.slurm_policy == "odin_wait_in_tool":
        debug("SLURM SUBMIT AS ODIN WAIT IN TOOL")
        return simple_odin_submit_and_wait_tool_wait_in_tool
    elif cfg.slurm_policy == "kias":
        debug("SLURM SUBMIT AS KIAS")
        return simple_kias_submit_and_wait_tool
    elif cfg.slurm_policy == "kisti_private":
        debug("SLURM SUBMIT AS KISTI_PRIVATE")
        return kisti_private_submit_and_wait_tool
    else:
        raise ValueError(f"Unknown slurm policy: {cfg.slurm_policy}")


class MonolithicAgentConfig(AgentConfig):
    """AgentConfig subclass with two-stage initialization.

    Stage 1 - config() time (during agent discovery):
        - tools=[] (empty, other agents haven't registered yet)
        - system_prompt="" (placeholder, rebuilt in Stage 2)
        - Stores template components (_system_prompt_template, _common,
          _core_guidance, _tool_selection_guide)

    Stage 2 - get_agent() time (at runtime):
        - Collects tools from all registered agents
        - Replaces standard switch_venv with monolithic variant
        - Rebuilds system_prompt using stored template components

    This deferred initialization enables:
        1. Access to all registered agent tools
        2. Template reuse without re-reading knowledge files
    """

    _system_prompt_template: str = ""
    _common: str = ""
    _core_guidance: str = ""
    _tool_selection_guide: str = ""

    def get_agent(
        self,
        llm: LLM | str | None = None,
        additional_tools: list[AsyncBaseTool] | None = None,
        additional_system_prompt: str = "",
        as_paimon_agent: bool = True,
        with_critic: bool = False,
        **kwargs,
    ) -> FunctionAgent:

        llm = llm or self.default_llm
        if isinstance(llm, str):
            llm = paimon.llm.get_llm(llm)

        # Collect tools from all registered agents
        additional_tools = additional_tools or []
        collected_tools = collect_all_tools(omit_agents=[self.name])
        collected_tools = _apply_slurm_policy(collected_tools)

        # Replace standard switch_venv with monolithic variant that returns all FF guides
        collected_tools = [
            t for t in collected_tools if t.metadata.get_name() != "switch_venv"
        ]
        collected_tools.append(_monolithic_switch_venv_tool)

        collected_tools.append(retrieve_expert_knowledge_tool)

        debug(f"[MonolithicAgent] Collected {len(collected_tools)} tools")

        domain_expertise = _get_domain_expertise()
        system_prompt = self._system_prompt_template.format(
            common=self._common,
            core_guidance=self._core_guidance,
            domain_expertise=domain_expertise,
            tool_selection_guide=self._tool_selection_guide,
        )

        # Create agent instance
        cls_ = PaimonAgent if as_paimon_agent else FunctionAgent
        instance = cls_(
            name=self.name,
            description=self.description,
            tools=collected_tools + additional_tools,
            system_prompt=system_prompt + additional_system_prompt,
            llm=llm,
            **kwargs,
        )

        # Attached from paimon.agent.agent_config
        if with_critic:
            debug(
                f"[MonolithicAgent] Attaching critics to: {self.critic_gate_tool_names}"
            )
            assert as_paimon_agent, "monolithic_agent.py"
            assert instance.tools, "monolithic_agent.py"
            tools_tmp = []
            flags_cnt = 0
            for tool in instance.tools:
                if tool.metadata.get_name() in self.critic_gate_tool_names:
                    assert isinstance(tool, FunctionTool), "monolithic_agent.py"
                    tool = attach_critic_committee(tool)
                    setattr(tool, "_paimon_critic_attached", True)
                    flags_cnt += 1
                    debug(
                        f"[MonolithicAgent] Critic attached to: {tool.metadata.get_name()}"
                    )
                tools_tmp.append(tool)
            instance.tools = tools_tmp

            if len(self.critic_gate_tool_names) != flags_cnt:
                raise ValueError(
                    "Some tools are missing to attach critic: "
                    f"{self.critic_gate_tool_names}, {instance.tools}"
                )
            debug(f"[MonolithicAgent] Successfully attached {flags_cnt} critics")

        debug("[MonolithicAgent] Agent instance created successfully")
        return instance


def config() -> AgentConfig:
    """Configuration for the monolithic agent."""
    debug("[config] Building monolithic agent configuration")

    # Compose comprehensive system prompt with all domain expertise
    system_prompt_template = """\
{common}

{core_guidance}

{domain_expertise}

{tool_selection_guide}
"""

    common = get_knowledge("agents/common")
    core_guidance = get_knowledge("agents/monolithic_agent/core")
    tool_selection_guide = get_knowledge(
        "agents/monolithic_agent/tool_selection_guide"
    )
    task_types = _collect_all_task_types()
    config_instance = MonolithicAgentConfig(
        name=name,
        description=description,
        tools=[],  # Collected later at get_agent() time
        system_prompt="",  # Rebuilt at get_agent() time
        task_types=task_types,
        critic_gate_tool_names=["complete_task", "submit_and_wait"],
        auxiliary=True,
        # pre_init_hook=monolithic_pre_init_hook,
    )

    config_instance._system_prompt_template = system_prompt_template
    config_instance._common = common
    config_instance._core_guidance = core_guidance
    config_instance._tool_selection_guide = tool_selection_guide

    return config_instance

"""Agent interface"""

from dataclasses import dataclass, field
from typing import Callable, Awaitable

from llama_index.core.llms.llm import LLM
from llama_index.core.tools import AsyncBaseTool, FunctionTool
from llama_index.core.agent.workflow import FunctionAgent

from paimon.world.critic import attach_critic_committee
from paimon.agent.paimon_agent import PaimonAgent
from paimon.models import TaskTypeLiteral, SubtaskAgentState, SubtaskWithDir
from paimon.knowledge.library import get_knowledge
import paimon.llm
from paimon import cfg


# Type alias for pre-initialization hooks
PreInitHook = Callable[[str, SubtaskAgentState, SubtaskWithDir], Awaitable[str]]


@dataclass(frozen=True)
class AgentConfig:
    """Agent config"""

    name: str
    description: str
    tools: list[AsyncBaseTool]
    system_prompt: str
    critic_gate_tool_names: list[str] = field(
        default_factory=lambda: ["complete_task"]
    )
    task_types: list[TaskTypeLiteral] | None = None
    ff_prompt_key: str | None = None
    default_llm: LLM | str = cfg.default_expert_llm
    auxiliary: bool = False
    pre_init_hook: PreInitHook | None = None

    def get_agent(
        self,
        llm: LLM | str | None = None,
        interactive_mode: bool = False,
        additional_tools: list[AsyncBaseTool] | None = None,
        additional_system_prompt: str = "",
        as_paimon_agent: bool = True,
        with_critic: bool = False,
        **kwargs,
    ) -> FunctionAgent:
        llm = llm or self.default_llm
        if isinstance(llm, str):
            llm = paimon.llm.get_llm(llm)

        additional_tools = additional_tools or []

        # self.system_prompt is str, so this is copy
        system_prompt = self.system_prompt
        common_sys = "agents/common"
        if interactive_mode:
            common_sys = "agents/common_interactive"

        system_prompt = system_prompt.replace(
            "{{common}}",
            get_knowledge(common_sys, with_debug_preamble=cfg.debug_preamble),
        )

        cls_ = PaimonAgent if as_paimon_agent else FunctionAgent
        instance = cls_(
            name=self.name,
            description=self.description,
            tools=self.tools + additional_tools,
            system_prompt=system_prompt + additional_system_prompt,
            llm=llm,
            **kwargs,
        )

        if with_critic:
            assert as_paimon_agent, "agent_config.py"
            assert instance.tools, "agent_config.py"
            tools_tmp = []
            flags_cnt = 0
            for tool in instance.tools:
                if tool.metadata.get_name() in self.critic_gate_tool_names:
                    assert isinstance(tool, FunctionTool), "agent_config.py"
                    tool = attach_critic_committee(tool)
                    setattr(tool, "_paimon_critic_attached", True)
                    flags_cnt += 1
                tools_tmp.append(tool)
            instance.tools = tools_tmp

            if len(self.critic_gate_tool_names) != flags_cnt:
                raise ValueError(
                    f"Some tools are missing to attach critic:"
                    f"{self.critic_gate_tool_names}, {instance.tools}"
                )

        return instance

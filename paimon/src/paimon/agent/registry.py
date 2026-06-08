from pathlib import Path
from importlib import import_module
from typing import Any

from llama_index.core.tools import AsyncBaseTool

from paimon.agent.agent_config import AgentConfig
from paimon.util.log import debug


EXECUTORS_DIR = Path(__file__).parent / "executors"

_AGENT_REGISTRY: dict[str, AgentConfig] = {}


def _discover_agents() -> None:
    for py_file in EXECUTORS_DIR.glob("*.py"):
        if py_file.stem.startswith("_"):
            continue
        
        module_name = f"paimon.agent.executors.{py_file.stem}"
        try:
            mod = import_module(module_name)
            if hasattr(mod, "config"):
                cfg: AgentConfig = mod.config()
                if cfg.name in _AGENT_REGISTRY:
                    raise ValueError(f"Duplicate agent name: {cfg.name}")
                _AGENT_REGISTRY[cfg.name] = cfg
        except Exception as e:
            raise RuntimeError(f"Failed to load {module_name}: {e}") from e


def get_registry() -> dict[str, AgentConfig]:
    if not _AGENT_REGISTRY:
        _discover_agents()
    return _AGENT_REGISTRY


def list_agents(omit_auxiliary: bool = True) -> list[str]:
    """List all agent names available

    Parameters
    ----------
    omit_auxiliary
        omit auxiliary agent which are not desigened to be used directly

    Returns
    -------
    list of agent names
    """
    regi = get_registry()

    ret = []
    for name, agent_cfg in regi.items():
        if omit_auxiliary and agent_cfg.auxiliary:
            continue
        ret.append(name)
    return ret


def get_agent_config(name: str) -> AgentConfig:
    """Get agent config from agent's name

    Parameters
    ----------
    name
        agent name

    Returns
    -------
    AgentConfig
        a class with 'NAME', 'DESCRIPTION', 'TOOLS' attributes and 'get_agent' method
    """
    try:
        return get_registry()[name]
    except KeyError:
        raise ValueError(f"Agent with name '{name}' not found in registry.")


def describe_agent(name: str) -> dict[str, Any]:
    """Returns a dict of agnet description and name.

    Parameters
    ----------
    name
        agent name

    Returns
    -------
    agent dict
        dict with 'name' and 'description' keys
    """
    agent_config = get_agent_config(name)
    return {
        "name": agent_config.name,
        "description": agent_config.description,
        "task_types": agent_config.task_types,
    }


def short_agent_descriptions(names: list[str]) -> str:
    ret = []
    for name in names:
        cfg = get_agent_config(name)
        assert cfg.task_types
        task_types = ", ".join([f"[[{tt}]]" for tt in cfg.task_types])
        ret.append(f"- {name}: {task_types}")
    return "\n".join(ret)


def long_agent_descriptions(names: list[str]) -> str:
    ret = []
    for name in names:
        cfg = get_agent_config(name)
        assert cfg.task_types
        task_types = ", ".join([f"[[{tt}]]" for tt in cfg.task_types])
        desc = cfg.description
        ret.append(f"""\
<{name}>
<task_types>{task_types}</task_types>
<description>
{desc}
</description>
</{name}>
""")
    return "\n".join(ret)


def collect_all_tools(
    omit_auxiliary: bool = True,
    omit_agents: list[str] | None = None,
) -> list[AsyncBaseTool]:
    """Collect all unique tools from registered agents.

    Parameters
    ----------
    omit_auxiliary
        Skip auxiliary agents (debugging, mock)
    omit_agents
        List of agent names to skip

    Returns
    -------
    List of unique tools (deduplicated by name)
    """
    from paimon.world.common_tools import (
        write_file_tool,
        run_python_tool,
        run_bash_tool,
        inspect_h5_tool,
    )

    omit_agents = omit_agents or []
    registry = get_registry()
    seen: dict[str, AsyncBaseTool] = {}

    # Prefer common tools versions for deduplication
    for tool in [write_file_tool, run_python_tool, run_bash_tool, inspect_h5_tool]:
        tool_name = tool.metadata.get_name()
        seen[tool_name] = tool
        debug(f"[collect_all_tools] Added common tool: {tool_name}")

    for name, agent_cfg in registry.items():
        if omit_auxiliary and agent_cfg.auxiliary:
            continue
        if name in omit_agents:
            debug(f"[collect_all_tools] Skipping agent: {name}")
            continue
        debug(f"[collect_all_tools] Processing agent: {name}")
        for tool in agent_cfg.tools:
            tool_name = tool.metadata.get_name()
            if tool_name not in seen:
                seen[tool_name] = tool
                debug(f"[collect_all_tools]   Added tool: {tool_name}")
            else:
                debug(f"[collect_all_tools]   Skipped duplicate: {tool_name}")

    debug(f"[collect_all_tools] Total unique tools collected: {len(seen)}")
    return list(seen.values())

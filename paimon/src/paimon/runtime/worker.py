"""Pre-warmed worker entry point.

This module does all heavy imports at module level so they are ready
when the worker receives its init command via stdin.
"""

import sys
import json
import asyncio

# Pre-warm heavy imports
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAIResponses  # noqa: F401
from llama_index.core.base.llms.types import ChatMessage  # noqa: F401
from llama_index.core.memory import ChatMemoryBuffer  # noqa: F401

from paimon.agent.paimon_agent import PaimonAgent  # noqa: F401
import paimon.world as world  # noqa: F401
from paimon.runtime.interface.web import WebInterface  # noqa: F401
from paimon.models import SubtaskAgentState, Plan, Subtask  # noqa: F401
from paimon.token_sum import TokenSum  # noqa: F401
from paimon.util.artifacts import (  # noqa: F401
    save_global_run_artifacts,
    restore_global_run_artifacts,
)
from paimon.llm.base import get_llm  # noqa: F401
from paimon.workflow.subtask_agent import (  # noqa: F401
    get_standalone_agent_workflow_with_context,
)

from paimon.runtime.runner import (
    runner_single_agent_workflow,
    runner_multi_agent_workflow,
    runner_chat,
)
import paimon.audit as audit
from paimon.util.log import debug, info


def _ensure_env_and_sync_config(env_id: str, config: dict) -> dict:
    """Check whether env_id and folder already exist."""
    is_new = not world.is_id_already_present(env_id)

    # load if alreayd exist. create new (dir) if it doesn't exist
    env_id = world.new_environment(id=env_id)
    env = world.get_env(env_id)

    if is_new:
        merged = dict(config)
        merged["is_new"] = True
    else:
        merged = env.read_json(".config.json")
        merged.update(config)
        merged["is_new"] = False

    env.write_json(merged, ".config.json")
    return merged


async def worker_main() -> None:
    # Signal ready to parent
    sys.stdout.write("ready\n")
    sys.stdout.flush()

    # Wait for init command on stdin
    loop = asyncio.get_event_loop()
    init_json = await loop.run_in_executor(None, sys.stdin.readline)

    if not init_json.strip():
        debug("[worker] Empty init command, exiting")
        return

    init_cmd = json.loads(init_json)
    env_id = init_cmd["env_id"]
    config = init_cmd["config"]
    socket_path = init_cmd["socket_path"]

    debug(f"[worker] Received init: env_id={env_id}")

    config = _ensure_env_and_sync_config(env_id, config)

    agent = config.get("agent")
    workflow_type = config.get("workflow_type", "single")

    info(f"[runner] {'=' * 50}")
    info(f"[runner] Session started")
    info(f"[runner] {'=' * 50}")
    if agent == "Chatting":
        await runner_chat(env_id, config, socket_path)
    elif workflow_type == "orchestrator":
        if not config["is_new"]:
            info("[runner] Resurrection is not supported for orchestrator. Closing.")
            return
        with audit.scope(env_id=env_id, sub_wd=None):
            await runner_multi_agent_workflow(env_id, config, socket_path)
    else:
        with audit.scope(env_id=env_id, sub_wd="01_working_directory"):
            await runner_single_agent_workflow(env_id, config, socket_path)

    debug("[worker] Close completely")


if __name__ == "__main__":
    asyncio.run(worker_main())

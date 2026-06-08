import json
from datetime import datetime
from typing import Any
import warnings

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

from paimon.util.chat import dump_chat
from paimon.token_sum import TokenSum
from paimon.world import get_env
from paimon.world.environment import Environment
from paimon.util.log import debug
import paimon.audit as audit


def save_global_run_artifacts(
    env: Environment | str | None,
    *,
    metadata: dict[str, Any] | None = None,
    chat: list[ChatMessage] | None = None,
    tokens: TokenSum | None = None,
    last_event: Any | None = None,
    text_artifacts: dict[str, str] | None = None,
    json_artifacts: dict[str, Any] | None = None,
) -> None:
    """
    Save artifacts with canonical file name in environment.
    """
    if not env:  # env is not initialized
        debug("[save_global_run_artifacts] env is not initialized but called")
        return

    if isinstance(env, str):
        env = get_env(env)
    now = datetime.now()

    if metadata is not None:
        metadata.update({"env_id": env.id})
        env.write_json(metadata, ".globals.json")

    if chat is not None:
        env.write_json(dump_chat(chat), ".chat.json")

    if tokens is not None:
        token_data = tokens.to_dict()
        env.write_json(token_data, ".token.json")

    if last_event is not None:
        if isinstance(last_event, dict):
            event_entry = {"time": now.isoformat(), **last_event}
        else:
            event_entry = {"time": now.isoformat(), "name": last_event.__class__.__name__}
        env.append_json(key="events", value=event_entry, filename=".event.json")

    if text_artifacts:
        for filename, content in text_artifacts.items():
            if content:
                env.write_file(content, filename)

    if json_artifacts:
        for filename, data in json_artifacts.items():
            if data is None:
                continue
            content = data.model_dump() if hasattr(data, "model_dump") else data
            env.write_json(content, filename)

    audit.flush()


def restore_global_run_artifacts(env: Environment):
    try:
        chat_data = env.read_json(".chat.json")
        chat = ChatMemoryBuffer.from_dict(chat_data).get_all()
    except Exception as e:
        raise RuntimeError(
            f"Cannot restore run {env.id}: chat missing/unreadable ({e})"
        )

    try:
        metadata = env.read_json(".globals.json")
    except Exception as e:
        warnings.warn(f"globals.json load file, {env.id}: {e}")
        metadata = None

    try:
        token_data = env.read_json(".token.json")
        tokens = TokenSum.from_dict(token_data)
    except Exception as e:
        warnings.warn(f"Token restore failed; init zeros for run {env.id}: {e}")
        tokens = TokenSum(items=[])

    last_event = None
    try:
        event_data = env.read_json(".event.json")
        events = event_data["events"]
        last_event = events[-1] if events else None
    except Exception as e:
        warnings.warn(f"last event load fail, {env.id}: {e}")

    return {
        "chat": chat,
        "metadata": metadata,
        "tokens": tokens,
        "last_event": last_event,
    }


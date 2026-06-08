from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from paimon_web.util.log import debug


def parse_chat_json(chat_data: dict) -> list[ChatMessage]:
    """
    Parse chat JSON using ChatMemoryBuffer.

    Parameters
    ----------
    chat_data : dict
        Chat data loaded from .chat.json or .agent_memory.json

    Returns
    -------
    list[ChatMessage]
        List of ChatMessage objects with blocks, filtered to exclude system/developer messages
    """
    debug("[chat] Parsing chat JSON")
    try:
        buffer = ChatMemoryBuffer.from_dict(chat_data)
        messages = buffer.get_all()

        # Filter out system and developer messages
        filtered_messages = [
            msg
            for msg in messages
            if msg.role not in (MessageRole.SYSTEM, MessageRole.DEVELOPER)
        ]

        debug(f"[chat] Parsed {len(filtered_messages)} messages")
        return filtered_messages
    except Exception as e:
        debug(f"[chat] ChatMemoryBuffer parsing failed: {e}, using fallback")
        # Fallback returns empty list as we can't properly construct ChatMessage objects
        debug(
            "[chat] Fallback: returning empty list (cannot construct ChatMessage without proper structure)"
        )
        return []

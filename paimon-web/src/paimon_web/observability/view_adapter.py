import json
import re
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock


def strip_user_msg_tags(events: list[dict]) -> list[dict]:
    """Strip system-appended tags from user_msg content for display.

    The runner appends metadata like <wd_snapshot> and
    <user provided files> to user messages before saving.
    These should not be shown in the UI.
    """
    result = []
    for ev in events:
        if ev.get("name") == "user_msg" and "content" in ev:
            ev = {
                **ev,
                "content": strip_tagged_blocks(
                    ev["content"],
                    exceptions=["task", "user_query"],
                ),
            }
        result.append(ev)
    return result


def strip_tagged_blocks(text: str, exceptions: list[str] | None = None) -> str:
    if exceptions is None:
        exceptions = []

    pattern = re.compile(
        r"""
        <\s*(?P<tag_open>[^>]*)\s*>      # opening tag: allow spaces in tag name
        (?P<content>.*?)                 # content
        </\s*(?P<tag_close>[^>]*)\s*>    # closing tag: allow spaces in tag name
        """,
        re.DOTALL | re.VERBOSE,
    )

    def replacer(match: re.Match) -> str:
        tag_open = match.group("tag_open").strip()
        tag_close = match.group("tag_close").strip()
        content = match.group("content")
        if tag_open != tag_close:
            return match.group(0)
        if content.strip() == "":
            return ""

        return content if tag_open in exceptions else ""
    result = pattern.sub(replacer, text)
    result = re.sub(r"\n\s*\n+", "\n", result)

    return result.strip()


def _format_special_field(tool_name: str, field_name: str, value: Any) -> str:
    """
    Format special fields that contain code, instructions, or long content.
    These fields often have newlines/tabs that get collapsed in JSON.
    """
    if not isinstance(value, str):
        return value

    # Special handling for specific tool+field combinations
    if tool_name == "run_python" and field_name == "code":
        # Python code should be displayed as-is
        return value
    elif tool_name == "create_subtask" and field_name == "instruction":
        # Subtask instructions should preserve formatting
        return value
    elif field_name == "content":
        # Generic 'content' fields should preserve formatting
        return value

    return value


def _format_tool_kwargs(tool_name: str, kwargs: dict | str) -> str:
    """
    Format tool kwargs with special handling for fields with newlines/tabs.
    """
    # Parse kwargs if it's a string
    if isinstance(kwargs, str):
        try:
            kwargs = json.loads(kwargs)
        except Exception:
            # If parsing fails, return as-is
            return kwargs

    if not isinstance(kwargs, dict):
        return str(kwargs)

    # Extract and format special fields separately
    special_fields = {}
    regular_fields = {}

    for key, value in kwargs.items():
        # Check if this is a special field we want to format differently
        if (tool_name == "run_python" and key == "code") or \
           (tool_name == "create_subtask" and key == "instruction") or \
           (key == "content"):
            special_fields[key] = _format_special_field(tool_name, key, value)
        else:
            regular_fields[key] = value

    # Build formatted output
    result_parts = []

    # Add regular fields as pretty JSON
    if regular_fields:
        try:
            regular_json = json.dumps(regular_fields, indent=2, ensure_ascii=False)
            result_parts.append(regular_json)
        except Exception:
            result_parts.append(str(regular_fields))

    # Add special fields with preserved formatting
    for key, value in special_fields.items():
        if result_parts:
            result_parts.append("")  # Add blank line separator
        result_parts.append(f"#~# {key}:")
        result_parts.append(value)

    return "\n".join(result_parts)


def chat(msgs: list[ChatMessage]) -> list[ChatMessage]:
    """
    Adapt chat messages for better display in web interface.
    Formats tool call kwargs with special handling for code and content fields.
    """
    for msg in msgs:
        new_blocks = []
        for block in msg.blocks:
            if isinstance(block, ToolCallBlock):
                # Format tool kwargs with special handling
                block.tool_kwargs = _format_tool_kwargs(
                    block.tool_name, block.tool_kwargs
                )
            new_blocks.append(block)
        msg.blocks = new_blocks
    return msgs


def extract_trajectory(msgs: list[ChatMessage]) -> list[dict]:
    """
    Extract tool call trajectory from chat messages.
    Returns list of {step, tool_name, summary} dicts.
    """
    trajectory = []
    step = 0

    for msg in msgs:
        for block in msg.blocks:
            if isinstance(block, ToolCallBlock):
                step += 1
                # Extract brief summary from kwargs
                summary = _get_tool_summary(block.tool_name, block.tool_kwargs)
                trajectory.append({
                    "step": step,
                    "tool_name": block.tool_name,
                    "summary": summary,
                })

    return trajectory


def _get_tool_summary(tool_name: str, kwargs: dict | str) -> str:
    """Get brief summary of tool call for trajectory display."""
    if isinstance(kwargs, str):
        try:
            kwargs = json.loads(kwargs)
        except Exception:
            return ""

    if not isinstance(kwargs, dict):
        return ""

    # Tool-specific summaries for known tools
    if tool_name == "run_python":
        code = kwargs.get("code", "")
        if code:
            first_line = code.split("\n")[0][:40]
            return first_line + "..." if len(code) > 40 else first_line
    elif tool_name == "create_subtask":
        return _truncate(kwargs.get("subtask_name", ""), 30)
    elif tool_name == "complete_task":
        return "done"
    elif tool_name == "abort_task":
        return _truncate(kwargs.get("reason", ""), 30)
    elif tool_name == "run_bash":
        return _truncate(kwargs.get("command", ""), 40)
    elif tool_name == "write_file":
        return _truncate(kwargs.get("filename", ""), 30)
    elif tool_name == "read_file":
        return _truncate(kwargs.get("filename", ""), 30)

    # Generic fallback: try common field names first
    common_fields = [
        "name", "filename", "path", "command", "query", "message",
        "target", "source", "key", "id", "url", "action",
    ]
    for field in common_fields:
        if field in kwargs and isinstance(kwargs[field], str):
            return _truncate(kwargs[field], 30)

    # Last resort: first string value
    for v in kwargs.values():
        if isinstance(v, str) and v:
            return _truncate(v, 30)

    return ""


def _truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis if needed."""
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    return s[:max_len] + "..." if len(s) > max_len else s


def workspace_chat(msgs: list[ChatMessage]) -> list[dict]:
    """
    Extract simple user/assistant turns for workspace chat panel display.
    Returns list of messages with structured actions for unified rendering.

    Merges consecutive assistant messages into one (tool calls are often
    stored as separate messages in llama-index).
    """
    result = []
    for msg in msgs:
        role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if role not in ("user", "assistant"):
            continue

        # Extract text content and tool calls
        text_parts = []
        actions = []
        for block in msg.blocks:
            if hasattr(block, "block_type"):
                if block.block_type == "text" and hasattr(block, "text"):
                    text_parts.append(block.text)
                elif block.block_type == "tool_call":
                    actions.append({
                        "type": "tool",
                        "name": block.tool_name,
                        "status": "ok",  # Legacy format has no status info
                    })
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        content = "\n".join(text_parts).strip()
        if not content and not actions:
            continue

        # Merge consecutive assistant messages
        if role == "assistant" and result and result[-1]["role"] == "assistant":
            if actions:
                result[-1]["actions"].extend(actions)
            if content:
                prev = result[-1]["content"]
                new = f"{prev}\n{content}".strip() if prev else content
                result[-1]["content"] = new
        else:
            files = msg.additional_kwargs.get("files", [])
            result.append({
                "role": role,
                "content": strip_tagged_blocks(content, ["task", "user_query"]),
                "actions": actions,
                "files": files,
            })

    return result


# TODO: NOT USED
def events_to_workspace_chat(events: list[dict]) -> list[dict]:
    """Convert events to workspace chat format with structured actions.

    Returns list of messages where each message has:
    - role: "user" | "assistant"
    - content: str
    - files: list[str]
    - actions: list[dict] - structured actions for rendering:
        - {"type": "tool", "name": str, "status": "ok"|"err"|"pending"}
        - {"type": "subtask", "name": str, "agent": str, "status": str}
    """
    chat: list[dict] = []
    current_assistant: dict | None = None
    # Track pending tools/subtasks to update status when result arrives
    pending_tools: dict[str, dict] = {}  # tool_name -> item dict
    pending_subtask: dict | None = None

    def flush_assistant() -> None:
        nonlocal current_assistant, pending_tools, pending_subtask
        if current_assistant and (
            current_assistant["content"] or current_assistant["actions"]
        ):
            chat.append(current_assistant)
        current_assistant = None
        pending_tools = {}
        pending_subtask = None

    for ev in events:
        name = ev.get("name")

        if name == "user_msg":
            flush_assistant()
            chat.append({
                "role": "user",
                "content": ev.get("content", ""),
                "files": ev.get("files", []),
                "actions": [],
            })
            current_assistant = {
                "role": "assistant",
                "content": "",
                "actions": [],
                "files": [],
            }

        elif name == "ToolCall" and current_assistant is not None:
            tool_name = ev.get("tool", "")
            item = {
                "type": "tool",
                "name": tool_name,
                "status": "pending",
            }
            current_assistant["actions"].append(item)
            pending_tools[tool_name] = item

        elif name in (
            "ToolCallResult",
            "SimpleToolCallResultEvent",
        ) and current_assistant is not None:
            tool_name = ev.get("tool", "")
            status = "ok" if ev.get("success", True) else "err"
            if tool_name in pending_tools:
                pending_tools[tool_name]["status"] = status

        elif name == "AgentOutput" and current_assistant is not None:
            content = ev.get("content", "")
            if content:
                current_assistant["content"] += content

        elif (
            name == "InputRequiredWithStepEvent"
            and current_assistant is not None
        ):
            question = ev.get("question", "")
            if question:
                current_assistant["content"] += question

        elif name == "StartSubtask" and current_assistant is not None:
            subtask_name = ev.get("subtask_name", "")
            agent = ev.get("agent", "")
            item = {
                "type": "subtask",
                "name": subtask_name,
                "agent": agent,
                "status": "running",
            }
            current_assistant["actions"].append(item)
            pending_subtask = item

        elif name == "SubtaskSuccess" and current_assistant is not None:
            if pending_subtask:
                pending_subtask["status"] = "success"
            pending_subtask = None

        elif name == "SubtaskFail" and current_assistant is not None:
            if pending_subtask:
                pending_subtask["status"] = "fail"
            pending_subtask = None

        elif name == "TaskComplete" and current_assistant is not None:
            report = ev.get("report", "")
            if report:
                current_assistant["content"] += report

        elif name == "TaskFail" and current_assistant is not None:
            excuse = ev.get("excuse", "")
            current_assistant["content"] += f"Failed: {excuse}"

    flush_assistant()
    return chat

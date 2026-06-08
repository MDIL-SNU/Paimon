import json
from datetime import datetime
from typing import Any

from pydantic import TypeAdapter
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

from paimon.models import TodoItem
from paimon.util.context import get_env_with_sub_wd


AGENT_SYSTEM_PROMPT_TODO = """\
<todo-protocol>

<ordering>
You must call the `generate_todo_list` tool first before performing any other operation.
Each TODO item must represent a **single, minimal subgoal** that can be tested or verified independently.
Avoid grouping multiple logical operations or large workflows into one step.
Do not call `complete_task` tool until every TODO item is successfully marked with `mark_item_complete`.
</ordering>

<generate_todo_list tool usage>
IMPORTANT: You are the one who must design the TODO list yourself.
The `generate_todo_list` tool does not automatically create tasks for you.
You must construct the `todo_list` argument manually and then call the tool to record it.

The TODO list is sequential. Each item has a numeric id starting from 1, a concise description, and a comma-separated list of the tools you plan to use for that step.

Example: 
generate_todo_list(
{
    todo_list: [
        {"id": 1, "desc": "Locate input files", "tools": "run_bash"},
        {"id": 2, "desc": "Open and parse files", "tools": "run_bash, run_python"},
        {"id": 3, "desc": "Compute results from parsed data", "tools": "run_python"}
    ]
})

Each item must declare at least one tool.
You will be penalized if perform multiple TODO steps in a single code block or you skip tools you declared. 
If any system rule constrains the method/tooling for a step, the corresponding TODO item MUST include those tools.
</generate_todo_list tool usage>

<mark_item_complete tool usage>
Use `mark_item_complete` in ascending id order with no gaps.
For each completed item, include a brief `note` that summarizes the key result obtained, and explicitly confirm which tool(s) were used.
</mark_item_complete tool usage>

<tool usage validation>
Each TODO item must **actually use** all the tool(s) listed for that step.
If the expected tool is not called, the step is considered incomplete.
Perform one logical action per call; do not combine multiple TODO items into one tool execution.
</tool usage validation>

<regeneration>
If new information requires revising later steps, regenerate the list from a given id using
`generate_todo_list update_from="k"`. Items with id < k remain, and items with id ≥ k are replaced.
Note that ids are 1-indexed.
</regeneration>

<completion-guard>
Attempting to call `complete_task` before all TODO items are marked will result in an error.
Finish the list first. If you find it is impossible to accomplish the task, remember that you have the abort_task tool.
</completion-guard>

</todo-protocol>
"""  # noqa: E501


def _now() -> str:
    return datetime.now().isoformat(timespec="minutes")


def _render_status(todo: dict[str, Any]) -> str:
    subs: list[TodoItem] = todo.get("todo_list", [])
    if not subs:
        return "The TODO list is empty"

    completed: dict[int, str] = todo.get("completed", {})
    lines = [f"TODO status  (total: {len(subs)}, completed: {len(completed)})", ""]
    for todo_item in subs:
        is_complete = todo_item.id in completed
        mark = "[O]" if is_complete else "[ ]"
        lines.append(
            f"{mark} {todo_item.id}. {todo_item.desc} (tools: {todo_item.tools})"
        )
        if is_complete:
            lines.append(f"    note: {completed[todo_item.id]}")
    return "\n".join(lines)


async def generate_todo_list(
    ctx: Context, todo_list: list[dict[str, str]], update_from: int | None = None
) -> str:
    if not todo_list:
        return "The todo_list is empty"

    todo = await ctx.store.get("todo", {})

    # TODO: add id ascending check, tool existence check
    if update_from is not None and update_from < 1:
        return "update_from must be bigger than 0"

    ta = TypeAdapter(list[TodoItem])
    todo_list_parsed = ta.validate_python(todo_list)

    now = _now()
    if "todo_list" not in todo or update_from is None:
        todo["todo_list"] = todo_list_parsed
        todo["completed"] = {}  # dict[int, str], id & note
        todo["created_at"] = now
        todo["updated_at"] = now
        # TODO: Below line is wrong (re-generate full resets history)
        todo["history"] = [
            {"ts": now, "event": "generate_full", "count": len(todo_list_parsed)}
        ]
    else:
        keep = [x for x in todo["todo_list"] if x.id < update_from]
        if not keep[-1].id + 1 == todo_list_parsed[0].id:
            return "Update failed. IDs are not in consecutive order"

        todo["todo_list"] = keep + todo_list_parsed
        todo["completed"] = {
            id: note for id, note in todo["completed"].items() if id < update_from
        }
        todo["updated_at"] = now
        todo.setdefault("history", []).append(
            {"ts": now, "event": "generate_partial", "from_id": update_from}
        )

    await ctx.store.set("todo", todo)

    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    env.write_json(todo, filename=".todo_state.json", sub_wd=sub_wd)

    return _render_status(todo)


generate_todo_list_tool = FunctionTool.from_defaults(
    name="generate_todo_list",
    description="""\
Generate or partially update a sequential TODO list for the current subtask. 
Each item must include its numeric id, a short description, and a comma-separted list of tools that will be used. If update_from is given, keep earlier items and rplace latr ones. 
This tool must always be called first before any other operation. 
State is saved internally and mirrored to hidden files '.todo_stat.json' and'.todo_hitory.jso'.
    """,  # noqa: E501
    async_fn=generate_todo_list,
)


async def mark_item_complete(ctx: Context, item_id: int, note: str) -> str:
    todo = await ctx.store.get("todo", {})

    subs: list[TodoItem] = todo.get("todo_list", [])
    ids = [x.id for x in subs]

    if not subs:
        return "ERROR: no TODO list found. Call generate_todo_list first."
    if item_id not in ids:
        return f"ERROR: item_id {item_id} not found in current list."

    completed: dict[int, str] = todo.get("completed", {})
    if item_id in completed:
        return f"ERROR: item_id {item_id} already completed."

    expected = ids[0] if not completed else max(completed) + 1
    if item_id != expected:
        return f"ERROR: invalid order. Next required id is {expected}."

    completed[item_id] = note
    todo["completed"] = completed

    now = _now()
    todo.setdefault("history", []).append(
        {"ts": now, "event": "mark", "id": item_id, "note": note}
    )
    todo["updated_at"] = now
    await ctx.store.set("todo", todo)

    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    env.write_json(todo, filename=".todo_state.json", sub_wd=sub_wd)

    return _render_status(todo)


mark_item_complete_tool = FunctionTool.from_defaults(
    name="mark_item_complete",
    description="""\
Mark a TODO item as completed. Input fields: item_id (integer id) and note (short summary of what was achieved).
Marking must follow the exact sequential order of ids with no gaps.
""",  # noqa: E501
    async_fn=mark_item_complete,
)

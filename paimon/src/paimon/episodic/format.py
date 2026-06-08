"""Format episodic examples for prompt injection."""

import random
from typing import Literal

from .loader import load_all_for_expert, resolve_example
from .models import SubtaskExample


FLOW_HINT_ENTRY = """\
<subtask task_number="{task_number}" agent="{agent}">
{description}
</subtask>
"""

FEWSHOT_SCRIPT = """\
<script filename="{filename}">
{content}
</script>
"""

FEWSHOT_ENTRY = """\
<example subtask="{subtask_name}">
<summary>
{summary}
</summary>
{scripts}</example>
"""


def _select_trajectory(
    trajs: dict[str, list[SubtaskExample]],
    select: Literal["first", "random"],
) -> tuple[str, list[SubtaskExample]]:
    """Return (trajectory_id, examples) based on selection strategy."""
    if select == "first":
        tid = next(iter(trajs))
    elif select == "random":
        tid = random.choice(list(trajs.keys()))
    else:
        raise ValueError(f"Unknown select strategy: {select}")

    return tid, trajs[tid]


def format_flow_hints(
    examples: list[SubtaskExample],
    expert_knowledge: str,
    trajectory_id: str,
) -> str:
    """Format examples as a compact flow summary for the orchestrator.

    Shows subtask sequence with descriptions only (no scripts).
    Includes expert_knowledge key and trajectory_id so the planner
    can construct example_ids.
    Returns empty string if examples is empty.
    """
    if not examples:
        return ""

    entries = ""
    for ex in examples:
        entries += FLOW_HINT_ENTRY.format(
            task_number=ex.task_number,
            agent=ex.agent,
            description=ex.description,
        )

    example_id = f"{expert_knowledge}/{trajectory_id}:{examples[0].task_number}"

    return (
        f'<prior_successful_workflow expert_knowledge="{expert_knowledge}"'
        f' trajectory_id="{trajectory_id}">\n'
        f"{entries}"
        f"To reference a subtask as a few-shot example, use example_ids "
        f'with format: "{example_id}"\n'
        "</prior_successful_workflow>"
    )


def format_fewshot_examples(examples: list[SubtaskExample]) -> str:
    """Format examples with summaries and working scripts for agent prompts.

    Returns empty string if examples is empty.
    """
    if not examples:
        return ""

    entries = ""
    for ex in examples:
        scripts = ""
        for fname, content in ex.working_scripts.items():
            scripts += FEWSHOT_SCRIPT.format(filename=fname, content=content)

        entries += FEWSHOT_ENTRY.format(
            subtask_name=ex.subtask_name,
            summary=ex.summary,
            scripts=scripts,
        )

    return (
        "\n<reference_examples>\n"
        f"{entries}"
        "</reference_examples>"
    )


# -- Convenience functions (load + format) ----------------------------------


def get_flow_hints_for_expert(
    expert_knowledge: str,
    select: Literal["first", "random"] = "random",
) -> str:
    """Load episodic examples for an expert knowledge and return formatted flow hints.

    Parameters
    ----------
    expert_knowledge
        Expert knowledge key (e.g. "sim_liquid_electrolyte")
    select
        Strategy to pick among multiple trajectories
    """
    all_trajs = load_all_for_expert(expert_knowledge)
    if not all_trajs:
        return ""

    trajectory_id, examples = _select_trajectory(
        {tid: t.subtask_examples for tid, t in all_trajs.items()},
        select,
    )
    return format_flow_hints(examples, expert_knowledge, trajectory_id)


def get_fewshot_prompt(
    expert_knowledge: str,
    trajectory_id: str,
    task_number: int,
) -> str:
    """Resolve a single example by its coordinates and return formatted fewshot block."""
    example = resolve_example(expert_knowledge, trajectory_id, task_number)
    return format_fewshot_examples([example])
